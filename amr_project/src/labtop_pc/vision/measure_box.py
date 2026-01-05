# src/labtop_pc/vision/measure_box.py
from __future__ import annotations
from typing import Optional, Dict, Any
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ✅ 설정 파일 불러오기
from .vision_config import DEFAULT_VISION_CONFIG as cfg

# =========================================================
# Helper Functions (수학 계산용 - 그대로 유지)
# =========================================================
def poly_shrink_towards_center(poly4x2: np.ndarray, margin_px: float):
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    v = p - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return p - (v / norm) * margin_px

def depth_roi_stats(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray):
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    d = d[(d > 0) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)]
    if d.size == 0: return 0.0, 0.0, 0
    return float(np.median(d)), float(np.median(np.abs(d - float(np.median(d))))), int(d.size)

def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z_m: float):
    X = (cx - intr.ppx) / intr.fx * Z_m
    Y = (cy - intr.ppy) / intr.fy * Z_m
    return float(X), float(Y)

def obb_angle_deg_upright0_rightplus(poly4x2: np.ndarray) -> float:
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    q = p - c
    cov = np.cov(q.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    vx, vy = float(v[0]), float(v[1])
    if vy < 0: vx, vy = -vx, -vy
    return -float(np.degrees(np.arctan2(vx, vy)))

def is_jump(prev, cur):
    if prev is None: return False
    if abs(cur["move_x_mm"] - prev["move_x_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_y_mm"] - prev["move_y_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_z_mm"] - prev["move_z_mm"]) > cfg.jump_z_mm: return True
    return False

# ============================================================
# Vision Class (D405 최적화 - 단순 루프)
# ============================================================
class _VisionRuntime:
    def __init__(self):
        self._running = False
        self._thread = None
        self._model = None
        
        # 데이터 공유 변수
        self._latest_sample = None
        self._collect_samples = []
        self._is_measuring = False

    def start(self):
        if self._running: return
        
        print(f"[Vision] YOLO 모델 로딩... ({cfg.model_path})")
        try:
            self._model = YOLO(cfg.model_path)
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Vision] D405 시스템 시작")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        print("[Vision] 시스템 종료")

    def measure_avg(self, n=10, timeout=5.0):
        if not self._running: return None
        if n is None: n = 10
        if timeout is None: timeout = 5.0

        self._collect_samples = []
        self._is_measuring = True
        
        print(f"[Vision] 측정 시작 (목표: {n}개)")
        start = time.time()
        while time.time() - start < timeout:
            if len(self._collect_samples) >= n:
                break
            time.sleep(0.05)
        
        self._is_measuring = False
        
        if len(self._collect_samples) == 0:
            print("[Vision] 측정 실패: 데이터 없음")
            return None
            
        # 평균 계산
        arr = np.array([[s["move_x_mm"], s["move_y_mm"], s["move_z_mm"], s["angle_deg"]] for s in self._collect_samples])
        m = np.mean(arr, axis=0)
        return {
            "move_x_mm": float(m[0]), "move_y_mm": float(m[1]), "move_z_mm": float(m[2]),
            "angle_deg": float(m[3])
        }

    def _loop(self):
        # 1. 파이프라인 설정
        pipeline = rs.pipeline()
        config = rs.config()
        
        # [중요] D405 대역폭 이슈 해결을 위해 848x480으로 고정 (가장 안정적)
        # 1280x720을 둘 다 켜면 USB가 못 버틸 수 있습니다.
        STREAM_W, STREAM_H, FPS = 848, 480, 30
        
        config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)

        try:
            profile = pipeline.start(config)
        except Exception as e:
            print(f"[ERROR] 리얼센스 시작 실패: {e}")
            self._running = False
            return

        # D405는 어차피 같은 렌즈지만, 데이터 구조 정렬을 위해 align 사용
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # 필터
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()

        if cfg.show_preview:
            cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)

        prev_valid = None

        while self._running:
            try:
                # [핵심] 아까 잘 되던 코드처럼 무한 대기 (Wait)
                frames = pipeline.wait_for_frames()
                
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # 데이터 변환
                img = np.asanyarray(color_frame.get_data())
                
                # 깊이 필터
                d_frame = spatial.process(depth_frame)
                d_frame = temporal.process(d_frame)
                d_u16 = np.asanyarray(d_frame.get_data())
                
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

                vis = img.copy()

                # YOLO 추론
                results = self._model.predict(img, imgsz=cfg.imgsz, conf=cfg.conf_thres, verbose=False)
                r = results[0]

                best_sample = None

                if hasattr(r, 'obb') and r.obb is not None:
                    candidates = []
                    for i, conf in enumerate(r.obb.conf):
                        if conf < cfg.conf_thres: continue
                        
                        poly = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                        
                        # ROI 축소
                        poly_s = poly_shrink_towards_center(poly, 5)
                        # 좌표 클램핑 (해상도에 맞게)
                        poly_s[:, 0] = np.clip(poly_s[:, 0], 0, STREAM_W-1)
                        poly_s[:, 1] = np.clip(poly_s[:, 1], 0, STREAM_H-1)
                        
                        # 깊이 계산
                        z_m, mad, count = depth_roi_stats(d_u16, depth_scale, poly_s)

                        if z_m > 0 and count > 50: # 픽셀 수 기준 완화
                            angle = obb_angle_deg_upright0_rightplus(poly)
                            candidates.append({'z_m': z_m, 'poly': poly, 'angle': angle})

                    if candidates:
                        candidates.sort(key=lambda x: x['z_m'])
                        best = candidates[0]
                        
                        cx, cy = np.mean(best['poly'][:, 0]), np.mean(best['poly'][:, 1])
                        tx, ty = XY_from_pixel_and_Z(cx, cy, intr, best['z_m'])
                        
                        raw = {
                            "move_x_mm": tx * 1000.0,
                            "move_y_mm": ty * 1000.0,
                            "move_z_mm": best['z_m'] * 1000.0,
                            "angle_deg": best['angle']
                        }

                        # 점프 필터
                        if not is_jump(prev_valid, raw):
                            prev_valid = raw
                            best_sample = raw
                            txt = f"XYZ: {raw['move_x_mm']:.0f}, {raw['move_y_mm']:.0f}, {raw['move_z_mm']:.0f}"
                            cv2.polylines(vis, [np.int32(best['poly'])], True, (0, 255, 0), 2)
                            cv2.putText(vis, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.polylines(vis, [np.int32(best['poly'])], True, (0, 0, 255), 2)
                            cv2.putText(vis, "JUMP", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 데이터 수집 (메인 스레드 요청 시)
                if self._is_measuring and best_sample:
                    self._collect_samples.append(best_sample)
                    cv2.putText(vis, f"Collecting: {len(self._collect_samples)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # 화면 출력
                if cfg.show_preview:
                    cv2.imshow(cfg.preview_win_name, vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            except Exception as e:
                # 에러나도 죽지 않고 재시도 (아까 잘되던 코드 방식)
                # print(f"[WARN] 루프 에러: {e}") 
                time.sleep(0.01)

        pipeline.stop()
        if cfg.show_preview:
            cv2.destroyAllWindows()

_RT = _VisionRuntime()

def start_stream(): _RT.start()
def stop_stream(): _RT.stop()
def measure_avg(n=None, timeout_sec=None): return _RT.measure_avg(n, timeout_sec)
def is_running(): return _RT._running