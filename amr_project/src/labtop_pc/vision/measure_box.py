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
# Helper Functions (수학 및 전처리)
# =========================================================
def poly_shrink_towards_center(poly4x2: np.ndarray, margin_px: float):
    """박스 테두리 노이즈 제거를 위해 중심 쪽으로 축소"""
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    v = p - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return p - (v / norm) * margin_px

def depth_roi_stats(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray):
    """축소된 박스 내부의 깊이값 중앙값(Median) 계산"""
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
    
    # 마스크 영역 내 깊이 추출
    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    
    # 유효 범위 필터링
    d = d[(d > 0) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)]
    
    if d.size == 0: return 0.0, 0.0, 0
    return float(np.median(d)), float(np.median(np.abs(d - float(np.median(d))))), int(d.size)

def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z_m: float):
    """2D 픽셀 좌표 + 깊이(Z) -> 3D 공간 좌표(X, Y) 변환"""
    X = (cx - intr.ppx) / intr.fx * Z_m
    Y = (cy - intr.ppy) / intr.fy * Z_m
    return float(X), float(Y)

def obb_angle_deg_upright0_rightplus(poly4x2: np.ndarray) -> float:
    """PCA를 이용한 OBB 주축 각도 계산"""
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
    """이전 값과 비교하여 너무 크게 튀면(Jump) False 반환"""
    if prev is None: return False
    if abs(cur["move_x_mm"] - prev["move_x_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_y_mm"] - prev["move_y_mm"]) > cfg.jump_xy_mm: return True
    # Z값 점프는 물체 높이가 다를 수 있으므로 조금 더 관대하게 보거나 상황에 따라 주석 처리 가능
    if abs(cur["move_z_mm"] - prev["move_z_mm"]) > cfg.jump_z_mm: return True
    return False

# ============================================================
# Vision Class (싱글톤 패턴의 백그라운드 런타임)
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
        
        # [수정 1] 측정 시작 시 과거 기록(Jump 비교용)을 초기화하기 위한 플래그
        self._reset_history_flag = False

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
        
        # [수정 1] 새로운 측정이 시작되었으므로, JUMP 비교 로직을 리셋하도록 요청
        self._reset_history_flag = True
        
        print(f"[Vision] 측정 시작 (목표: {n}개, 이전 위치 기록 리셋)")
        start = time.time()
        while time.time() - start < timeout:
            if len(self._collect_samples) >= n:
                break
            time.sleep(0.05)
        
        self._is_measuring = False
        
        if len(self._collect_samples) == 0:
            print("[Vision] 측정 실패: 데이터 없음")
            return None
            
        # 수집된 데이터 평균 계산
        arr = np.array([[s["move_x_mm"], s["move_y_mm"], s["move_z_mm"], s["angle_deg"]] for s in self._collect_samples])
        m = np.mean(arr, axis=0)
        return {
            "move_x_mm": float(m[0]), 
            "move_y_mm": float(m[1]), 
            "move_z_mm": float(m[2]),
            "angle_deg": float(m[3])
        }

    def _loop(self):
        # 1. 리얼센스 파이프라인 설정
        pipeline = rs.pipeline()
        config = rs.config()
        
        # [핵심] Config 파일의 설정을 그대로 사용 (640x480, 30fps)
        STREAM_W = cfg.width
        STREAM_H = cfg.height
        FPS = cfg.fps
        
        # [수정 2] 이미지 중심 좌표 미리 계산 (중앙에서 가까운 박스 찾기용)
        IMG_CENTER_X = STREAM_W / 2.0
        IMG_CENTER_Y = STREAM_H / 2.0
        
        config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)

        try:
            profile = pipeline.start(config)
        except Exception as e:
            print(f"[ERROR] 리얼센스 시작 실패: {e}")
            print(f" -> 요청 해상도: {STREAM_W}x{STREAM_H}, FPS: {FPS}")
            self._running = False
            return

        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()

        if cfg.show_preview:
            cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)

        prev_valid = None

        while self._running:
            # [수정 1 관련] 외부에서 리셋 요청이 들어오면 prev_valid를 None으로 만듦
            if self._reset_history_flag:
                prev_valid = None
                self._reset_history_flag = False
                # print("[Vision] JUMP 감지용 히스토리 리셋 완료")

            try:
                # 프레임 대기
                frames = pipeline.wait_for_frames()
                
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # 이미지 변환
                img = np.asanyarray(color_frame.get_data())
                
                # 깊이 필터 적용
                d_frame = spatial.process(depth_frame)
                d_frame = temporal.process(d_frame)
                d_u16 = np.asanyarray(d_frame.get_data())
                
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

                vis = img.copy()

                # YOLO 추론 (학습 해상도 imgsz=640 적용)
                results = self._model.predict(img, imgsz=cfg.imgsz, conf=cfg.conf_thres, verbose=False)
                r = results[0]

                best_sample = None

                # OBB 결과 처리
                if hasattr(r, 'obb') and r.obb is not None:
                    candidates = []
                    for i, conf in enumerate(r.obb.conf):
                        if conf < cfg.conf_thres: continue
                        
                        poly = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                        
                        # ROI 축소 (노이즈 방지)
                        poly_s = poly_shrink_towards_center(poly, 5)
                        
                        # 좌표 클램핑 (해상도 범위 내로 제한)
                        poly_s[:, 0] = np.clip(poly_s[:, 0], 0, STREAM_W-1)
                        poly_s[:, 1] = np.clip(poly_s[:, 1], 0, STREAM_H-1)
                        
                        # 깊이 계산
                        z_m, mad, count = depth_roi_stats(d_u16, depth_scale, poly_s)

                        # 유효성 검사 (깊이값 존재 및 픽셀 수 충족)
                        if z_m > 0 and count > 50:
                            # 2D 중심점 계산
                            cx = np.mean(poly[:, 0])
                            cy = np.mean(poly[:, 1])
                            
                            # [수정 2] 카메라 중심과의 거리(pixel) 계산
                            dist_from_center = np.sqrt((cx - IMG_CENTER_X)**2 + (cy - IMG_CENTER_Y)**2)
                            
                            angle = obb_angle_deg_upright0_rightplus(poly)
                            
                            candidates.append({
                                'z_m': z_m, 
                                'poly': poly, 
                                'angle': angle,
                                'dist_center': dist_from_center, # 중심과의 거리 저장
                                'cx': cx, 'cy': cy
                            })

                    if candidates:
                        # [수정 2] 정렬 기준 변경:
                        # 1순위: 화면 중앙과의 거리 (dist_center) -> 작을수록 좋음
                        # 2순위: 카메라 깊이 (z_m) -> (옵션: 너무 멀면 제외하거나 하지만 보통 중앙이 우선)
                        candidates.sort(key=lambda x: x['dist_center'])
                        
                        best = candidates[0]
                        
                        # 3D 좌표 변환 (이미 계산된 cx, cy 사용)
                        tx, ty = XY_from_pixel_and_Z(best['cx'], best['cy'], intr, best['z_m'])
                        
                        raw = {
                            "move_x_mm": tx * 1000.0,
                            "move_y_mm": ty * 1000.0,
                            "move_z_mm": best['z_m'] * 1000.0,
                            "angle_deg": best['angle']
                        }

                        # 점프 필터 (값이 튀는지 확인)
                        # [참고] measure_avg 호출 직후에는 prev_valid가 None이 되어 무조건 통과됨
                        if not is_jump(prev_valid, raw):
                            prev_valid = raw
                            best_sample = raw
                            
                            # 시각화 (정상 - 초록색)
                            txt = f"XYZ: {raw['move_x_mm']:.0f}, {raw['move_y_mm']:.0f}, {raw['move_z_mm']:.0f}"
                            cv2.polylines(vis, [np.int32(best['poly'])], True, (0, 255, 0), 2)
                            cv2.circle(vis, (int(best['cx']), int(best['cy'])), 5, (0, 255, 0), -1) # 중심점 표시
                            cv2.putText(vis, txt, (int(best['cx']), int(best['cy'])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            # 시각화 (점프 - 빨간색)
                            cv2.polylines(vis, [np.int32(best['poly'])], True, (0, 0, 255), 2)
                            cv2.putText(vis, "JUMP", (int(best['cx']), int(best['cy'])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 데이터 수집 (measure_avg 호출 시 동작)
                if self._is_measuring and best_sample:
                    self._collect_samples.append(best_sample)
                    cv2.putText(vis, f"Collecting: {len(self._collect_samples)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # 화면 출력
                if cfg.show_preview:
                    # 화면 중앙 십자선 표시 (참고용)
                    cv2.line(vis, (int(IMG_CENTER_X), 0), (int(IMG_CENTER_X), STREAM_H), (255, 255, 0), 1)
                    cv2.line(vis, (0, int(IMG_CENTER_Y)), (STREAM_W, int(IMG_CENTER_Y)), (255, 255, 0), 1)
                    
                    cv2.imshow(cfg.preview_win_name, vis)
                    # GUI 갱신 및 종료 키(ESC) 확인
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            except Exception as e:
                # 일시적 에러는 무시하고 계속 시도
                time.sleep(0.01)

        # 루프 종료 후 정리
        pipeline.stop()
        if cfg.show_preview:
            cv2.destroyAllWindows()

# 전역 인스턴스 생성
_RT = _VisionRuntime()

# 외부에서 호출할 함수들
def start_stream(): _RT.start()
def stop_stream(): _RT.stop()
def measure_avg(n=None, timeout_sec=None): return _RT.measure_avg(n, timeout_sec)
def is_running(): return _RT._running