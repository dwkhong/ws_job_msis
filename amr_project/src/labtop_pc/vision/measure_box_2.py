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
# Helper Functions (수학 및 전처리)
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

# [NEW] 3D 평면 기울기 계산 함수
def calculate_surface_orientation(depth_u16, depth_scale, poly4x2, intr):
    """
    박스 내부의 3D 점들을 이용해 평면을 피팅하고, 카메라 기준 RX, RY 기울기를 계산
    Return: (rx_deg, ry_deg)
    """
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)
    
    # 1. 마스크 생성 및 유효 픽셀 추출
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
    
    # ROI 내의 좌표들 (v: y좌표, u: x좌표)
    v_coords, u_coords = np.where(mask == 255)
    z_vals = depth_u16[v_coords, u_coords].astype(np.float32) * depth_scale
    
    # 유효 깊이 필터링
    valid_idx = (z_vals > cfg.depth_min_m) & (z_vals <= cfg.depth_max_m)
    if np.sum(valid_idx) < 50: # 점이 너무 적으면 0도 리턴
        return 0.0, 0.0

    u_valid = u_coords[valid_idx]
    v_valid = v_coords[valid_idx]
    z_valid = z_vals[valid_idx]

    # 성능을 위해 점 개수 다운샘플링 (최대 500개만 사용)
    if len(z_valid) > 500:
        indices = np.random.choice(len(z_valid), 500, replace=False)
        u_valid = u_valid[indices]
        v_valid = v_valid[indices]
        z_valid = z_valid[indices]

    # 2. 3D Point Cloud 변환 (Vectorized)
    # X = (u - ppx) * z / fx
    # Y = (v - ppy) * z / fy
    X = (u_valid - intr.ppx) * z_valid / intr.fx
    Y = (v_valid - intr.ppy) * z_valid / intr.fy
    Z = z_valid

    # (N, 3) 형태의 점 구름
    points = np.vstack((X, Y, Z)).T

    # 3. 평면 피팅 (SVD 이용)
    # 중심점 제거
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # SVD 수행 -> V의 마지막 행이 법선 벡터(Normal Vector)
    try:
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2, :] # [nx, ny, nz]
    except:
        return 0.0, 0.0

    # 법선 벡터 방향 정리 (카메라 쪽을 향하도록, 즉 nz가 음수가 되도록)
    # Realsense 좌표계: Z가 전방(+)이므로 평면이 카메라를 보면 Normal Z는 (-)여야 함
    if normal[2] > 0:
        normal = -normal

    # 4. 각도 계산 (Normal Vector -> Euler Angle)
    # nx, ny, nz
    # RY (Pitch): XZ 평면에서의 기울기 -> atan2(nx, nz)
    # RX (Roll):  YZ 평면에서의 기울기 -> atan2(ny, nz)
    
    # 단순화된 계산 (작은 각도 근사)
    # 카메라 좌표계 기준:
    # - 박스 오른쪽이 들리면: Normal X가 (-) -> RY (+) 
    # - 박스 위쪽이 들리면:   Normal Y가 (+) -> RX (+)
    
    # 각도 변환 (Rad -> Deg)
    # normal[2]가 주축이므로 이를 기준으로 계산
    # 주의: 로봇 좌표계와 카메라 좌표계의 축 방향을 고려해야 함
    
    # Camera Coords: X(Right), Y(Down), Z(Forward)
    # Robot Coords에 맞게 변환 필요하지만 일단 카메라 기준 각도로 반환
    
    rad_ry = np.arctan2(normal[0], abs(normal[2])) # Pitch-like
    rad_rx = np.arctan2(normal[1], abs(normal[2])) # Roll-like
    
    deg_ry = np.degrees(rad_ry)
    deg_rx = np.degrees(rad_rx)

    # 노이즈로 인한 미세 각도 무시 (Deadzone)
    if abs(deg_ry) < 2.0: deg_ry = 0.0
    if abs(deg_rx) < 2.0: deg_rx = 0.0

    return float(deg_rx), float(deg_ry)


def is_jump(prev, cur):
    if prev is None: return False
    if abs(cur["move_x_mm"] - prev["move_x_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_y_mm"] - prev["move_y_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_z_mm"] - prev["move_z_mm"]) > cfg.jump_z_mm: return True
    return False

# ============================================================
# Vision Class
# ============================================================
class _VisionRuntime:
    def __init__(self):
        self._running = False
        self._thread = None
        self._model = None
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
            
        arr = np.array([[s["move_x_mm"], s["move_y_mm"], s["move_z_mm"], s["angle_deg"], s["ry_deg"], s["rx_deg"]] for s in self._collect_samples])
        m = np.mean(arr, axis=0)
        return {
            "move_x_mm": float(m[0]), 
            "move_y_mm": float(m[1]), 
            "move_z_mm": float(m[2]),
            "angle_deg": float(m[3]),
            "ry_deg": float(m[4]),  # [NEW] 계산된 Pitch
            "rx_deg": float(m[5])   # [NEW] 계산된 Roll
        }

    def _loop(self):
        pipeline = rs.pipeline()
        config = rs.config()
        
        STREAM_W, STREAM_H, FPS = cfg.width, cfg.height, cfg.fps
        config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)

        try:
            profile = pipeline.start(config)
        except Exception as e:
            print(f"[ERROR] 리얼센스 시작 실패: {e}")
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
            try:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame: continue

                img = np.asanyarray(color_frame.get_data())
                d_frame = spatial.process(depth_frame)
                d_frame = temporal.process(d_frame)
                d_u16 = np.asanyarray(d_frame.get_data())
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

                vis = img.copy()

                results = self._model.predict(img, imgsz=cfg.imgsz, conf=cfg.conf_thres, verbose=False)
                r = results[0]

                best_sample = None

                if hasattr(r, 'obb') and r.obb is not None:
                    candidates = []
                    for i, conf in enumerate(r.obb.conf):
                        if conf < cfg.conf_thres: continue
                        
                        poly = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                        poly_s = poly_shrink_towards_center(poly, 5)
                        
                        poly_s[:, 0] = np.clip(poly_s[:, 0], 0, STREAM_W-1)
                        poly_s[:, 1] = np.clip(poly_s[:, 1], 0, STREAM_H-1)
                        
                        z_m, mad, count = depth_roi_stats(d_u16, depth_scale, poly_s)

                        if z_m > 0 and count > 50:
                            angle_rz = obb_angle_deg_upright0_rightplus(poly)
                            
                            # [NEW] 여기서 표면 기울기(RX, RY) 계산
                            calc_rx, calc_ry = calculate_surface_orientation(d_u16, depth_scale, poly_s, intr)
                            
                            candidates.append({
                                'z_m': z_m, 'poly': poly, 
                                'angle': angle_rz, 
                                'rx': calc_rx, 'ry': calc_ry
                            })

                    if candidates:
                        candidates.sort(key=lambda x: x['z_m'])
                        best = candidates[0]
                        
                        cx, cy = np.mean(best['poly'][:, 0]), np.mean(best['poly'][:, 1])
                        tx, ty = XY_from_pixel_and_Z(cx, cy, intr, best['z_m'])
                        
                        raw = {
                            "move_x_mm": tx * 1000.0,
                            "move_y_mm": ty * 1000.0,
                            "move_z_mm": best['z_m'] * 1000.0,
                            "angle_deg": best['angle'],
                            "ry_deg": best['ry'], # [NEW]
                            "rx_deg": best['rx']  # [NEW]
                        }

                        if not is_jump(prev_valid, raw):
                            prev_valid = raw
                            best_sample = raw
                            
                            # 시각화 정보 추가 (Tilt 정보 표시)
                            txt = f"Z:{raw['move_z_mm']:.0f} RY:{raw['ry_deg']:.1f} RX:{raw['rx_deg']:.1f}"
                            cv2.polylines(vis, [np.int32(best['poly'])], True, (0, 255, 0), 2)
                            cv2.putText(vis, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.polylines(vis, [np.int32(best['poly'])], True, (0, 0, 255), 2)
                            cv2.putText(vis, "JUMP", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if self._is_measuring and best_sample:
                    self._collect_samples.append(best_sample)
                    cv2.putText(vis, f"Collecting: {len(self._collect_samples)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if cfg.show_preview:
                    cv2.imshow(cfg.preview_win_name, vis)
                    if cv2.waitKey(1) & 0xFF == 27: break

            except Exception as e:
                time.sleep(0.01)

        pipeline.stop()
        if cfg.show_preview:
            cv2.destroyAllWindows()

_RT = _VisionRuntime()
def start_stream(): _RT.start()
def stop_stream(): _RT.stop()
def measure_avg(n=None, timeout_sec=None): return _RT.measure_avg(n, timeout_sec)
def is_running(): return _RT._running