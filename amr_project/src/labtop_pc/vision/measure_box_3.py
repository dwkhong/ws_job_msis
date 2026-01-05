# src/labtop_pc/vision/measure_box.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import time
import threading
import math
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ✅ 1. 설정 파일 불러오기
from .vision_config import DEFAULT_VISION_CONFIG as cfg

# =========================================================
# 2. Helper Functions
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def poly_shrink_towards_center(poly4x2: np.ndarray, margin_px: float):
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    v = p - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return p - (v / norm) * margin_px

def depth_roi_stats(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray):
    """ ROI 내부 깊이값의 중앙값(Median) 계산 """
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)

    # ROI 내부 데이터만 추출
    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    
    # 유효 범위 필터링
    d = d[(d > 0) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)]
    
    if d.size == 0:
        return 0.0, 0.0, 0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, int(d.size)

def calc_surface_angle(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray, intr):
    """
    [NEW] ROI 영역의 3D 점들을 분석하여 표면의 기울기(Ry, Rx)를 계산함 (SVD 평면 피팅)
    return: (ry_deg, rx_deg)
    """
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)

    # 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
    
    # 데이터 추출 (속도를 위해 다운샘플링 ::3)
    ys, xs = np.where(mask[::3, ::3] == 255)
    ys *= 3; xs *= 3
    
    z_vals = depth_u16[ys, xs].astype(np.float32) * depth_scale
    
    # 노이즈 제거 및 유효 값 필터링
    valid = (z_vals > cfg.depth_min_m) & (z_vals < cfg.depth_max_m)
    if np.sum(valid) < 50: # 점이 너무 적으면 계산 불가
        return 0.0, 0.0

    z_v = z_vals[valid]
    u_v = xs[valid]
    v_v = ys[valid]

    # 3D 좌표 변환 (간이 공식)
    x_v = (u_v - intr.ppx) / intr.fx * z_v
    y_v = (v_v - intr.ppy) / intr.fy * z_v
    
    # 평면 피팅 (SVD)
    points = np.vstack((x_v, y_v, z_v)).T
    centered = points - points.mean(axis=0)
    
    # SVD 수행 -> 가장 작은 고유값의 벡터가 '법선 벡터(Normal)'
    try:
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2, :] 
    except:
        return 0.0, 0.0

    # 법선 벡터 방향 정리 (카메라 쪽을 향하도록, nz가 음수가 되게)
    if normal[2] > 0:
        normal = -normal

    # 각도 변환 (Normal -> Euler)
    nx, ny, nz = normal
    
    # Pitch (Ry): 앞뒤 기울기
    pitch_rad = math.asin(np.clip(-ny, -1.0, 1.0))
    # Roll (Rx): 좌우 기울기
    roll_rad = math.atan2(nx, -nz)

    return np.degrees(pitch_rad), np.degrees(roll_rad)

def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z_m: float):
    point_3d = rs.rs2_deproject_pixel_to_point(intr, [float(cx), float(cy)], float(Z_m))
    return point_3d[0], point_3d[1]

def obb_angle_deg_upright0_rightplus(poly4x2: np.ndarray) -> float:
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    q = p - c
    cov = np.cov(q.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    vx, vy = float(v[0]), float(v[1])
    if vy < 0: vx, vy = -vx, -vy
    angle = float(np.degrees(np.arctan2(vx, vy)))
    return -angle

def is_jump(prev, cur):
    if prev is None: return False
    if abs(cur["move_x_mm"] - prev["move_x_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_y_mm"] - prev["move_y_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_z_mm"] - prev["move_z_mm"]) > cfg.jump_z_mm: return True
    if abs(cur["angle_deg"] - prev["angle_deg"]) > cfg.jump_ang_deg: return True
    return False

# ============================================================
# 3. Vision Runtime Class
# ============================================================
class _VisionRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._model: Optional[YOLO] = None
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._depth_scale: float = 0.001

        self._inference_mode = False
        self._req_active = False      
        self._req_n = 10
        self._req_samples = []
        self._req_result = None
        self._prev_valid = None 

    def is_running(self) -> bool:
        with self._lock: return bool(self._running)

    def start(self):
        with self._lock:
            if self._running: return
            print(f"[Vision] YOLO 모델 로딩 중... ({cfg.model_path})")
            try: self._model = YOLO(cfg.model_path)
            except Exception as e:
                print(f"[ERROR] 모델 로드 실패: {e}")
                return

            print("[Vision] RealSense 시작...")
            self._pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            rs_config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
            
            try: prof = self._pipeline.start(rs_config)
            except Exception as e:
                print(f"[ERROR] 카메라 시작 실패: {e}")
                return

            self._align = rs.align(rs.stream.color)
            self._depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
            self._running = True
            self._inference_mode = False
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print("[Vision] 카메라 시작됨")

    def stop(self):
        with self._lock:
            self._running = False
            self._cv.notify_all()
        if self._thread: self._thread.join(); self._thread = None
        if self._pipeline: self._pipeline.stop(); self._pipeline = None
        try: cv2.destroyAllWindows()
        except: pass

    def measure_avg(self, n: int = None, timeout_sec: float = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._running: return None
            if n is None: n = cfg.avg_n
            if timeout_sec is None: timeout_sec = cfg.timeout_sec

            self._inference_mode = True 
            self._prev_valid = None 
            self._req_samples = []
            self._req_result = None
            self._req_n = n
            self._req_active = True
            deadline = time.time() + timeout_sec
            self._cv.notify_all()

        try:
            while True:
                with self._lock:
                    if self._req_result is not None:
                        res = self._req_result
                        self._req_result = None
                        self._req_active = False
                        self._inference_mode = False
                        return res
                    if not self._running or time.time() > deadline:
                        self._req_active = False; self._inference_mode = False
                        return None
                    self._cv.wait(timeout=0.1)
        except KeyboardInterrupt: return None

    def _loop(self):
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        temporal = rs.temporal_filter()
        hole = rs.hole_filling_filter()

        if cfg.show_preview:
            cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)

        while self._running:
            try:
                frames = self._pipeline.wait_for_frames()
                frames = self._align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or not depth: continue
            except: continue

            img = np.asanyarray(color.get_data())
            vis = img.copy()

            with self._lock:
                do_inference = self._inference_mode
                req_active = self._req_active

            if do_inference:
                depth = spatial.process(depth).as_depth_frame()
                depth = temporal.process(depth).as_depth_frame()
                depth = hole.process(depth).as_depth_frame()
                d_u16 = np.asanyarray(depth.get_data())
                intr = color.profile.as_video_stream_profile().get_intrinsics()

                try:
                    results = self._model.predict(img, imgsz=cfg.imgsz, conf=cfg.conf_thres, iou=cfg.iou_thres, verbose=False)
                    r = results[0]
                except: r = None

                cur_data = None
                
                # [변경] 모든 박스를 검사하여 '가장 높은(Z가 작은)' 박스를 찾음
                candidates = [] # (z_m, poly, angle, stats)

                if r and getattr(r, 'obb', None) is not None:
                    # 모든 검출된 객체 순회
                    for i, conf in enumerate(r.obb.conf):
                        if conf < cfg.conf_thres: continue
                        
                        poly = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                        
                        # ROI 축소 및 Depth 계산
                        poly_s = poly_shrink_towards_center(poly, cfg.roi_margin_px)
                        poly_s[:, 0] = np.clip(poly_s[:, 0], 0, cfg.width-1)
                        poly_s[:, 1] = np.clip(poly_s[:, 1], 0, cfg.height-1)
                        
                        z_m, mad, count = depth_roi_stats(d_u16, self._depth_scale, poly_s)

                        # 유효하면 후보 등록
                        if z_m > 0 and count >= cfg.min_roi_pixels and mad <= cfg.mad_thres_m:
                            angle = obb_angle_deg_upright0_rightplus(poly)
                            candidates.append({
                                'z_m': z_m, 'poly': poly, 'poly_s': poly_s, 'angle': angle
                            })
                            # 시각화 (후보들 얇게 표시)
                            cv2.polylines(vis, [np.int32(poly)], True, (0, 100, 100), 1)

                    # 후보 중 가장 높은(Z값이 작은) 박스 선택
                    if candidates:
                        # Z값 기준으로 오름차순 정렬 (작을수록 가까움=높음)
                        candidates.sort(key=lambda x: x['z_m'])
                        best = candidates[0] # Best Candidate
                        
                        # Best 정보 추출
                        z_m = best['z_m']
                        poly = best['poly']
                        poly_s = best['poly_s']
                        angle = best['angle']
                        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
                        
                        # 좌표 변환
                        x_m, y_m = XY_from_pixel_and_Z(cx, cy, intr, z_m)
                        
                        # [추가] 표면 기울기(Ry, Rx) 계산
                        tilt_ry, tilt_rx = calc_surface_angle(d_u16, self._depth_scale, poly_s, intr)

                        temp = {
                            "move_x_mm": x_m * 1000.0, 
                            "move_y_mm": y_m * 1000.0, 
                            "move_z_mm": z_m * 1000.0, 
                            "angle_deg": angle,
                            "tilt_ry": tilt_ry,  # 기울기 정보 추가
                            "tilt_rx": tilt_rx
                        }
                        
                        # 시각화 (Best는 굵게)
                        cv2.polylines(vis, [np.int32(poly)], True, (0, 0, 255), 3)
                        
                        if not is_jump(self._prev_valid, temp):
                            self._prev_valid = temp
                            cur_data = temp
                            txt = f"X:{temp['move_x_mm']:.0f} Z:{temp['move_z_mm']:.0f} Ry:{tilt_ry:.1f}"
                            cv2.putText(vis, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        else:
                            cv2.putText(vis, "JUMP!", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # 샘플 수집 및 평균 계산
                with self._lock:
                    if req_active and cur_data:
                        self._req_samples.append(cur_data)
                        if len(self._req_samples) >= self._req_n:
                            arr = np.array([[s["move_x_mm"], s["move_y_mm"], s["move_z_mm"], s["angle_deg"], s["tilt_ry"], s["tilt_rx"]] for s in self._req_samples])
                            mean_val = np.mean(arr, axis=0)
                            self._req_result = {
                                "move_x_mm": float(mean_val[0]),
                                "move_y_mm": float(mean_val[1]),
                                "move_z_mm": float(mean_val[2]),
                                "angle_deg": float(mean_val[3]),
                                "tilt_ry": float(mean_val[4]), # 평균 기울기
                                "tilt_rx": float(mean_val[5])
                            }
                            self._cv.notify_all()
                
                if cfg.show_overlay:
                    cv2.putText(vis, f"MEASURING ({len(self._req_samples)}/{self._req_n})", (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if cfg.show_overlay:
                    cv2.putText(vis, "MONITOR MODE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if cfg.show_preview:
                cv2.imshow(cfg.preview_win_name, vis)
                if cv2.waitKey(1) & 0xFF == 27: self.stop(); break
        
        if cfg.show_preview:
            try:
                cv2.destroyAllWindows()
            except:
                pass

_RT = _VisionRuntime()

def start_stream(): _RT.start()
def stop_stream(): _RT.stop()
def measure_avg(n: int = None, timeout_sec: float = None) -> Optional[Dict[str, Any]]: return _RT.measure_avg(n, timeout_sec)
def is_running() -> bool: return _RT.is_running()