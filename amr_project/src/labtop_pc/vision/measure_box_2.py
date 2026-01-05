# src/labtop_pc/vision/measure_box.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ✅ 1. 작성하신 설정 파일 불러오기 (cfg로 줄여서 사용)
from .vision_config import DEFAULT_VISION_CONFIG as cfg

# =========================================================
# 2. Helper Functions (cfg 값 사용)
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

    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    
    # ✅ cfg 설정값 사용
    d = d[(d > 0) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)]
    
    if d.size == 0:
        return 0.0, 0.0, 0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, int(d.size)

def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z_m: float):
    # [기존 코드 주석 처리] ----------------------------------------------------
    # 수동 핀홀 모델 공식: 렌즈 왜곡(Distortion)이 고려되지 않아 외곽 오차 발생 가능
    # X = (cx - intr.ppx) / intr.fx * Z_m
    # Y = (cy - intr.ppy) / intr.fy * Z_m
    # return float(X), float(Y)
    # ------------------------------------------------------------------------

    # [✅ 변경된 코드] RealSense SDK 공식 함수 사용 (Improvement 1)
    # rs2_deproject_pixel_to_point 함수는 내부 왜곡 파라미터(Coeffs)까지 고려하여
    # 2D 픽셀(Pixel) 좌표를 3D 공간(Point) 좌표로 정확하게 변환해 줍니다.
    point_3d = rs.rs2_deproject_pixel_to_point(intr, [float(cx), float(cy)], float(Z_m))
    
    # point_3d는 [x, y, z] 리스트를 반환함
    return point_3d[0], point_3d[1] # x, y (meters)

def obb_angle_deg_upright0_rightplus(poly4x2: np.ndarray) -> float:
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    q = p - c
    cov = np.cov(q.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)].astype(np.float32)

    vx, vy = float(v[0]), float(v[1])
    if vy < 0:
        vx, vy = -vx, -vy

    angle = float(np.degrees(np.arctan2(vx, vy)))
    return -angle

def is_jump(prev, cur):
    """ 값이 갑자기 튀는지 검사 (✅ cfg 설정값 사용) """
    if prev is None:
        return False
    # 이동량(mm) 및 각도(deg) 차이 비교
    if abs(cur["move_x_mm"] - prev["move_x_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_y_mm"] - prev["move_y_mm"]) > cfg.jump_xy_mm: return True
    if abs(cur["move_z_mm"] - prev["move_z_mm"]) > cfg.jump_z_mm: return True
    if abs(cur["angle_deg"] - prev["angle_deg"]) > cfg.jump_ang_deg: return True
    return False


# ============================================================
# 3. Vision Runtime Class (Singleton, Threaded)
# ============================================================
class _VisionRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 리소스
        self._model: Optional[YOLO] = None
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._depth_scale: float = 0.001

        # 상태 플래그 
        self._inference_mode = False  # False=대기(화면만), True=측정(YOLO)
        self._req_active = False      
        self._req_n = 10
        self._req_samples = []
        self._req_result = None
        
        # 필터용 이전 값
        self._prev_valid = None 

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def start(self):
        """ (0번) 카메라 켜기 + 모델 로드 (대기 모드 시작) """
        with self._lock:
            if self._running:
                return

            # ✅ cfg.model_path 사용
            print(f"[Vision] YOLO 모델 로딩 중... ({cfg.model_path})")
            try:
                self._model = YOLO(cfg.model_path)
            except Exception as e:
                print(f"[ERROR] 모델 로드 실패: {e}")
                return

            print("[Vision] RealSense 시작...")
            self._pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # ✅ cfg.width, cfg.height, cfg.fps 사용
            rs_config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            rs_config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
            
            try:
                prof = self._pipeline.start(rs_config)
            except Exception as e:
                print(f"[ERROR] 카메라 시작 실패: {e}")
                return

            self._align = rs.align(rs.stream.color)
            self._depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

            self._running = True
            self._inference_mode = False # 기본: CCTV 모드 (YOLO Off)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print("[Vision] 카메라 시작됨 (Monitor Mode)")

    def stop(self):
        with self._lock:
            self._running = False
            self._cv.notify_all()
        
        if self._thread:
            self._thread.join()
            self._thread = None
        
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        
        try: cv2.destroyAllWindows()
        except: pass
        print("[Vision] 시스템 종료")

    def measure_avg(self, n: int = None, timeout_sec: float = None) -> Optional[Dict[str, Any]]:
        """ (2번) 측정 모드 전환 -> 결과 반환 -> 대기 모드 복귀 """
        with self._lock:
            if not self._running:
                print("[Vision] 카메라가 꺼져있습니다. 0번으로 켜세요.")
                return None
            
            # ✅ 인자 없으면 config 값 사용
            if n is None: n = cfg.avg_n
            if timeout_sec is None: timeout_sec = cfg.timeout_sec

            # 측정 시작 세팅
            self._inference_mode = True 
            self._prev_valid = None # 필터 리셋
            self._req_samples = []
            self._req_result = None
            self._req_n = n
            self._req_active = True
            
            deadline = time.time() + timeout_sec
            self._cv.notify_all()

        # 결과 대기 (Blocking)
        try:
            while True:
                with self._lock:
                    if self._req_result is not None:
                        res = self._req_result
                        self._req_result = None
                        self._req_active = False
                        self._inference_mode = False # ★ 다시 모니터 모드로
                        return res
                    
                    if not self._running: return None
                    if time.time() > deadline:
                        self._req_active = False
                        self._inference_mode = False
                        return None
                    
                    self._cv.wait(timeout=0.1)
        except KeyboardInterrupt:
            return None

    def _loop(self):
        # 리얼센스 필터 미리 생성
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        temporal = rs.temporal_filter()
        hole = rs.hole_filling_filter()

        # ✅ cfg.preview_win_name 사용
        if cfg.show_preview:
            cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)

        while self._running:
            try:
                frames = self._pipeline.wait_for_frames()
                frames = self._align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame: continue
            except: continue

            img = np.asanyarray(color_frame.get_data())
            vis = img.copy()

            # 현재 모드 확인 (Thread-Safe)
            with self._lock:
                do_inference = self._inference_mode
                req_active = self._req_active

            if do_inference:
                # =========================
                # [모드 2] 측정 중 (YOLO ON)
                # =========================
                # 1. Depth 필터링
                depth_frame = spatial.process(depth_frame).as_depth_frame()
                depth_frame = temporal.process(depth_frame).as_depth_frame()
                depth_frame = hole.process(depth_frame).as_depth_frame()
                d_u16 = np.asanyarray(depth_frame.get_data())
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

                # 2. YOLO 추론 (✅ cfg.imgsz, cfg.conf_thres, cfg.iou_thres 사용)
                try:
                    results = self._model.predict(
                        img, 
                        imgsz=cfg.imgsz, 
                        conf=cfg.conf_thres, 
                        iou=cfg.iou_thres,
                        verbose=False
                    )
                    r = results[0]
                except: r = None

                cur_data = None
                
                # 3. 박스 추출 및 계산
                if r and getattr(r, 'obb', None) is not None:
                    if r.obb.xyxyxyxy is not None and len(r.obb.xyxyxyxy) > 0:
                        # 가장 신뢰도 높은 1개 사용
                        idx = int(r.obb.conf.argmax())
                        poly = r.obb.xyxyxyxy[idx].cpu().numpy().reshape(4, 2)
                        
                        # 시각화
                        cv2.polylines(vis, [np.int32(poly)], True, (0, 0, 255), 2)

                        # ROI 축소 (✅ cfg.roi_margin_px)
                        poly_s = poly_shrink_towards_center(poly, cfg.roi_margin_px)
                        poly_s[:, 0] = np.clip(poly_s[:, 0], 0, cfg.width-1)
                        poly_s[:, 1] = np.clip(poly_s[:, 1], 0, cfg.height-1)
                        
                        # Depth 통계 (✅ cfg.mad_thres_m)
                        z_m, mad, count = depth_roi_stats(d_u16, self._depth_scale, poly_s)

                        # 유효성 검사 (✅ cfg.min_roi_pixels)
                        if z_m > 0 and count >= cfg.min_roi_pixels and mad <= cfg.mad_thres_m:
                            cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
                            x_m, y_m = XY_from_pixel_and_Z(cx, cy, intr, z_m)
                            angle = obb_angle_deg_upright0_rightplus(poly)
                            
                            temp = {
                                "move_x_mm": x_m * 1000.0, 
                                "move_y_mm": y_m * 1000.0, 
                                "move_z_mm": z_m * 1000.0, 
                                "angle_deg": angle
                            }
                            
                            # 점프 필터 (튀는 값 방지)
                            if not is_jump(self._prev_valid, temp):
                                self._prev_valid = temp
                                cur_data = temp
                                
                                # 좌표 표시
                                if cfg.show_overlay:
                                    txt = f"{temp['move_x_mm']:.0f},{temp['move_y_mm']:.0f},{temp['move_z_mm']:.0f}"
                                    cv2.putText(vis, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                                
                                # ✅ 샘플 수집 시 로그 출력 (설정값에 따름)
                                if cfg.print_selected_each_accept:
                                    print(f"[Sample] {txt} | Ang: {angle:.1f}")
                            else:
                                if cfg.show_overlay:
                                    cv2.putText(vis, "JUMP!", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # 4. 샘플 수집 및 종료 조건
                with self._lock:
                    if req_active and cur_data:
                        self._req_samples.append(cur_data)
                        
                        if len(self._req_samples) >= self._req_n:
                            arr = np.array([[s["move_x_mm"], s["move_y_mm"], s["move_z_mm"], s["angle_deg"]] for s in self._req_samples])
                            mean_val = np.mean(arr, axis=0)
                            self._req_result = {
                                "move_x_mm": float(mean_val[0]),
                                "move_y_mm": float(mean_val[1]),
                                "move_z_mm": float(mean_val[2]),
                                "angle_deg": float(mean_val[3])
                            }
                            self._cv.notify_all()
                
                # UI: 측정 중 표시
                if cfg.show_overlay:
                    cv2.putText(vis, f"MEASURING ({len(self._req_samples)}/{self._req_n})", (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                # =========================
                # [모드 1] 대기 모드 (YOLO OFF)
                # =========================
                if cfg.show_overlay:
                    cx, cy = cfg.width // 2, cfg.height // 2
                    cv2.line(vis, (cx-20, cy), (cx+20, cy), (0, 255, 0), 1)
                    cv2.line(vis, (cx, cy-20), (cx, cy+20), (0, 255, 0), 1)
                    cv2.putText(vis, "MONITOR MODE (No AI)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 화면 출력
            if cfg.show_preview:
                cv2.imshow(cfg.preview_win_name, vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop()
                    break
        
        if cfg.show_preview:
            try: cv2.destroyAllWindows()
            except: pass

# 싱글톤 인스턴스
_RT = _VisionRuntime()

# -----------------------------
# 4. Public API
# -----------------------------
def start_stream():
    """ 0번 메뉴: 카메라 켜기 """
    _RT.start()

def stop_stream():
    """ 종료 시 호출 """
    _RT.stop()

def measure_avg(n: int = None, timeout_sec: float = None) -> Optional[Dict[str, Any]]:
    """ 2번 메뉴: 측정 수행 """
    return _RT.measure_avg(n, timeout_sec)

def is_running() -> bool:
    return _RT.is_running()