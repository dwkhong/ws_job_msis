from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

from .vision_config import VisionConfig, DEFAULT_VISION_CONFIG

# =========================================================
# 2. Helper Functions (수학 계산)
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def order_points(pts):
    """ 좌표를 [Top-Left, Top-Right, Bottom-Right, Bottom-Left] 순서로 정렬 """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def solve_pose_pnp_calibrated(poly4x2: np.ndarray, intr, cfg: VisionConfig):
    """ 
    왜곡 계수와 IPPE 알고리즘을 사용해 정확한 3D 위치를 계산 
    (화면에 그리는 용도가 아니라, 로봇에게 줄 정확한 좌표를 계산함)
    """
    # 1. 이미지 포인트 정렬
    image_points = order_points(poly4x2)

    # 2. 실제 박스 모델 매칭 (가로/세로 비율 자동 감지)
    len_top = np.linalg.norm(image_points[0] - image_points[1])
    len_side = np.linalg.norm(image_points[1] - image_points[2])
    
    if len_top > len_side:
        real_w, real_h = max(cfg.box_w_mm, cfg.box_h_mm), min(cfg.box_w_mm, cfg.box_h_mm)
    else:
        real_w, real_h = min(cfg.box_w_mm, cfg.box_h_mm), max(cfg.box_w_mm, cfg.box_h_mm)

    sx, sy = real_w / 2.0, real_h / 2.0
    
    # 3D 점 정의 (중심 0,0,0)
    object_points = np.array([
        [-sx, -sy, 0], [ sx, -sy, 0], [ sx,  sy, 0], [-sx,  sy, 0]
    ], dtype=np.float32)

    # 3. 카메라 매트릭스 구성
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # ★ 핵심: 왜곡 계수 적용
    dist_coeffs = np.array(cfg.dist_coeffs, dtype=np.float32)

    # 4. SolvePnP 실행 (IPPE Flag 사용)
    success, rvec, tvec = cv2.solvePnP(
        object_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success: return None

    # 5. 결과 변환
    X_mm = float(tvec[0])
    Y_mm = float(tvec[1])
    Z_mm = float(tvec[2])

    rmat, _ = cv2.Rodrigues(rvec)
    angle_rad = np.arctan2(rmat[1, 0], rmat[0, 0])
    angle_deg = np.degrees(angle_rad)

    return {"X": X_mm, "Y": Y_mm, "Z": Z_mm, "A": angle_deg}

def is_jump(prev, cur, cfg: VisionConfig):
    """ 값이 너무 튀는지 검사 (노이즈 필터링) """
    if prev is None: return False
    if abs(cur["Xmm"] - prev["Xmm"]) > cfg.jump_xy_mm: return True
    if abs(cur["Ymm"] - prev["Ymm"]) > cfg.jump_xy_mm: return True
    if abs(cur["Zmm"] - prev["Zmm"]) > cfg.jump_z_mm: return True
    return False

def draw_overlay_xyz_angle(img, Xmm, Ymm, Zmm, angle, cfg: VisionConfig):
    """ 화면 좌상단에 현재 좌표 표시 """
    if not cfg.show_overlay: return
    line1 = f"PnP X {Xmm:+.1f}  Y {Ymm:+.1f}  Z {Zmm:+.1f} (mm)"
    line2 = f"angle {angle:+.2f} deg"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w1, h1), _ = cv2.getTextSize(line1, font, cfg.overlay_font_scale, cfg.overlay_thickness)
    w = max(w1, 250)
    
    cv2.rectangle(img, (6, 6), (6 + w + 12, 6 + h1*2 + 25), (0, 0, 0), -1)
    cv2.addWeighted(img.copy(), 0.35, img, 0.65, 0, img)
    cv2.putText(img, line1, (10, 25), font, cfg.overlay_font_scale, (0, 255, 255), cfg.overlay_thickness, cv2.LINE_AA)
    cv2.putText(img, line2, (10, 50), font, cfg.overlay_font_scale, (255, 255, 255), cfg.overlay_thickness, cv2.LINE_AA)


# =========================================================
# 3. Vision Runtime (Threaded Class) - 핵심 엔진
# =========================================================
class _VisionRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._cfg: VisionConfig = DEFAULT_VISION_CONFIG
        self._model: Optional[YOLO] = None
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None

        # 측정 요청 상태
        self._req_active: bool = False
        self._req_n: int = 10
        self._req_deadline: float = 0.0
        self._req_samples: List[List[float]] = []
        self._req_result: Optional[Dict[str, Any]] = None
        
        # 화면 표시용 (최신 유효 값)
        self._last_disp = {"Xmm": None, "Ymm": None, "Zmm": None, "angle": None}

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def start(self, cfg: VisionConfig = DEFAULT_VISION_CONFIG):
        """ 시스템 시작 (카메라 켜기) """
        with self._lock:
            if self._running: return
            self._cfg = cfg
            
            print(f"Loading Model: {cfg.model_path}")
            self._model = YOLO(cfg.model_path)

            print("Starting RealSense Pipeline...")
            self._pipeline = rs.pipeline()
            rs_cfg = rs.config()
            rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            # PnP 방식은 Depth 맵이 필요 없지만, 파이프라인 안정성을 위해 켜둡니다.
            rs_cfg.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
            self._pipeline.start(rs_cfg)
            self._align = rs.align(rs.stream.color)

            if cfg.show_preview:
                cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cfg.preview_win_name, cfg.width, cfg.height)

            self._req_active = False
            self._req_samples = []
            self._req_result = None
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="VisionLoop", daemon=True)
            self._thread.start()
            print(">>> Vision System Started Successfully.")

    def stop(self):
        """ 시스템 종료 (카메라 끄기) """
        with self._lock:
            if not self._running: return
            self._running = False
            self._cv.notify_all()
        
        if self._thread: self._thread.join(timeout=2.0)
        try:
            if self._pipeline: self._pipeline.stop()
        except: pass
        if self._cfg.show_preview: 
            try: cv2.destroyAllWindows()
            except: pass
            
        self._pipeline = None
        self._model = None
        print(">>> Vision System Stopped.")

    def measure_avg(self, n: int = 10, timeout_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """ [메인 스레드 호출용] n개의 데이터를 모아서 평균값 반환 """
        with self._lock:
            if not self._running: return None
            if timeout_sec is None: timeout_sec = self._cfg.timeout_sec

            self._req_active = True
            self._req_n = int(max(1, n))
            self._req_deadline = time.time() + float(timeout_sec)
            self._req_samples = []
            self._req_result = None
            self._cv.notify_all() # 스레드 깨우기

            # 결과가 나올 때까지 대기 (Blocking)
            while True:
                if self._req_result is not None:
                    res = dict(self._req_result)
                    self._req_result = None
                    return res # 성공
                
                if not self._running: return None # 시스템 종료됨
                
                if time.time() >= self._req_deadline:
                    self._req_active = False
                    return None # 타임아웃

                self._cv.wait(timeout=0.1)

    def _loop(self):
        """ [백그라운드 스레드] 무한 루프 """
        cfg = self._cfg
        prev_valid = None
        consec_skips = 0

        while self._running:
            # 1. 프레임 획득
            try:
                frames = self._pipeline.wait_for_frames()
                frames = self._align.process(frames)
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                
                frame = np.asanyarray(color_frame.get_data())
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            except: 
                continue

            vis = frame.copy()
            
            # 2. YOLO 추론
            try:
                results = self._model.predict(frame, imgsz=cfg.imgsz, conf=cfg.conf_thres, verbose=False)
                r = results[0]
            except: r = None
            
            candidates = []
            img_cx, img_cy = cfg.width/2, cfg.height/2

            if r and getattr(r, "obb", None) is not None and r.obb.xyxyxyxy is not None:
                polys = r.obb.xyxyxyxy.cpu().numpy()
                confs = r.obb.conf.cpu().numpy()
                
                for poly, cf in zip(polys, confs):
                    poly = poly.reshape(4, 2)
                    cx = np.mean(poly[:, 0])
                    cy = np.mean(poly[:, 1])
                    dist2 = (cx - img_cx)**2 + (cy - img_cy)**2
                    candidates.append((dist2, poly))

            # 3. 데이터 처리
            current_data = None
            if candidates:
                candidates.sort(key=lambda x: x[0]) # 중앙 우선
                _, poly = candidates[0]

                # ★★★ 핵심: 왜곡 보정된 PnP 계산 ★★★
                res = solve_pose_pnp_calibrated(poly, intr, cfg)

                if res:
                    # 화면엔 깔끔하게 녹색 박스만 그리기 (사용자 요청)
                    poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [poly_i], True, (0, 255, 0), 2)
                    cv2.circle(vis, (int(np.mean(poly[:,0])), int(np.mean(poly[:,1]))), 5, (0,0,255), -1)

                    # 값은 보정된 값 사용
                    cur = {"Xmm": res['X'], "Ymm": res['Y'], "Zmm": res['Z'], "angle": res['A']}
                    
                    # 튀는 값 필터링
                    if res['Z'] > 100.0:
                        if not is_jump(prev_valid, cur, cfg):
                            prev_valid = cur
                            consec_skips = 0
                            current_data = cur
                        else:
                            consec_skips += 1
                            if consec_skips > cfg.max_consec_skips_reset: prev_valid = None
                    else:
                        consec_skips += 1
            else:
                consec_skips += 1
                if consec_skips >= cfg.max_consec_skips_reset: prev_valid = None
            
            # 4. 측정 요청 처리
            with self._lock:
                if self._req_active:
                    if time.time() > self._req_deadline:
                        self._req_active = False # 타임아웃
                        self._cv.notify_all()
                    elif current_data:
                        self._req_samples.append([current_data["Xmm"], current_data["Ymm"], current_data["Zmm"], current_data["angle"]])
                        
                        # 목표 개수 도달 시 평균 계산 후 종료
                        if len(self._req_samples) >= self._req_n:
                            arr = np.array(self._req_samples)
                            avg = np.mean(arr, axis=0)
                            self._req_result = {
                                "cam_x_mm": float(avg[0]), "cam_y_mm": float(avg[1]), 
                                "cam_z_mm": float(avg[2]), "angle_deg": float(avg[3])
                            }
                            self._req_active = False
                            self._cv.notify_all()

            # 5. UI 그리기
            if cfg.show_preview:
                if prev_valid:
                    self._last_disp = prev_valid
                    draw_overlay_xyz_angle(vis, prev_valid["Xmm"], prev_valid["Ymm"], prev_valid["Zmm"], prev_valid["angle"], cfg)
                
                if self._req_active:
                    cnt = len(self._req_samples)
                    cv2.putText(vis, f"MEASURING... {cnt}/{self._req_n}", 
                                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.imshow(cfg.preview_win_name, vis)
                if cv2.waitKey(1) == 27: self._running = False


# 싱글톤 인스턴스 생성
_RT = _VisionRuntime()


# =========================================================
# 4. Public API (외부 사용용)
# =========================================================
def start_stream(cfg: VisionConfig = DEFAULT_VISION_CONFIG):
    _RT.start(cfg)

def stop_stream():
    _RT.stop()

def measure_avg(n: int = 10, timeout_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    return _RT.measure_avg(n=n, timeout_sec=timeout_sec)
