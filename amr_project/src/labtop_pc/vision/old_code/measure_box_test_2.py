from __future__ import annotations
from typing import Optional, Dict, Any, List
import time
import threading

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

from .vision_config import VisionConfig, DEFAULT_VISION_CONFIG


# =========================================================
# 2. Helper Functions (PnP 제거됨, Depth 직접 계산)
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def poly_shrink_towards_center(poly4x2: np.ndarray, margin_px: float):
    """ 박스 테두리 노이즈(배경)를 피하기 위해 폴리곤을 안쪽으로 살짝 축소 """
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    v = p - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return p - (v / norm) * margin_px


def depth_roi_stats(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray, cfg: VisionConfig):
    """ 폴리곤 내부의 Depth 값들의 중앙값(Median)을 구함 """
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)

    # 마스크 영역의 깊이 값 추출
    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    
    # 유효 범위 필터링
    d = d[(d > 0) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)]
    
    if d.size == 0:
        return 0.0, 0.0, 0

    # 중앙값(Median) 사용 -> 노이즈에 강함
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) # 편차
    return med, mad, int(d.size)


def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z_m: float):
    """ Pinhole Model로 3D 좌표 계산 """
    X = (cx - intr.ppx) / intr.fx * Z_m
    Y = (cy - intr.ppy) / intr.fy * Z_m
    return float(X), float(Y)  # meters


def obb_angle_deg_upright0_rightplus(poly4x2: np.ndarray) -> float:
    """ 회전된 박스(OBB)의 각도 계산 """
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
    angle = -angle
    return angle


def is_jump(prev, cur, cfg: VisionConfig):
    """ 값이 갑자기 튀는지 검사 """
    if prev is None:
        return False
    if abs(cur["Xmm"] - prev["Xmm"]) > cfg.jump_xy_mm:
        return True
    if abs(cur["Ymm"] - prev["Ymm"]) > cfg.jump_xy_mm:
        return True
    if abs(cur["Zmm"] - prev["Zmm"]) > cfg.jump_z_mm:
        return True
    if abs(cur["angle"] - prev["angle"]) > cfg.jump_ang_deg:
        return True
    return False


def draw_overlay_xyz_angle(img, Xmm, Ymm, Zmm, angle, cfg: VisionConfig):
    """ 화면 좌상단에 현재 좌표 표시 """
    if not cfg.show_overlay:
        return

    if Xmm is None or Ymm is None or Zmm is None:
        return

    line1 = f"Depth X {Xmm:+.1f}  Y {Ymm:+.1f}  Z {Zmm:+.1f} (mm)"
    line2 = f"Angle {angle:+.2f} deg"

    x, y = 10, 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = cfg.overlay_font_scale
    th = cfg.overlay_thickness

    (w1, h1), _ = cv2.getTextSize(line1, font, fs, th)
    (w2, h2), _ = cv2.getTextSize(line2, font, fs, th)
    w = max(w1, w2)
    h = h1 + h2 + 18

    # 검은 배경 박스
    overlay = img.copy()
    cv2.rectangle(overlay, (6, 6), (6 + w + 12, 6 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # 텍스트
    cv2.putText(img, line1, (x, y + 18), font, fs, (0, 255, 255), th, cv2.LINE_AA) # 노란색
    cv2.putText(img, line2, (x, y + 18 + h1 + 6), font, fs, (255, 255, 255), th, cv2.LINE_AA) # 흰색


# ============================================================
# 3. Vision Runtime Class
# ============================================================
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
        self._depth_scale: float = 0.001

        # 필터 (속도 저하 원인이면 주석 처리 유지)
        self._temporal = rs.temporal_filter()
        self._spatial = rs.spatial_filter()
        self._hole = rs.hole_filling_filter()

        self._req_active: bool = False
        self._req_n: int = 10
        self._req_deadline: float = 0.0
        self._req_samples: List[List[float]] = []
        self._req_result: Optional[Dict[str, Any]] = None

        self._last_disp = {"Xmm": None, "Ymm": None, "Zmm": None, "angle": None}

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def start(self, cfg: VisionConfig = DEFAULT_VISION_CONFIG):
        with self._lock:
            if self._running:
                return

            self._cfg = cfg
            print(f"Loading Model: {cfg.model_path}")
            self._model = YOLO(cfg.model_path)

            print("Starting RealSense Pipeline...")
            self._pipeline = rs.pipeline()
            rs_cfg = rs.config()
            rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            rs_cfg.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
            profile = self._pipeline.start(rs_cfg)

            self._align = rs.align(rs.stream.color)

            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())

            # 필터 옵션
            self._spatial.set_option(rs.option.filter_magnitude, 2)
            self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            self._spatial.set_option(rs.option.filter_smooth_delta, 20)

            self._req_active = False
            self._req_samples = []
            self._req_result = None
            self._last_disp = {"Xmm": None, "Ymm": None, "Zmm": None, "angle": None}

            self._running = True
            self._thread = threading.Thread(target=self._loop, name="VisionStream", daemon=True)
            self._thread.start()
            print(">>> Vision System Started (Depth Only).")

    def stop(self):
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._cv.notify_all()

        if self._thread is not None:
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass
        self._thread = None

        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        except Exception:
            pass

        with self._lock:
            self._pipeline = None
            self._align = None
            self._model = None
            self._req_active = False
            self._req_samples = []
            self._req_result = None
        print(">>> Vision System Stopped.")

    def measure_avg(self, n: int = 10, timeout_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._running:
                return None

            if timeout_sec is None:
                timeout_sec = float(getattr(self._cfg, "timeout_sec", 6.0))

            # ★ [수정됨] 매 요청마다 상태를 확실하게 초기화
            self._req_samples = []
            self._req_result = None
            self._req_n = int(max(1, n))
            self._req_deadline = time.time() + float(timeout_sec)
            self._req_active = True
            
            self._cv.notify_all()

            # 결과 대기 (Blocking)
            while True:
                if self._req_result is not None:
                    out = dict(self._req_result)
                    self._req_result = None
                    self._req_active = False # 확실히 끔
                    return out

                if not self._running:
                    return None

                if time.time() >= self._req_deadline:
                    self._req_active = False
                    self._req_samples = []
                    return None # 타임아웃

                self._cv.wait(timeout=0.1)

    def _loop(self):
        cfg = self._cfg
        prev_valid = None
        consec_skips = 0

        # UI 창 생성
        if cfg.show_preview:
            try:
                cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cfg.preview_win_name, cfg.width, cfg.height)
            except Exception as e:
                print(f"[Vision] Window create failed: {e}")

        try:
            while True:
                # ★ 루프 시작 시점에 Lock을 걸지 않고, 필요한 순간에만 상태를 확인함
                if not self._running:
                    break

                # 1. 프레임 획득
                try:
                    frames = self._pipeline.wait_for_frames()
                    frames = self._align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    # 필터 적용 (속도 문제시 주석 유지)
                    depth_frame = self._spatial.process(depth_frame).as_depth_frame()
                    depth_frame = self._temporal.process(depth_frame).as_depth_frame()
                    depth_frame = self._hole.process(depth_frame).as_depth_frame()

                    frame = np.asanyarray(color_frame.get_data())
                    intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                    depth_u16 = np.asanyarray(depth_frame.get_data())
                except Exception:
                    time.sleep(0.01)
                    continue

                # 2. YOLO 추론
                try:
                    results = self._model.predict(
                        frame,
                        imgsz=cfg.imgsz,
                        conf=cfg.conf_thres,
                        iou=cfg.iou_thres,
                        verbose=False
                    )
                    r = results[0] if results else None
                except Exception:
                    r = None

                vis = frame.copy()

                # 3. 후보 박스 찾기
                candidates = []
                img_cx = (cfg.width - 1) * 0.5
                img_cy = (cfg.height - 1) * 0.5
                current_data_valid = None 

                if r is not None and getattr(r, "obb", None) is not None and r.obb is not None:
                    obb = r.obb
                    if obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
                        polys = obb.xyxyxyxy.cpu().numpy()
                        confs = obb.conf.cpu().numpy().astype(float)
                        clss  = obb.cls.cpu().numpy().astype(int)

                        for poly8, cf, ci in zip(polys, confs, clss):
                            if float(cf) < cfg.conf_thres:
                                continue
                            poly = poly8.reshape(4, 2)
                            cx_det = float(np.mean(poly[:, 0]))
                            cy_det = float(np.mean(poly[:, 1]))
                            dx = cx_det - img_cx
                            dy = cy_det - img_cy
                            dist2 = dx * dx + dy * dy
                            candidates.append((dist2, -float(cf), float(cf), int(ci), poly, cx_det, cy_det))

                # 4. 좌표 계산 및 그리기
                if candidates:
                    candidates.sort()
                    dist2, _ncf, cf, ci, poly, cx_det_f, cy_det_f = candidates[0]
                    cx = clamp(int(round(cx_det_f)), 0, cfg.width - 1)
                    cy = clamp(int(round(cy_det_f)), 0, cfg.height - 1)

                    # Draw
                    poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [poly_i], True, (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

                    # Shrink ROI
                    poly_shrunk = poly_shrink_towards_center(poly, cfg.roi_margin_px)
                    poly_shrunk[:, 0] = np.clip(poly_shrunk[:, 0], 0, cfg.width - 1)
                    poly_shrunk[:, 1] = np.clip(poly_shrunk[:, 1], 0, cfg.height - 1)

                    # Get Depth
                    Z_roi_m, mad_m, roi_n = depth_roi_stats(depth_u16, self._depth_scale, poly_shrunk, cfg)

                    depth_ok = (
                        (Z_roi_m > 0.0) and
                        (roi_n >= cfg.min_roi_pixels) and
                        (mad_m <= cfg.mad_thres_m)
                    )

                    if depth_ok:
                        Z_use_m = Z_roi_m
                        Z_use_mm = Z_use_m * 1000.0

                        if (cfg.z_range_mm[0] <= Z_use_mm <= cfg.z_range_mm[1]):
                            # Calculate XYZ
                            X_m, Y_m = XY_from_pixel_and_Z(cx, cy, intr, Z_use_m)
                            angle = obb_angle_deg_upright0_rightplus(poly)

                            cur = {
                                "Xmm": X_m * 1000.0,
                                "Ymm": Y_m * 1000.0,
                                "Zmm": Z_use_m * 1000.0,
                                "angle": float(angle),
                            }

                            if not is_jump(prev_valid, cur, cfg):
                                prev_valid = cur
                                consec_skips = 0
                                current_data_valid = cur # 유효 데이터!
                                self._last_disp.update(cur)
                            else:
                                consec_skips += 1
                        else:
                            consec_skips += 1
                    else:
                        consec_skips += 1
                
                # 5. 측정 요청 처리 (Lock 안에서 상태 확인 및 데이터 추가)
                with self._lock:
                    if self._req_active:
                        # 타임아웃 체크
                        if time.time() >= self._req_deadline:
                            self._req_active = False
                            self._req_samples = []
                            self._cv.notify_all()
                        
                        # 데이터 수집
                        elif current_data_valid:
                            self._req_samples.append([
                                current_data_valid["Xmm"], 
                                current_data_valid["Ymm"], 
                                current_data_valid["Zmm"], 
                                current_data_valid["angle"]
                            ])

                            if len(self._req_samples) >= self._req_n:
                                arr = np.array(self._req_samples, dtype=np.float32)
                                out = {
                                    "cam_x_mm": float(arr[:, 0].mean()),
                                    "cam_y_mm": float(arr[:, 1].mean()),
                                    "cam_z_mm": float(arr[:, 2].mean()),
                                    "angle_deg": float(arr[:, 3].mean()),
                                }
                                self._req_result = out
                                self._req_active = False
                                self._req_samples = []
                                self._cv.notify_all()

                # 6. UI Update
                if cfg.show_preview:
                    if cfg.show_overlay and self._last_disp["Xmm"] is not None:
                        draw_overlay_xyz_angle(
                            vis,
                            self._last_disp["Xmm"],
                            self._last_disp["Ymm"],
                            self._last_disp["Zmm"],
                            self._last_disp["angle"],
                            cfg
                        )
                    
                    # 현재 측정 중인지 확인 (Lock 없이 읽어도 UI용이라 괜찮음)
                    if self._req_active:
                         cnt = len(self._req_samples)
                         cv2.putText(vis, f"MEASURING... {cnt}/{self._req_n}", 
                                     (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    cv2.imshow(cfg.preview_win_name, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        self.stop()
                        break
        
        finally:
            if cfg.show_preview:
                try: cv2.destroyAllWindows()
                except: pass


# singleton runtime
_RT = _VisionRuntime()


# -----------------------------
# 4. Public API
# -----------------------------
def start_stream(cfg: VisionConfig = DEFAULT_VISION_CONFIG):
    _RT.start(cfg)

def stop_stream():
    _RT.stop()

def is_running() -> bool:
    return _RT.is_running()

def measure_avg(n: int = 10, timeout_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    return _RT.measure_avg(n=n, timeout_sec=timeout_sec)