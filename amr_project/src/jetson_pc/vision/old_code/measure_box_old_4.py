from __future__ import annotations
from typing import Optional, Dict, Any, List
import time
import threading

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

from .vision_config import VisionConfig, DEFAULT_VISION_CONFIG


# -----------------------------
# Local helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def poly_shrink_towards_center(poly4x2: np.ndarray, margin_px: float):
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    v = p - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return p - (v / norm) * margin_px


def depth_roi_stats(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray, cfg: VisionConfig):
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)

    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    d = d[(d > 0) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)]
    if d.size == 0:
        return 0.0, 0.0, 0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, int(d.size)


def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z_m: float):
    X = (cx - intr.ppx) / intr.fx * Z_m
    Y = (cy - intr.ppy) / intr.fy * Z_m
    return float(X), float(Y)  # meters


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
    angle = -angle
    return angle


def is_jump(prev, cur, cfg: VisionConfig):
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
    if not cfg.show_overlay:
        return

    line1 = f"cam X {Xmm:+.1f}  Y {Ymm:+.1f}  Z {Zmm:+.1f}  (mm)"
    line2 = f"angle {angle:+.2f} deg"

    x, y = 10, 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = cfg.overlay_font_scale
    th = cfg.overlay_thickness

    (w1, h1), _ = cv2.getTextSize(line1, font, fs, th)
    (w2, h2), _ = cv2.getTextSize(line2, font, fs, th)
    w = max(w1, w2)
    h = h1 + h2 + 18

    overlay = img.copy()
    cv2.rectangle(overlay, (6, 6), (6 + w + 12, 6 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    cv2.putText(img, line1, (x, y + 18), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(img, line2, (x, y + 18 + h1 + 6), font, fs, (255, 255, 255), th, cv2.LINE_AA)


# ============================================================
# Streamer + On-demand measurement (press 2 to request)
#  - ✅ 박스 실측크기 기반(Z_size) 제거
#  - ✅ Depth ROI(median)만 사용
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

        self._temporal = rs.temporal_filter()
        self._spatial = rs.spatial_filter()
        self._hole = rs.hole_filling_filter()

        # measure request state
        self._req_active: bool = False
        self._req_n: int = 10
        self._req_deadline: float = 0.0
        self._req_samples: List[List[float]] = []  # [[Xmm,Ymm,Zmm,angle],...]
        self._req_result: Optional[Dict[str, Any]] = None

        # overlay memo
        self._last_disp = {"Xmm": None, "Ymm": None, "Zmm": None, "angle": None}

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def start(self, cfg: VisionConfig = DEFAULT_VISION_CONFIG):
        with self._lock:
            if self._running:
                return

            self._cfg = cfg
            self._model = YOLO(cfg.model_path)

            self._pipeline = rs.pipeline()
            rs_cfg = rs.config()
            rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            rs_cfg.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
            profile = self._pipeline.start(rs_cfg)

            self._align = rs.align(rs.stream.color)

            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())

            # filters
            self._spatial.set_option(rs.option.filter_magnitude, 2)
            self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            self._spatial.set_option(rs.option.filter_smooth_delta, 20)

            if cfg.show_preview:
                cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cfg.preview_win_name, cfg.width, cfg.height)

            self._req_active = False
            self._req_samples.clear()
            self._req_result = None
            self._last_disp = {"Xmm": None, "Ymm": None, "Zmm": None, "angle": None}

            self._running = True
            self._thread = threading.Thread(target=self._loop, name="VisionStream", daemon=True)
            self._thread.start()

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

        try:
            if self._cfg.show_preview:
                cv2.destroyWindow(self._cfg.preview_win_name)
        except Exception:
            pass

        with self._lock:
            self._pipeline = None
            self._align = None
            self._model = None
            self._req_active = False
            self._req_samples.clear()
            self._req_result = None

    def measure_avg(self, n: int = 10, timeout_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        ✅ 스트리밍은 계속 유지한 채로,
        '지금부터' valid 샘플 n개 모아 평균 반환.
        """
        with self._lock:
            if not self._running:
                return None

            if timeout_sec is None:
                timeout_sec = float(getattr(self._cfg, "timeout_sec", 6.0))

            self._req_active = True
            self._req_n = int(max(1, n))
            self._req_deadline = time.time() + float(timeout_sec)
            self._req_samples = []
            self._req_result = None
            self._cv.notify_all()

            # wait
            while True:
                if self._req_result is not None:
                    out = dict(self._req_result)
                    self._req_result = None
                    return out

                if not self._running:
                    return None

                if time.time() >= self._req_deadline:
                    # timeout -> request cancel
                    self._req_active = False
                    self._req_samples = []
                    return None

                self._cv.wait(timeout=0.2)

    def _loop(self):
        cfg = self._cfg
        prev_valid = None
        consec_skips = 0

        while True:
            with self._lock:
                if not self._running:
                    break
                req_active = self._req_active
                req_n = self._req_n
                deadline = self._req_deadline

            # fetch frames
            try:
                assert self._pipeline is not None
                assert self._align is not None
                frames = self._pipeline.wait_for_frames()
                frames = self._align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                depth_frame = self._spatial.process(depth_frame).as_depth_frame()
                depth_frame = self._temporal.process(depth_frame).as_depth_frame()
                depth_frame = self._hole.process(depth_frame).as_depth_frame()

                frame = np.asanyarray(color_frame.get_data())
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                depth_u16 = np.asanyarray(depth_frame.get_data())
            except Exception:
                continue

            # yolo
            try:
                assert self._model is not None
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

            # candidates
            candidates = []
            img_cx = (cfg.width - 1) * 0.5
            img_cy = (cfg.height - 1) * 0.5

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

            # draw (항상 프리뷰엔 보여줌)
            if candidates:
                candidates.sort()
                dist2, _ncf, cf, ci, poly, cx_det_f, cy_det_f = candidates[0]
                cx = clamp(int(round(cx_det_f)), 0, cfg.width - 1)
                cy = clamp(int(round(cy_det_f)), 0, cfg.height - 1)

                poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [poly_i], True, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # measurement collection only when requested
            if req_active:
                # timeout cancel inside loop
                if time.time() >= deadline:
                    with self._lock:
                        self._req_active = False
                        self._req_samples = []
                        self._cv.notify_all()
                    req_active = False

                if req_active and candidates:
                    candidates.sort()
                    dist2, _ncf, cf, ci, poly, cx_det_f, cy_det_f = candidates[0]
                    cx = clamp(int(round(cx_det_f)), 0, cfg.width - 1)
                    cy = clamp(int(round(cy_det_f)), 0, cfg.height - 1)

                    poly_shrunk = poly_shrink_towards_center(poly, cfg.roi_margin_px)
                    poly_shrunk[:, 0] = np.clip(poly_shrunk[:, 0], 0, cfg.width - 1)
                    poly_shrunk[:, 1] = np.clip(poly_shrunk[:, 1], 0, cfg.height - 1)

                    # ✅ Depth ROI만 사용
                    Z_roi_m, mad_m, roi_n = depth_roi_stats(depth_u16, self._depth_scale, poly_shrunk, cfg)

                    depth_ok = (
                        (Z_roi_m > 0.0) and
                        (roi_n >= cfg.min_roi_pixels) and
                        (mad_m <= cfg.mad_thres_m)
                    )

                    # ✅ depth가 나쁘면 이번 프레임은 샘플로 안 씀 (fallback 없음)
                    if not depth_ok:
                        consec_skips += 1
                        continue

                    Z_use_m = Z_roi_m
                    Z_use_mm = Z_use_m * 1000.0

                    if not (cfg.z_range_mm[0] <= Z_use_mm <= cfg.z_range_mm[1]):
                        consec_skips += 1
                        continue

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

                        with self._lock:
                            if self._req_active:  # still active
                                self._req_samples.append([cur["Xmm"], cur["Ymm"], cur["Zmm"], cur["angle"]])

                                if len(self._req_samples) >= req_n:
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
                    else:
                        consec_skips += 1

            # overlay display memory
            if candidates and cfg.show_overlay and prev_valid is not None:
                self._last_disp.update(prev_valid)

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

                cv2.imshow(cfg.preview_win_name, vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    # ESC -> stop
                    self.stop()
                    break


# singleton runtime
_RT = _VisionRuntime()


# -----------------------------
# Public API (main에서 호출)
# -----------------------------
def start_stream(cfg: VisionConfig = DEFAULT_VISION_CONFIG):
    """스트리밍 시작(계속 프리뷰)"""
    _RT.start(cfg)


def stop_stream():
    """스트리밍 종료"""
    _RT.stop()


def is_running() -> bool:
    return _RT.is_running()


def measure_avg(n: int = 10, timeout_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    2번에서 호출:
    스트리밍은 계속 유지하고, n개 샘플 평균값만 반환
    """
    return _RT.measure_avg(n=n, timeout_sec=timeout_sec)



