from __future__ import annotations
from typing import Optional, Dict
import os
import time
import threading
import traceback

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ✅ rc처럼 모듈 자체를 cfg로 import
from . import vision_config as cfg


# =========================================================
# Helper Functions
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
    d = d[(d > 0) & (d >= cfg.DEPTH_MIN_M) & (d <= cfg.DEPTH_MAX_M)]

    if d.size == 0:
        return 0.0, 0.0, 0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, int(d.size)


def XY_from_pixel_and_Z(cx: float, cy: float, intr, Z_m: float):
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
    if vy < 0:
        vx, vy = -vx, -vy
    return -float(np.degrees(np.arctan2(vx, vy)))


def is_jump(prev: Optional[Dict[str, float]], cur: Dict[str, float]) -> bool:
    if prev is None:
        return False
    if abs(cur["move_x_mm"] - prev["move_x_mm"]) > cfg.JUMP_XY_MM:
        return True
    if abs(cur["move_y_mm"] - prev["move_y_mm"]) > cfg.JUMP_XY_MM:
        return True
    if abs(cur["move_z_mm"] - prev["move_z_mm"]) > cfg.JUMP_Z_MM:
        return True
    return False


# ============================================================
# Vision Runtime
# ============================================================
class _VisionRuntime:
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._model = None

        self._latest_sample: Optional[Dict[str, float]] = None
        self._collect_samples: list[Dict[str, float]] = []

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._measuring_evt = threading.Event()
        self._reset_evt = threading.Event()

    def start(self):
        if self._running:
            return

        # DISPLAY 없으면 preview 자동 OFF
        if bool(getattr(cfg, "SHOW_PREVIEW", False)):
            if os.environ.get("DISPLAY", "") == "":
                print("[Vision] DISPLAY가 없어 preview를 자동으로 끕니다.")
                try:
                    cfg.SHOW_PREVIEW = False  # 모듈 상수라도 파이썬에선 재할당 가능
                except Exception:
                    pass

        print(f"[Vision] YOLO 모델 로딩... ({cfg.MODEL_PATH})")
        try:
            self._model = YOLO(cfg.MODEL_PATH, task="obb")
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            return

        self._stop_evt.clear()
        self._measuring_evt.clear()
        self._reset_evt.clear()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Vision] D405 시스템 시작")

    def stop(self):
        self._running = False
        self._stop_evt.set()

        th = self._thread
        if th is not None:
            th.join(timeout=2.0)
            if th.is_alive():
                print("[Vision] stop(): thread가 2초 내 종료되지 않았습니다(백그라운드에서 종료될 수 있음).")
        self._thread = None
        print("[Vision] 시스템 종료")

    def measure_avg(self, n: int = 10, timeout: float = 5.0) -> Optional[Dict[str, float]]:
        if not self._running:
            return None

        n = int(cfg.AVG_N if n is None else n)
        timeout = float(cfg.TIMEOUT_SEC if timeout is None else timeout)

        with self._lock:
            self._collect_samples = []

        self._measuring_evt.set()
        self._reset_evt.set()

        print(f"[Vision] 측정 시작 (목표: {n}개, 히스토리 리셋)")
        start_t = time.time()

        while time.time() - start_t < timeout:
            with self._lock:
                cnt = len(self._collect_samples)
            if cnt >= n:
                break
            time.sleep(0.02)

        self._measuring_evt.clear()

        with self._lock:
            samples = list(self._collect_samples)

        if len(samples) == 0:
            print("[Vision] 측정 실패: 데이터 없음")
            return None

        if len(samples) < n:
            print(f"[Vision] 경고: 목표 {n}개 중 {len(samples)}개만 수집됨 (timeout={timeout}s)")

        arr = np.array(
            [[s["move_x_mm"], s["move_y_mm"], s["move_z_mm"], s["angle_deg"]] for s in samples],
            dtype=np.float32,
        )
        m = np.mean(arr, axis=0)

        return {
            "move_x_mm": float(m[0]),
            "move_y_mm": float(m[1]),
            "move_z_mm": float(m[2]),
            "angle_deg": float(m[3]),
        }

    def _loop(self):
        pipeline = None
        last_log_t = 0.0

        try:
            pipeline = rs.pipeline()
            config = rs.config()

            STREAM_W = int(cfg.WIDTH)
            STREAM_H = int(cfg.HEIGHT)
            FPS = int(cfg.FPS)

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

            if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                try:
                    cv2.namedWindow(cfg.PREVIEW_WIN_NAME, cv2.WINDOW_NORMAL)
                except Exception:
                    print("[Vision] preview window 생성 실패. SHOW_PREVIEW=False로 진행합니다.")
                    try:
                        cfg.SHOW_PREVIEW = False
                    except Exception:
                        pass

            prev_valid: Optional[Dict[str, float]] = None

            roi_margin_px = float(getattr(cfg, "ROI_MARGIN_PX", 5.0))
            min_roi_pixels = int(getattr(cfg, "MIN_ROI_PIXELS", 50))

            while self._running and (not self._stop_evt.is_set()):
                if self._reset_evt.is_set():
                    prev_valid = None
                    self._reset_evt.clear()

                try:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    img = np.asanyarray(color_frame.get_data())

                    d_frame = spatial.process(depth_frame)
                    d_frame = temporal.process(d_frame)
                    d_u16 = np.asanyarray(d_frame.get_data())

                    intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                    vis = img.copy()

                    results = self._model.predict(
                        img,
                        imgsz=int(cfg.IMGSZ),
                        conf=float(cfg.CONF_THRES),
                        iou=float(cfg.IOU_THRES),
                        verbose=False,
                    )
                    r = results[0]

                    best_sample: Optional[Dict[str, float]] = None

                    if hasattr(r, "obb") and r.obb is not None:
                        candidates = []
                        for i, conf in enumerate(r.obb.conf):
                            if float(conf) < float(cfg.CONF_THRES):
                                continue

                            poly = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)

                            poly_s = poly_shrink_towards_center(poly, roi_margin_px)

                            poly_s[:, 0] = np.clip(poly_s[:, 0], 0, STREAM_W - 1)
                            poly_s[:, 1] = np.clip(poly_s[:, 1], 0, STREAM_H - 1)

                            z_m, mad, count = depth_roi_stats(d_u16, depth_scale, poly_s)

                            if z_m > 0 and count > min_roi_pixels:
                                cx = float(np.mean(poly[:, 0]))
                                cy = float(np.mean(poly[:, 1]))

                                dist_from_center = float(
                                    np.sqrt((cx - IMG_CENTER_X) ** 2 + (cy - IMG_CENTER_Y) ** 2)
                                )
                                angle = obb_angle_deg_upright0_rightplus(poly)

                                candidates.append(
                                    {
                                        "z_m": float(z_m),
                                        "poly": poly,
                                        "angle": float(angle),
                                        "dist_center": dist_from_center,
                                        "cx": cx,
                                        "cy": cy,
                                    }
                                )

                        if candidates:
                            candidates.sort(key=lambda x: x["dist_center"])
                            best = candidates[0]

                            tx, ty = XY_from_pixel_and_Z(best["cx"], best["cy"], intr, best["z_m"])
                            raw = {
                                "move_x_mm": float(tx * 1000.0),
                                "move_y_mm": float(ty * 1000.0),
                                "move_z_mm": float(best["z_m"] * 1000.0),
                                "angle_deg": float(best["angle"]),
                            }

                            if not is_jump(prev_valid, raw):
                                prev_valid = raw
                                best_sample = raw

                                if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                                    txt = (
                                        f"XYZ: {raw['move_x_mm']:.0f}, {raw['move_y_mm']:.0f}, "
                                        f"{raw['move_z_mm']:.0f}"
                                    )
                                    cv2.polylines(vis, [np.int32(best["poly"])], True, (0, 255, 0), 2)
                                    cv2.circle(vis, (int(best["cx"]), int(best["cy"])), 5, (0, 255, 0), -1)
                                    cv2.putText(
                                        vis,
                                        txt,
                                        (int(best["cx"]), int(best["cy"])),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 0),
                                        2,
                                    )
                            else:
                                if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                                    cv2.polylines(vis, [np.int32(best["poly"])], True, (0, 0, 255), 2)
                                    cv2.putText(
                                        vis,
                                        "JUMP",
                                        (int(best["cx"]), int(best["cy"])),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 0, 255),
                                        2,
                                    )

                    if best_sample is not None:
                        with self._lock:
                            self._latest_sample = best_sample
                            if self._measuring_evt.is_set():
                                self._collect_samples.append(best_sample)

                    if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                        cv2.line(vis, (int(IMG_CENTER_X), 0), (int(IMG_CENTER_X), STREAM_H), (255, 255, 0), 1)
                        cv2.line(vis, (0, int(IMG_CENTER_Y)), (STREAM_W, int(IMG_CENTER_Y)), (255, 255, 0), 1)

                        if self._measuring_evt.is_set():
                            with self._lock:
                                cnum = len(self._collect_samples)
                            cv2.putText(
                                vis,
                                f"Collecting: {cnum}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                            )

                        cv2.imshow(cfg.PREVIEW_WIN_NAME, vis)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break

                except Exception:
                    now = time.time()
                    if now - last_log_t > 2.0:
                        print("[Vision] loop exception:")
                        print(traceback.format_exc())
                        last_log_t = now
                    time.sleep(0.01)

        finally:
            try:
                if pipeline is not None:
                    pipeline.stop()
            except Exception:
                pass

            if bool(getattr(cfg, "SHOW_PREVIEW", False)):
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

# ============================================================
# 외부 API
# ============================================================
_RT = _VisionRuntime()

_LAST_MEASURE_AVG = None

def start_stream():
    _RT.start()

def stop_stream():
    _RT.stop()

def toggle_stream():
    if is_running():
        stop_stream()
    else:
        start_stream()

def measure_avg(n=None, timeout_sec=None):
    return _RT.measure_avg(n=n, timeout=timeout_sec)

def get_last_measure_avg():
    return _LAST_MEASURE_AVG

def cmd_measure_avg(n=None, timeout_sec=None):
    global _LAST_MEASURE_AVG  # ✅ 이거 반드시 있어야 함

    if not is_running():
        print("[Vision] Not running. (1번으로 Vision On)")
        return None

    res = measure_avg(n=n, timeout_sec=timeout_sec)
    _LAST_MEASURE_AVG = res

    if res is None:
        print("[Vision] measure_avg failed (no samples)")
        return None

    print(
        f"[Vision] avg XYZ(mm)=({res['move_x_mm']:.1f}, {res['move_y_mm']:.1f}, {res['move_z_mm']:.1f}) "
        f"angle(deg)={res['angle_deg']:.2f}"
    )
    return res

def is_running() -> bool:
    th = _RT._thread
    return bool(_RT._running and th is not None and th.is_alive())
