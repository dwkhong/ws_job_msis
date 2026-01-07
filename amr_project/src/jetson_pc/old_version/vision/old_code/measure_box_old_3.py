from __future__ import annotations
from typing import Optional, Dict, Any
import time

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
    cv2.rectangle(overlay, (6, 6), (6 + w + 12), (6 + h), (0, 0, 0), -1)
    # ✅ 위 줄에 오타가 있어서 아래처럼 고쳐야 함 (rectangle 인자)
    # cv2.rectangle(overlay, (6, 6), (6 + w + 12, 6 + h), (0, 0, 0), -1)
    # 하지만 여기서는 원본 스타일 유지 + 안전하게 수정해서 사용
    overlay = img.copy()
    cv2.rectangle(overlay, (6, 6), (6 + w + 12, 6 + h), (0, 0, 0), -1)

    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    cv2.putText(img, line1, (x, y + 18), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(img, line2, (x, y + 18 + h1 + 6), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def measure_box(cfg: VisionConfig = DEFAULT_VISION_CONFIG) -> Optional[Dict[str, Any]]:
    model = YOLO(cfg.model_path)

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
    rs_cfg.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)

    profile = pipeline.start(rs_cfg)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    temporal = rs.temporal_filter()
    spatial = rs.spatial_filter()
    hole = rs.hole_filling_filter()

    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    accepted_xyza = []  # [[Xmm,Ymm,Zmm,angle], ...]
    prev_valid = None
    consec_skips = 0
    t0 = time.time()

    if cfg.show_preview:
        cv2.namedWindow(cfg.preview_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.preview_win_name, cfg.width, cfg.height)

    last_disp = {"Xmm": None, "Ymm": None, "Zmm": None, "angle": None}

    try:
        while True:
            if time.time() - t0 > cfg.timeout_sec:
                return None

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = spatial.process(depth_frame).as_depth_frame()
            depth_frame = temporal.process(depth_frame).as_depth_frame()
            depth_frame = hole.process(depth_frame).as_depth_frame()

            frame = np.asanyarray(color_frame.get_data())
            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            depth_u16 = np.asanyarray(depth_frame.get_data())

            results = model.predict(frame, imgsz=cfg.imgsz, conf=cfg.conf_thres, iou=cfg.iou_thres, verbose=False)
            r = results[0]

            candidates = []
            img_cx = (cfg.width - 1) * 0.5
            img_cy = (cfg.height - 1) * 0.5

            if getattr(r, "obb", None) is not None and r.obb is not None:
                obb = r.obb
                if obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
                    polys = obb.xyxyxyxy.cpu().numpy()
                    confs = obb.conf.cpu().numpy().astype(float)
                    clss = obb.cls.cpu().numpy().astype(int)

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

            if not candidates:
                consec_skips += 1
                if consec_skips >= cfg.max_consec_skips_reset:
                    prev_valid = None
                    consec_skips = 0

                if cfg.show_preview:
                    vis = frame
                    if cfg.show_overlay and last_disp["Xmm"] is not None:
                        vis = vis.copy()
                        draw_overlay_xyz_angle(
                            vis,
                            last_disp["Xmm"],
                            last_disp["Ymm"],
                            last_disp["Zmm"],
                            last_disp["angle"],
                            cfg,
                        )
                    cv2.imshow(cfg.preview_win_name, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                continue

            candidates.sort()
            dist2, _ncf, cf, ci, poly, cx_det_f, cy_det_f = candidates[0]
            chosen_dist_px = float(np.sqrt(dist2))
            num_boxes = len(candidates)

            cx = clamp(int(round(cx_det_f)), 0, cfg.width - 1)
            cy = clamp(int(round(cy_det_f)), 0, cfg.height - 1)

            vis = frame.copy()
            poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_i], True, (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            poly_shrunk = poly_shrink_towards_center(poly, cfg.roi_margin_px)
            poly_shrunk[:, 0] = np.clip(poly_shrunk[:, 0], 0, cfg.width - 1)
            poly_shrunk[:, 1] = np.clip(poly_shrunk[:, 1], 0, cfg.height - 1)

            # ✅ Depth ROI만 사용 (박스 크기 기반 제거)
            Z_roi_m, mad_m, roi_n = depth_roi_stats(depth_u16, depth_scale, poly_shrunk, cfg)

            depth_ok = (Z_roi_m > 0.0 and roi_n >= cfg.min_roi_pixels and mad_m <= cfg.mad_thres_m)
            if not depth_ok:
                consec_skips += 1
                if consec_skips >= cfg.max_consec_skips_reset:
                    prev_valid = None
                    consec_skips = 0

                # 오버레이는 마지막 유효값 유지
                if cfg.show_preview:
                    if cfg.show_overlay and last_disp["Xmm"] is not None:
                        draw_overlay_xyz_angle(
                            vis,
                            last_disp["Xmm"],
                            last_disp["Ymm"],
                            last_disp["Zmm"],
                            last_disp["angle"],
                            cfg,
                        )
                    cv2.imshow(cfg.preview_win_name, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                continue

            Z_use_m = Z_roi_m
            Z_use_mm = Z_use_m * 1000.0

            if not (cfg.z_range_mm[0] <= Z_use_mm <= cfg.z_range_mm[1]):
                consec_skips += 1
                if consec_skips >= cfg.max_consec_skips_reset:
                    prev_valid = None
                    consec_skips = 0

                if cfg.show_preview:
                    if cfg.show_overlay and last_disp["Xmm"] is not None:
                        draw_overlay_xyz_angle(
                            vis,
                            last_disp["Xmm"],
                            last_disp["Ymm"],
                            last_disp["Zmm"],
                            last_disp["angle"],
                            cfg,
                        )
                    cv2.imshow(cfg.preview_win_name, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                continue

            X_m, Y_m = XY_from_pixel_and_Z(cx, cy, intr, Z_use_m)
            angle = obb_angle_deg_upright0_rightplus(poly)

            cur = {
                "Xmm": X_m * 1000.0,
                "Ymm": Y_m * 1000.0,
                "Zmm": Z_use_m * 1000.0,
                "angle": float(angle),
            }

            if is_jump(prev_valid, cur, cfg):
                consec_skips += 1
                if consec_skips >= cfg.max_consec_skips_reset:
                    prev_valid = None
                    consec_skips = 0

                # (선택) 오버레이는 갱신할지 말지 취향인데, 원래 코드 흐름 유지하려면 갱신
                last_disp.update(cur)

                if cfg.show_preview:
                    draw_overlay_xyz_angle(vis, cur["Xmm"], cur["Ymm"], cur["Zmm"], cur["angle"], cfg)
                    cv2.imshow(cfg.preview_win_name, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                continue

            # accept
            consec_skips = 0
            prev_valid = cur
            accepted_xyza.append([cur["Xmm"], cur["Ymm"], cur["Zmm"], cur["angle"]])
            last_disp.update(cur)

            if cfg.show_preview:
                draw_overlay_xyz_angle(vis, cur["Xmm"], cur["Ymm"], cur["Zmm"], cur["angle"], cfg)
                cv2.imshow(cfg.preview_win_name, vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    return None

            if cfg.print_selected_each_accept:
                print(
                    f"[{len(accepted_xyza)}/{cfg.avg_n}] picked=1/{num_boxes}  "
                    f"dist_to_img_center={chosen_dist_px:.1f}px  conf={cf:.2f}  cls={ci}"
                )

            if len(accepted_xyza) >= cfg.avg_n:
                break

        cam_mean = np.mean(np.array(accepted_xyza, dtype=np.float32), axis=0)
        return {
            "cam_x_mm": float(cam_mean[0]),
            "cam_y_mm": float(cam_mean[1]),
            "cam_z_mm": float(cam_mean[2]),
            "angle_deg": float(cam_mean[3]),
        }

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if cfg.show_preview:
            try:
                cv2.destroyWindow(cfg.preview_win_name)
            except Exception:
                pass
