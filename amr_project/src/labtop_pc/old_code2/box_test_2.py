import os
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import time
from datetime import datetime

MODEL_PATH = r"C:\Users\rhdeh\ws_job_msis\amr_project\src\labtop_pc\model\model_obj\best_obj_20251218.pt"

CONF_THRES = 0.85
IOU_THRES  = 0.75
IMGSZ      = 640

# -----------------------------
# ✅ 박스 실측 크기 (cm)
# -----------------------------
BOX_W_CM = 23.0
BOX_H_CM = 9.5

# -----------------------------
# ✅ Depth ROI 파라미터
# -----------------------------
ROI_MARGIN_PX   = 6
MIN_ROI_PIXELS  = 80
MAD_THRES_M     = 0.02
DEPTH_MIN_M     = 0.15
DEPTH_MAX_M     = 3.00

# -----------------------------
# ✅ 녹화 설정
# -----------------------------
RECORD_DIR = "recordings"
RECORD_FPS = 30
RECORD_FOURCC_TRY = ["mp4v", "avc1", "XVID"]  # 환경 따라 mp4v가 안 먹으면 대체
DRAW_CROSSHAIR = True


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
    d = d[(d > 0) & (d >= DEPTH_MIN_M) & (d <= DEPTH_MAX_M)]
    if d.size == 0:
        return 0.0, 0.0, 0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, int(d.size)


def edges_long_short_px(poly4x2: np.ndarray):
    p = poly4x2.astype(np.float32)
    edges = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    return float(max(edges)), float(min(edges))


def estimate_Z_from_size(poly4x2: np.ndarray, intr, W_cm: float, H_cm: float) -> float:
    long_px, short_px = edges_long_short_px(poly4x2)

    W_m = W_cm / 100.0
    H_m = H_cm / 100.0

    if W_m >= H_m:
        Z1 = (intr.fx * W_m) / max(long_px, 1e-6)
        Z2 = (intr.fy * H_m) / max(short_px, 1e-6)
    else:
        Z1 = (intr.fx * H_m) / max(long_px, 1e-6)
        Z2 = (intr.fy * W_m) / max(short_px, 1e-6)

    return float(0.5 * (Z1 + Z2))


def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z: float):
    X = (cx - intr.ppx) / intr.fx * Z
    Y = (cy - intr.ppy) / intr.fy * Z
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

    angle = float(np.degrees(np.arctan2(vx, vy)))
    return -angle


def draw_small_overlay(img, Xcm, Ycm, Zcm, angle_deg):
    line1 = f"XYZ(cm): {Xcm:+.2f}, {Ycm:+.2f}, {Zcm:+.2f}"
    line2 = f"angle(deg): {angle_deg:+.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    th = 2

    (w1, h1), _ = cv2.getTextSize(line1, font, fs, th)
    (w2, h2), _ = cv2.getTextSize(line2, font, fs, th)
    w = max(w1, w2)
    h = h1 + h2 + 18

    x, y = 10, 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w + 16, y + h + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    cv2.putText(img, line1, (x + 8, y + 22), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(img, line2, (x + 8, y + 22 + h1 + 8), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def open_writer(width, height):
    os.makedirs(RECORD_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # mp4 우선 시도, 안되면 avi로
    for fourcc_str in RECORD_FOURCC_TRY:
        ext = "mp4" if fourcc_str.lower() in ["mp4v", "avc1"] else "avi"
        out_path = os.path.join(RECORD_DIR, f"obb_centerpick_{ts}.{ext}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, float(RECORD_FPS), (width, height))
        if writer.isOpened():
            print(f"[REC] Recording -> {os.path.abspath(out_path)}  (fourcc={fourcc_str})")
            return writer, out_path

    print("[REC-WARN] VideoWriter open failed (codec 문제일 수 있음). 녹화 없이 진행합니다.")
    return None, None


def main():
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded:", MODEL_PATH)
    print("[INFO] ESC to quit")

    pipeline = rs.pipeline()
    config = rs.config()

    width, height, fps = 640, 480, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[INFO] depth_scale = {depth_scale:.8f} m/unit")

    temporal = rs.temporal_filter()
    spatial = rs.spatial_filter()
    hole = rs.hole_filling_filter()

    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    cv2.namedWindow("OBB Live (center pick)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OBB Live (center pick)", width, height)

    writer, out_path = open_writer(width, height)

    img_cx = (width - 1) * 0.5
    img_cy = (height - 1) * 0.5

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole.process(depth_frame)

            frame = np.asanyarray(color_frame.get_data())
            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            depth_u16 = np.asanyarray(depth_frame.get_data())

            vis = frame.copy()

            results = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
            r = results[0]

            # ✅ "화면 중앙과 가장 가까운 박스" 선택
            candidates = []  # (dist2_to_img_center, -conf, conf, cls, poly(4,2), cx, cy)
            if getattr(r, "obb", None) is not None and r.obb is not None:
                obb = r.obb
                if obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
                    polys = obb.xyxyxyxy.cpu().numpy()
                    confs = obb.conf.cpu().numpy().astype(float)
                    clss  = obb.cls.cpu().numpy().astype(int)

                    for poly8, cf, ci in zip(polys, confs, clss):
                        if float(cf) < CONF_THRES:
                            continue
                        poly = poly8.reshape(4, 2)
                        cx_det = float(np.mean(poly[:, 0]))
                        cy_det = float(np.mean(poly[:, 1]))
                        dx = cx_det - img_cx
                        dy = cy_det - img_cy
                        dist2 = dx * dx + dy * dy
                        candidates.append((dist2, -float(cf), float(cf), int(ci), poly, cx_det, cy_det))

            if candidates:
                candidates.sort()
                dist2, _ncf, cf, ci, poly, cx_f, cy_f = candidates[0]

                cx = int(round(cx_f))
                cy = int(round(cy_f))
                cx = clamp(cx, 0, width - 1)
                cy = clamp(cy, 0, height - 1)

                # OBB + center
                poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [poly_i], True, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

                # Z fuse
                poly_shrunk = poly_shrink_towards_center(poly, ROI_MARGIN_PX)
                poly_shrunk[:, 0] = np.clip(poly_shrunk[:, 0], 0, width - 1)
                poly_shrunk[:, 1] = np.clip(poly_shrunk[:, 1], 0, height - 1)

                Z_roi_m, mad_m, roi_n = depth_roi_stats(depth_u16, depth_scale, poly_shrunk)
                Z_size_m = estimate_Z_from_size(poly, intr, BOX_W_CM, BOX_H_CM)

                use_depth = (Z_roi_m > 0.0 and roi_n >= MIN_ROI_PIXELS and mad_m <= MAD_THRES_M)
                if use_depth:
                    alpha = clamp(0.85 - (mad_m / max(1e-6, MAD_THRES_M)) * 0.35, 0.55, 0.90)
                    Z_use_m = alpha * Z_roi_m + (1.0 - alpha) * Z_size_m
                else:
                    Z_use_m = Z_size_m

                if Z_use_m > 0:
                    X_m, Y_m = XY_from_pixel_and_Z(cx, cy, intr, Z_use_m)
                    angle = obb_angle_deg_upright0_rightplus(poly)
                    draw_small_overlay(vis, X_m * 100.0, Y_m * 100.0, Z_use_m * 100.0, angle)

            # 얇은 십자선(원하면 끄기)
            if DRAW_CROSSHAIR:
                cx0, cy0 = width // 2, height // 2
                cv2.line(vis, (cx0, 0), (cx0, height - 1), (255, 255, 255), 1)
                cv2.line(vis, (0, cy0), (width - 1, cy0), (255, 255, 255), 1)

            cv2.imshow("OBB Live (center pick)", vis)

            # ✅ 녹화 저장
            if writer is not None:
                writer.write(vis)

            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if out_path is not None:
            print(f"[REC] Saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
