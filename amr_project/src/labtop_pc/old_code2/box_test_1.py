import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import time

MODEL_PATH = r"C:\Users\rhdeh\ws_job_msis\amr_project\src\labtop_pc\model\model_obj\best_obj_20251218.pt"

CONF_THRES = 0.85
IOU_THRES  = 0.75
IMGSZ      = 640

# -----------------------------
# ✅ 박스 실측 크기 (cm)  <<<<< 여기만 너 박스에 맞게 수정
# (카메라가 보는 '앞면' 기준 가로/세로)
# -----------------------------
BOX_W_CM = 23.0
BOX_H_CM = 9.5

# -----------------------------
# ✅ Depth ROI 샘플링/안정화 파라미터
# -----------------------------
ROI_MARGIN_PX = 6      # OBB 마스크를 살짝 안쪽으로 줄이는 효과(경계 섞임 줄임)
MIN_ROI_PIXELS = 80    # ROI에서 depth 유효 픽셀 최소 개수
MAD_THRES_M = 0.02     # ROI depth 흔들림 허용(대충 2cm)
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 3.00


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def draw_hud(img, lines, x=10, y=10, line_h=24):
    pad = 8
    w = max([cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for s in lines] + [10])
    h = line_h * len(lines)
    x2, y2 = x + w + pad * 2, y + h + pad * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    ty = y + pad + 18
    for s in lines:
        cv2.putText(img, s, (x + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        ty += line_h


def poly_shrink_towards_center(poly4x2: np.ndarray, margin_px: float):
    """OBB 4점을 중심으로 margin만큼 안쪽으로 당김(경계 depth 섞임 완화)"""
    p = poly4x2.astype(np.float32)
    c = p.mean(axis=0, keepdims=True)
    v = p - c
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    p2 = p - (v / norm) * margin_px
    return p2


def depth_roi_stats(depth_u16: np.ndarray, depth_scale: float, poly4x2: np.ndarray):
    """
    poly 내부 픽셀들의 depth를 모아 median + MAD(robust spread) 계산
    return: (median_m, mad_m, valid_count)
    """
    h, w = depth_u16.shape[:2]
    poly = np.round(poly4x2).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)

    d = depth_u16[mask == 255].astype(np.float32) * depth_scale
    d = d[(d > 0) & (d >= DEPTH_MIN_M) & (d <= DEPTH_MAX_M)]

    if d.size == 0:
        return 0.0, 0.0, 0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))  # median absolute deviation
    return med, mad, int(d.size)


def estimate_Z_from_size(poly4x2: np.ndarray, intr, W_cm: float, H_cm: float) -> float:
    """박스 실측 크기(W/H)와 OBB 픽셀 크기로 Z(m) 추정"""
    p = poly4x2.astype(np.float32)
    edges = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    long_px = float(max(edges))
    short_px = float(min(edges))

    W_m = W_cm / 100.0
    H_m = H_cm / 100.0

    if W_m >= H_m:
        Z1 = (intr.fx * W_m) / max(long_px, 1e-6)
        Z2 = (intr.fy * H_m) / max(short_px, 1e-6)
    else:
        Z1 = (intr.fx * H_m) / max(long_px, 1e-6)
        Z2 = (intr.fy * W_m) / max(short_px, 1e-6)

    Z = 0.5 * (Z1 + Z2)
    return float(Z)


def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z: float):
    X = (cx - intr.ppx) / intr.fx * Z
    Y = (cy - intr.ppy) / intr.fy * Z
    return float(X), float(Y)


def main():
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded:", MODEL_PATH)
    print(f"[INFO] conf>={CONF_THRES}, iou={IOU_THRES}, imgsz={IMGSZ}")
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

    cv2.namedWindow("OBB Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OBB Live", width, height)

    last = {
        "ok": False,
        "Xcm": 0.0, "Ycm": 0.0, "Zcm": 0.0,
        "distcm": 0.0,
        "angle": 0.0,
        "conf": 0.0,
        "cls": -1,
        "t": 0.0,
        "cx": width // 2,
        "cy": height // 2,
        "Zdepth_cm": 0.0,
        "Zsize_cm": 0.0,
        "Zuse_cm": 0.0,
        "mad_cm": 0.0,
        "roi_n": 0,
    }

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
            status = "WAIT"

            results = model.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False
            )
            r = results[0]

            best = None  # (conf, cls, poly4x2)
            if getattr(r, "obb", None) is not None and r.obb is not None:
                obb = r.obb
                if obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
                    polys = obb.xyxyxyxy.cpu().numpy()
                    confs = obb.conf.cpu().numpy().astype(float)
                    clss  = obb.cls.cpu().numpy().astype(int)

                    keep = confs >= CONF_THRES
                    for poly8, cf, ci in zip(polys[keep], confs[keep], clss[keep]):
                        if best is None or cf > best[0]:
                            best = (cf, ci, poly8.reshape(4, 2))

            if best is not None:
                cf, ci, poly = best

                poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [poly_i], True, (0, 255, 0), 2)

                cx = int(np.mean(poly[:, 0]))
                cy = int(np.mean(poly[:, 1]))
                cx = clamp(cx, 0, width - 1)
                cy = clamp(cy, 0, height - 1)
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

                poly_shrunk = poly_shrink_towards_center(poly, ROI_MARGIN_PX)
                poly_shrunk[:, 0] = np.clip(poly_shrunk[:, 0], 0, width - 1)
                poly_shrunk[:, 1] = np.clip(poly_shrunk[:, 1], 0, height - 1)

                Z_roi_m, mad_m, roi_n = depth_roi_stats(depth_u16, depth_scale, poly_shrunk)
                Z_size_m = estimate_Z_from_size(poly, intr, BOX_W_CM, BOX_H_CM)

                use_depth = (Z_roi_m > 0.0 and roi_n >= MIN_ROI_PIXELS and mad_m <= MAD_THRES_M)
                if use_depth:
                    alpha = clamp(0.85 - (mad_m / MAD_THRES_M) * 0.35, 0.55, 0.90)
                    Z_use_m = alpha * Z_roi_m + (1.0 - alpha) * Z_size_m
                    status = "OK_FUSED"
                else:
                    Z_use_m = Z_size_m
                    status = "OK_SIZE_ONLY" if Z_use_m > 0 else "DEPTH_INVALID"

                if Z_use_m > 0.0:
                    X, Y = XY_from_pixel_and_Z(cx, cy, intr, Z_use_m)
                    Z = Z_use_m
                    dist = float(np.sqrt(X * X + Y * Y + Z * Z))
                    angle = obb_angle_deg_upright0_rightplus(poly)

                    last["ok"] = True
                    last["Xcm"], last["Ycm"], last["Zcm"] = X * 100.0, Y * 100.0, Z * 100.0
                    last["distcm"] = dist * 100.0
                    last["angle"] = angle
                    last["conf"] = float(cf)
                    last["cls"] = int(ci)
                    last["t"] = time.time()
                    last["cx"], last["cy"] = cx, cy
                    last["Zdepth_cm"] = Z_roi_m * 100.0
                    last["Zsize_cm"]  = Z_size_m * 100.0
                    last["Zuse_cm"]   = Z_use_m * 100.0
                    last["mad_cm"]    = mad_m * 100.0
                    last["roi_n"]     = int(roi_n)

                poly2_i = np.round(poly_shrunk).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [poly2_i], True, (255, 255, 0), 1)

            else:
                status = "NO_DET"

            # ✅ 화면 정중앙(카메라 중심) 아주 얇은 십자선(1px)
            cx0, cy0 = width // 2, height // 2
            cv2.line(vis, (cx0, 0), (cx0, height - 1), (255, 255, 255), 1)     # vertical
            cv2.line(vis, (0, cy0), (width - 1, cy0), (255, 255, 255), 1)      # horizontal

            # HUD
            age = time.time() - last["t"] if last["ok"] else 999.0
            stale = "STALE" if (not last["ok"] or age > 0.5) else "LIVE"

            hud_lines = [
                f"status: {status} / {stale} (age={age:.2f}s)",
                f"conf={last['conf']:.2f}  cls={last['cls']}",
                f"cam XYZ(cm)=({last['Xcm']:+.2f}, {last['Ycm']:+.2f}, {last['Zcm']:+.2f})",
                f"dist={last['distcm']:.2f} cm   angle={last['angle']:+.2f} deg",
                f"center px=({last['cx']},{last['cy']})",
                f"Z(depth/size/use)=( {last['Zdepth_cm']:.1f} / {last['Zsize_cm']:.1f} / {last['Zuse_cm']:.1f} ) cm",
                f"ROI n={last['roi_n']}  MAD={last['mad_cm']:.2f} cm  (thres={MAD_THRES_M*100:.1f}cm)",
                f"BOX(WxH)=( {BOX_W_CM:.1f} x {BOX_H_CM:.1f} ) cm",
            ]
            draw_hud(vis, hud_lines)

            cv2.imshow("OBB Live", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

