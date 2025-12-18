import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import time

MODEL_PATH = r"C:\Users\rhdeh\Desktop\ws_job_msis\src\model_obj\best_obj_20251218.pt"

CONF_THRES = 0.85
IOU_THRES  = 0.75
IMGSZ      = 640

# -----------------------------
# ✅ 박스 실측 크기 (cm)
# -----------------------------
BOX_W_CM = 23.0
BOX_H_CM = 9.5

# -----------------------------
# ✅ Camera -> Gripper Offset (cm)
# -----------------------------
OFF_X_CM = 0.0
OFF_Y_CM = -7.0
OFF_Z_CM = -18.0

# -----------------------------
# ✅ 목표: 유효 샘플 N개
# -----------------------------
AVG_N = 10
TIMEOUT_SEC = 25.0

# -----------------------------
# ✅ Depth ROI 안정화 파라미터
# -----------------------------
ROI_MARGIN_PX  = 6
MIN_ROI_PIXELS = 120
MAD_THRES_M    = 0.020
DEPTH_MIN_M    = 0.15
DEPTH_MAX_M    = 3.00

# -----------------------------
# ✅ "이상한 값" 제거용 추가 방어
# -----------------------------
Z_RANGE_CM = (15.0, 120.0)
SIZE_REL_ERR_MAX = 0.25

JUMP_XY_CM    = 3.5
JUMP_Z_CM     = 6.0
JUMP_ANG_DEG  = 10.0

MAX_CONSEC_SKIPS_RESET = 15

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

def edges_long_short_px(poly4x2: np.ndarray):
    p = poly4x2.astype(np.float32)
    edges = [np.linalg.norm(p[(i+1) % 4] - p[i]) for i in range(4)]
    long_px = float(max(edges))
    short_px = float(min(edges))
    return long_px, short_px

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

    return float(0.5 * (Z1 + Z2))  # meters

def XY_from_pixel_and_Z(cx: int, cy: int, intr, Z_m: float):
    X = (cx - intr.ppx) / intr.fx * Z_m
    Y = (cy - intr.ppy) / intr.fy * Z_m
    return float(X), float(Y)

def size_consistency_check(poly4x2, intr, Z_use_m, W_cm, H_cm, rel_err_max=0.25):
    long_px, short_px = edges_long_short_px(poly4x2)

    L1_cm = (long_px  * Z_use_m / intr.fx) * 100.0
    L2_cm = (short_px * Z_use_m / intr.fy) * 100.0

    W_big = max(W_cm, H_cm)
    H_sml = min(W_cm, H_cm)

    err1 = abs(L1_cm - W_big) / max(1e-6, W_big)
    err2 = abs(L2_cm - H_sml) / max(1e-6, H_sml)

    ok = (err1 <= rel_err_max) and (err2 <= rel_err_max)
    return ok, L1_cm, L2_cm, err1, err2

def is_jump(prev, cur):
    if prev is None:
        return False
    if abs(cur["Xcm"] - prev["Xcm"]) > JUMP_XY_CM: return True
    if abs(cur["Ycm"] - prev["Ycm"]) > JUMP_XY_CM: return True
    if abs(cur["Zcm"] - prev["Zcm"]) > JUMP_Z_CM:  return True
    if abs(cur["angle"] - prev["angle"]) > JUMP_ANG_DEG: return True
    return False

def main():
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded:", MODEL_PATH)
    print(f"[INFO] Need {AVG_N} valid samples. Timeout={TIMEOUT_SEC}s")
    print(f"[INFO] BOX(WxH) = {BOX_W_CM:.1f} x {BOX_H_CM:.1f} cm")
    print(f"[INFO] Offset cam->gripper (cm): X{OFF_X_CM:+.1f}, Y{OFF_Y_CM:+.1f}, Z{OFF_Z_CM:+.1f}")
    print(f"[INFO] conf>={CONF_THRES}, iou={IOU_THRES}, imgsz={IMGSZ}\n")

    pipeline = rs.pipeline()
    config = rs.config()

    width, height, fps = 640, 480, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[INFO] depth_scale = {depth_scale:.8f} m/unit\n")

    temporal = rs.temporal_filter()
    spatial  = rs.spatial_filter()
    hole     = rs.hole_filling_filter()

    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    accepted = []
    prev_valid = None
    consec_skips = 0
    t0 = time.time()

    try:
        while True:
            if time.time() - t0 > TIMEOUT_SEC:
                print("\n[FAIL] Timeout: not enough valid samples.")
                break

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

            results = model.predict(
                frame, imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES, verbose=False
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
                            best = (float(cf), int(ci), poly8.reshape(4, 2))

            if best is None:
                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0
                continue

            cf, ci, poly = best

            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            cx = clamp(cx, 0, width - 1)
            cy = clamp(cy, 0, height - 1)

            poly_shrunk = poly_shrink_towards_center(poly, ROI_MARGIN_PX)
            poly_shrunk[:, 0] = np.clip(poly_shrunk[:, 0], 0, width - 1)
            poly_shrunk[:, 1] = np.clip(poly_shrunk[:, 1], 0, height - 1)

            Z_roi_m, mad_m, roi_n = depth_roi_stats(depth_u16, depth_scale, poly_shrunk)
            Z_size_m = estimate_Z_from_size(poly, intr, BOX_W_CM, BOX_H_CM)

            depth_ok = (Z_roi_m > 0.0 and roi_n >= MIN_ROI_PIXELS and mad_m <= MAD_THRES_M)

            if depth_ok:
                alpha = clamp(0.85 - (mad_m / max(1e-6, MAD_THRES_M)) * 0.35, 0.55, 0.90)
                Z_use_m = alpha * Z_roi_m + (1.0 - alpha) * Z_size_m
                z_mode = "FUSED"
            else:
                Z_use_m = Z_size_m
                z_mode = "SIZE"

            Z_use_cm = Z_use_m * 100.0
            if not (Z_RANGE_CM[0] <= Z_use_cm <= Z_RANGE_CM[1]):
                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0
                continue

            X_m, Y_m = XY_from_pixel_and_Z(cx, cy, intr, Z_use_m)
            Z_m = Z_use_m
            dist_m = float(np.sqrt(X_m*X_m + Y_m*Y_m + Z_m*Z_m))
            angle = obb_angle_deg_upright0_rightplus(poly)

            cur = {
                "conf": cf,
                "cls": ci,
                "Xcm": X_m * 100.0,
                "Ycm": Y_m * 100.0,
                "Zcm": Z_m * 100.0,
                "distcm": dist_m * 100.0,
                "angle": float(angle),
                "Zdepth_cm": Z_roi_m * 100.0,
                "Zsize_cm": Z_size_m * 100.0,
                "Zuse_cm": Z_use_cm,
                "roi_n": roi_n,
                "mad_cm": mad_m * 100.0,
                "mode": z_mode,
            }

            ok_sz, est_long_cm, est_short_cm, err1, err2 = size_consistency_check(
                poly, intr, Z_use_m, BOX_W_CM, BOX_H_CM, rel_err_max=SIZE_REL_ERR_MAX
            )
            if not ok_sz:
                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0
                continue

            if is_jump(prev_valid, cur):
                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0
                continue

            consec_skips = 0
            prev_valid = cur
            accepted.append(cur)

            # gripper 좌표/거리(그리퍼 원점 기준)
            gx = cur["Xcm"] + OFF_X_CM
            gy = cur["Ycm"] + OFF_Y_CM
            gz = cur["Zcm"] + OFF_Z_CM
            gdist = float(np.sqrt(gx*gx + gy*gy + gz*gz))

            print(f"[{len(accepted)}/{AVG_N}] conf={cur['conf']:.2f} "
                  f"camXYZ=({cur['Xcm']:+.2f},{cur['Ycm']:+.2f},{cur['Zcm']:+.2f}) camDist={cur['distcm']:.2f}  "
                  f"gripXYZ=({gx:+.2f},{gy:+.2f},{gz:+.2f}) gripDist={gdist:.2f}  "
                  f"ang={cur['angle']:+.2f}  "
                  f"Z(depth/size/use)=({cur['Zdepth_cm']:.1f}/{cur['Zsize_cm']:.1f}/{cur['Zuse_cm']:.1f}) {cur['mode']}")

            if len(accepted) >= AVG_N:
                break

        if len(accepted) >= AVG_N:
            cam_arr = np.array([[a["Xcm"], a["Ycm"], a["Zcm"], a["distcm"], a["angle"],
                                 a["Zdepth_cm"], a["Zsize_cm"], a["Zuse_cm"]] for a in accepted],
                               dtype=np.float32)
            cam_mean = cam_arr.mean(axis=0)
            cam_std  = cam_arr.std(axis=0)

            # ✅ gripper 배열/통계도 따로 계산
            g_list = []
            for a in accepted:
                gx = a["Xcm"] + OFF_X_CM
                gy = a["Ycm"] + OFF_Y_CM
                gz = a["Zcm"] + OFF_Z_CM
                gdist = float(np.sqrt(gx*gx + gy*gy + gz*gz))
                g_list.append([gx, gy, gz, gdist])

            g_arr = np.array(g_list, dtype=np.float32)
            g_mean = g_arr.mean(axis=0)
            g_std  = g_arr.std(axis=0)

            print("\n========== RESULT (AVERAGE over 10 valid) ==========")
            print(f"count : {AVG_N}")

            print(f"\n[CAMERA]")
            print(f"XYZ avg (cm)  : ({cam_mean[0]:+.2f}, {cam_mean[1]:+.2f}, {cam_mean[2]:+.2f})   "
                  f"std=({cam_std[0]:.2f},{cam_std[1]:.2f},{cam_std[2]:.2f})")
            print(f"dist avg (cm) : {cam_mean[3]:.2f}   std={cam_std[3]:.2f}")

            print(f"\n[GRIPPER]  (offset X{OFF_X_CM:+.1f}, Y{OFF_Y_CM:+.1f}, Z{OFF_Z_CM:+.1f})")
            print(f"XYZ avg (cm)  : ({g_mean[0]:+.2f}, {g_mean[1]:+.2f}, {g_mean[2]:+.2f})   "
                  f"std=({g_std[0]:.2f},{g_std[1]:.2f},{g_std[2]:.2f})")
            print(f"dist avg (cm) : {g_mean[3]:.2f}   std={g_std[3]:.2f}")

            print(f"\n[OTHERS]")
            print(f"angle_avg (deg)       : {cam_mean[4]:+.2f}  std={cam_std[4]:.2f}")
            print(f"Z avg(depth/size/use) : ({cam_mean[5]:.1f}/{cam_mean[6]:.1f}/{cam_mean[7]:.1f}) cm")
            print("====================================================\n")

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
