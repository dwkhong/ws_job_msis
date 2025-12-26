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
# ✅ 박스 실측 크기 (mm)
# -----------------------------
BOX_W_MM = 230.0
BOX_H_MM = 95.0

# -----------------------------
# ✅ Camera -> Gripper Offset (mm)
# -----------------------------
OFF_X_MM =  0
OFF_Y_MM = -70.0
OFF_Z_MM = -150

# -----------------------------
# ✅ 목표: 유효 샘플 N개
# -----------------------------
AVG_N = 10
TIMEOUT_SEC = 25.0

# -----------------------------
# ✅ Depth ROI 안정화 파라미터
# (depth는 RealSense 스케일상 m 단위로 처리)
# -----------------------------
ROI_MARGIN_PX  = 6
MIN_ROI_PIXELS = 120
MAD_THRES_M    = 0.020
DEPTH_MIN_M    = 0.15
DEPTH_MAX_M    = 3.00

# -----------------------------
# ✅ "이상한 값" 제거용 추가 방어 (mm 기준)
# -----------------------------
Z_RANGE_MM = (150.0, 1200.0)   # 15cm ~ 120cm
SIZE_REL_ERR_MAX = 0.25

JUMP_XY_MM    = 35.0           # 3.5cm
JUMP_Z_MM     = 60.0           # 6.0cm
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

def estimate_Z_from_size(poly4x2: np.ndarray, intr, W_mm: float, H_mm: float) -> float:
    long_px, short_px = edges_long_short_px(poly4x2)
    W_m = W_mm / 1000.0
    H_m = H_mm / 1000.0

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
    return float(X), float(Y)  # meters

def size_consistency_check(poly4x2, intr, Z_use_m, W_mm, H_mm, rel_err_max=0.25):
    long_px, short_px = edges_long_short_px(poly4x2)

    # pixel->meter->mm
    L1_mm = (long_px  * Z_use_m / intr.fx) * 1000.0
    L2_mm = (short_px * Z_use_m / intr.fy) * 1000.0

    W_big = max(W_mm, H_mm)
    H_sml = min(W_mm, H_mm)

    err1 = abs(L1_mm - W_big) / max(1e-6, W_big)
    err2 = abs(L2_mm - H_sml) / max(1e-6, H_sml)

    ok = (err1 <= rel_err_max) and (err2 <= rel_err_max)
    return ok, L1_mm, L2_mm, err1, err2

def is_jump(prev, cur):
    if prev is None:
        return False
    if abs(cur["Xmm"] - prev["Xmm"]) > JUMP_XY_MM: return True
    if abs(cur["Ymm"] - prev["Ymm"]) > JUMP_XY_MM: return True
    if abs(cur["Zmm"] - prev["Zmm"]) > JUMP_Z_MM:  return True
    if abs(cur["angle"] - prev["angle"]) > JUMP_ANG_DEG: return True
    return False

def main():
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded:", MODEL_PATH)
    print(f"[INFO] Need {AVG_N} valid samples. Timeout={TIMEOUT_SEC}s")
    print(f"[INFO] BOX(WxH) = {BOX_W_MM:.1f} x {BOX_H_MM:.1f} mm")
    print(f"[INFO] Offset cam->gripper (mm): X{OFF_X_MM:+.1f}, Y{OFF_Y_MM:+.1f}, Z{OFF_Z_MM:+.1f}")
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
                return None  # ✅ 실패 시 None 반환

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
            Z_size_m = estimate_Z_from_size(poly, intr, BOX_W_MM, BOX_H_MM)

            depth_ok = (Z_roi_m > 0.0 and roi_n >= MIN_ROI_PIXELS and mad_m <= MAD_THRES_M)

            if depth_ok:
                alpha = clamp(0.85 - (mad_m / max(1e-6, MAD_THRES_M)) * 0.35, 0.55, 0.90)
                Z_use_m = alpha * Z_roi_m + (1.0 - alpha) * Z_size_m
                z_mode = "FUSED"
            else:
                Z_use_m = Z_size_m
                z_mode = "SIZE"

            Z_use_mm = Z_use_m * 1000.0
            if not (Z_RANGE_MM[0] <= Z_use_mm <= Z_RANGE_MM[1]):
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
                "Xmm": X_m * 1000.0,
                "Ymm": Y_m * 1000.0,
                "Zmm": Z_m * 1000.0,
                "distmm": dist_m * 1000.0,
                "angle": float(angle),
                "Zdepth_mm": Z_roi_m * 1000.0,
                "Zsize_mm": Z_size_m * 1000.0,
                "Zuse_mm": Z_use_mm,
                "roi_n": roi_n,
                "mad_mm": mad_m * 1000.0,
                "mode": z_mode,
            }

            ok_sz, est_long_mm, est_short_mm, err1, err2 = size_consistency_check(
                poly, intr, Z_use_m, BOX_W_MM, BOX_H_MM, rel_err_max=SIZE_REL_ERR_MAX
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

            # gripper 좌표/거리 (mm)
            gx = cur["Xmm"] + OFF_X_MM
            gy = cur["Ymm"] + OFF_Y_MM
            gz = cur["Zmm"] + OFF_Z_MM
            gdist = float(np.sqrt(gx*gx + gy*gy + gz*gz))

            print(f"[{len(accepted)}/{AVG_N}] conf={cur['conf']:.2f} "
                  f"camXYZ(mm)=({cur['Xmm']:+.1f},{cur['Ymm']:+.1f},{cur['Zmm']:+.1f}) camDist(mm)={cur['distmm']:.1f}  "
                  f"gripXYZ(mm)=({gx:+.1f},{gy:+.1f},{gz:+.1f}) gripDist(mm)={gdist:.1f}  "
                  f"ang(deg)={cur['angle']:+.2f}  "
                  f"Z(depth/size/use)(mm)=({cur['Zdepth_mm']:.1f}/{cur['Zsize_mm']:.1f}/{cur['Zuse_mm']:.1f}) {cur['mode']}")

            if len(accepted) >= AVG_N:
                break

        # ✅ 여기부터 성공 케이스
        cam_arr = np.array([[a["Xmm"], a["Ymm"], a["Zmm"], a["distmm"], a["angle"],
                             a["Zdepth_mm"], a["Zsize_mm"], a["Zuse_mm"]] for a in accepted],
                           dtype=np.float32)
        cam_mean = cam_arr.mean(axis=0)
        cam_std  = cam_arr.std(axis=0)

        # ✅ gripper 배열/통계도 따로 계산 (mm)
        g_list = []
        for a in accepted:
            gx = a["Xmm"] + OFF_X_MM
            gy = a["Ymm"] + OFF_Y_MM
            gz = a["Zmm"] + OFF_Z_MM
            gdist = float(np.sqrt(gx*gx + gy*gy + gz*gz))
            g_list.append([gx, gy, gz, gdist])

        g_arr = np.array(g_list, dtype=np.float32)
        g_mean = g_arr.mean(axis=0)
        g_std  = g_arr.std(axis=0)

        # ✅ REAL MOVEMENT: 타겟으로 가려면 gripper 좌표의 "반대방향"으로 이동 (mm)
        move_x_mm = -float(g_mean[0])
        move_y_mm = +float(g_mean[1])
        move_z_mm = -float(g_mean[2])

        print("\n========== RESULT (AVERAGE over 10 valid) ==========")
        print(f"count : {AVG_N}")

        print(f"\n[CAMERA] (mm)")
        print(f"XYZ avg (mm)  : ({cam_mean[0]:+.1f}, {cam_mean[1]:+.1f}, {cam_mean[2]:+.1f})   "
              f"std=({cam_std[0]:.1f},{cam_std[1]:.1f},{cam_std[2]:.1f})")
        print(f"dist avg (mm) : {cam_mean[3]:.1f}   std={cam_std[3]:.1f}")

        print(f"\n[GRIPPER]  (offset X{OFF_X_MM:+.1f}, Y{OFF_Y_MM:+.1f}, Z{OFF_Z_MM:+.1f}) (mm)")
        print(f"XYZ avg (mm)  : ({g_mean[0]:+.1f}, {g_mean[1]:+.1f}, {g_mean[2]:+.1f})   "
              f"std=({g_std[0]:.1f},{g_std[1]:.1f},{g_std[2]:.1f})")
        print(f"dist avg (mm) : {g_mean[3]:.1f}   std={g_std[3]:.1f}")

        print(f"\n[REAL MOVEMENT] (to reach target) (mm)")
        print(f"moveXYZ(mm)   : ({move_x_mm:+.1f}, {move_y_mm:+.1f}, {move_z_mm:+.1f})")

        print(f"\n[OTHERS]")
        print(f"angle_avg (deg)       : {cam_mean[4]:+.2f}  std={cam_std[4]:.2f}")
        print(f"Z avg(depth/size/use) : ({cam_mean[5]:.1f}/{cam_mean[6]:.1f}/{cam_mean[7]:.1f}) mm")
        print("====================================================\n")

        # ✅✅✅ 여기서 “값을 가져갈 수 있도록” return 추가
        return {
            "move_x_mm": move_x_mm,
            "move_y_mm": move_y_mm,
            "move_z_mm": move_z_mm,
            "angle_deg": float(cam_mean[4]),

            # 필요하면 참고용 평균값도 같이 반환
            "gripper_mean_mm": (float(g_mean[0]), float(g_mean[1]), float(g_mean[2])),
            "camera_mean_mm":  (float(cam_mean[0]), float(cam_mean[1]), float(cam_mean[2])),
            "gripper_std_mm":  (float(g_std[0]), float(g_std[1]), float(g_std[2])),
            "camera_std_mm":   (float(cam_std[0]), float(cam_std[1]), float(cam_std[2])),
        }

    finally:
        pipeline.stop()

if __name__ == "__main__":
    _ = main()  # 단독 실행 시 출력만 하고 return은 버림
