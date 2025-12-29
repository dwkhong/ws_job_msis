# box_test_5.py
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
# ✅ 박스 실측 크기 (mm)
# -----------------------------
BOX_W_MM = 230.0
BOX_H_MM = 95.0

# -----------------------------
# ✅ Camera -> Gripper Offset (mm)
# -----------------------------
OFF_X_MM = -15.0
OFF_Y_MM = -70.0
OFF_Z_MM = -150.0

# ============================================================
# ✅ 로봇 베이스 yaw 보정
# ============================================================
BASE_YAW_OFFSET_DEG = -135

# ============================================================
# ✅ 축 부호 뒤집기 옵션
# ============================================================
FLIP_MOVE_X = False
FLIP_MOVE_Y = False

# -----------------------------
# ✅ 목표: 유효 샘플 N개
# -----------------------------
AVG_N = 30
TIMEOUT_SEC = 25.0

# -----------------------------
# ✅ Depth ROI 안정화 파라미터 (m)
# -----------------------------
ROI_MARGIN_PX  = 6
MIN_ROI_PIXELS = 120
MAD_THRES_M    = 0.020
DEPTH_MIN_M    = 0.15
DEPTH_MAX_M    = 3.00

# -----------------------------
# ✅ "이상한 값" 제거용 (mm)
# -----------------------------
Z_RANGE_MM = (150.0, 1200.0)
SIZE_REL_ERR_MAX = 0.25

JUMP_XY_MM    = 35.0
JUMP_Z_MM     = 60.0
JUMP_ANG_DEG  = 10.0

MAX_CONSEC_SKIPS_RESET = 15

# -----------------------------
# ✅ Preview + Overlay + Recording (2번 누를 때만)
# -----------------------------
SHOW_PREVIEW = True
PREVIEW_WIN_NAME = "OBB Preview (box_test_5)"

SHOW_OVERLAY_XYZ_ANGLE = True  # ✅ 좌상단에 값만 작게
OVERLAY_FONT_SCALE = 0.55
OVERLAY_THICKNESS = 2

RECORD_DURING_MEASURE = True   # ✅ 측정 루프 동안만 저장
RECORD_DIR = "recordings"
RECORD_FPS = 30
RECORD_FOURCC = "mp4v"         # mp4 저장
# 선택된 박스 정보 출력: 유효 샘플 accept될 때만 1줄
PRINT_SELECTED_EACH_ACCEPT = True


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
    edges = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
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

    L1_mm = (long_px * Z_use_m / intr.fx) * 1000.0
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
    if abs(cur["Xmm"] - prev["Xmm"]) > JUMP_XY_MM:
        return True
    if abs(cur["Ymm"] - prev["Ymm"]) > JUMP_XY_MM:
        return True
    if abs(cur["Zmm"] - prev["Zmm"]) > JUMP_Z_MM:
        return True
    if abs(cur["angle"] - prev["angle"]) > JUMP_ANG_DEG:
        return True
    return False


def rot2d_xy(x, y, deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return (c * x - s * y), (s * x + c * y)


def draw_overlay_xyz_angle(img, Xmm=None, Ymm=None, Zmm=None, angle=None):
    if not SHOW_OVERLAY_XYZ_ANGLE:
        return
    if Xmm is None or Ymm is None or Zmm is None or angle is None:
        return

    # 딱 필요한 값만, 작게
    line1 = f"X {Xmm:+.1f}  Y {Ymm:+.1f}  Z {Zmm:+.1f}  (mm)"
    line2 = f"angle {angle:+.2f} deg"

    x, y = 10, 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = OVERLAY_FONT_SCALE
    th = OVERLAY_THICKNESS

    # 배경 아주 작게(가독용)
    (w1, h1), _ = cv2.getTextSize(line1, font, fs, th)
    (w2, h2), _ = cv2.getTextSize(line2, font, fs, th)
    w = max(w1, w2)
    h = h1 + h2 + 18

    overlay = img.copy()
    cv2.rectangle(overlay, (6, 6), (6 + w + 12, 6 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    cv2.putText(img, line1, (x, y + 18), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(img, line2, (x, y + 18 + h1 + 6), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def main():
    model = YOLO(MODEL_PATH)

    pipeline = rs.pipeline()
    config = rs.config()

    width, height, fps = 640, 480, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    temporal = rs.temporal_filter()
    spatial = rs.spatial_filter()
    hole = rs.hole_filling_filter()

    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    accepted = []
    prev_valid = None
    consec_skips = 0
    t0 = time.time()

    # ✅ 녹화 준비(측정 세션 동안만)
    writer = None
    out_path = None
    if RECORD_DURING_MEASURE:
        os.makedirs(RECORD_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RECORD_DIR, f"box_test5_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*RECORD_FOURCC)
        writer = cv2.VideoWriter(out_path, fourcc, float(RECORD_FPS), (width, height))

    if SHOW_PREVIEW:
        cv2.namedWindow(PREVIEW_WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WIN_NAME, width, height)

    # 마지막으로 계산된 값(표시용)
    last_disp = {"Xmm": None, "Ymm": None, "Zmm": None, "angle": None}

    try:
        while True:
            if time.time() - t0 > TIMEOUT_SEC:
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

            results = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
            r = results[0]

            # 후보들: "이미지 중심에 가장 가까운 박스" 우선, 동률이면 conf 큰 것
            candidates = []  # (dist2, -conf, conf, cls, poly(4,2), cx, cy)
            img_cx = (width - 1) * 0.5
            img_cy = (height - 1) * 0.5

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

            if not candidates:
                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0

                # 프리뷰/녹화는 그냥 원본(frame)만
                vis = frame
                if SHOW_OVERLAY_XYZ_ANGLE and last_disp["Xmm"] is not None:
                    vis = vis.copy()
                    draw_overlay_xyz_angle(vis, last_disp["Xmm"], last_disp["Ymm"], last_disp["Zmm"], last_disp["angle"])

                if SHOW_PREVIEW:
                    cv2.imshow(PREVIEW_WIN_NAME, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                if writer is not None:
                    writer.write(vis)
                continue

            candidates.sort()
            dist2, _ncf, cf, ci, poly, cx_det_f, cy_det_f = candidates[0]
            chosen_dist_px = float(np.sqrt(dist2))
            num_boxes = len(candidates)

            cx = int(round(cx_det_f))
            cy = int(round(cy_det_f))
            cx = clamp(cx, 0, width - 1)
            cy = clamp(cy, 0, height - 1)

            # ===== 프리뷰(박스 + 빨간점 + 값 오버레이) =====
            vis = frame.copy()
            poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_i], True, (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # ===== 이하: 측정/필터 =====
            poly_shrunk = poly_shrink_towards_center(poly, ROI_MARGIN_PX)
            poly_shrunk[:, 0] = np.clip(poly_shrunk[:, 0], 0, width - 1)
            poly_shrunk[:, 1] = np.clip(poly_shrunk[:, 1], 0, height - 1)

            Z_roi_m, mad_m, roi_n = depth_roi_stats(depth_u16, depth_scale, poly_shrunk)
            Z_size_m = estimate_Z_from_size(poly, intr, BOX_W_MM, BOX_H_MM)

            depth_ok = (Z_roi_m > 0.0 and roi_n >= MIN_ROI_PIXELS and mad_m <= MAD_THRES_M)
            if depth_ok:
                alpha = clamp(0.85 - (mad_m / max(1e-6, MAD_THRES_M)) * 0.35, 0.55, 0.90)
                Z_use_m = alpha * Z_roi_m + (1.0 - alpha) * Z_size_m
            else:
                Z_use_m = Z_size_m

            Z_use_mm = Z_use_m * 1000.0
            if not (Z_RANGE_MM[0] <= Z_use_mm <= Z_RANGE_MM[1]):
                # 프리뷰/녹화는 계속(박스는 그려진 상태)
                if SHOW_OVERLAY_XYZ_ANGLE and last_disp["Xmm"] is not None:
                    draw_overlay_xyz_angle(vis, last_disp["Xmm"], last_disp["Ymm"], last_disp["Zmm"], last_disp["angle"])
                if SHOW_PREVIEW:
                    cv2.imshow(PREVIEW_WIN_NAME, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                if writer is not None:
                    writer.write(vis)

                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0
                continue

            X_m, Y_m = XY_from_pixel_and_Z(cx, cy, intr, Z_use_m)
            Z_m = Z_use_m
            dist_m = float(np.sqrt(X_m * X_m + Y_m * Y_m + Z_m * Z_m))
            angle = obb_angle_deg_upright0_rightplus(poly)

            cur = {
                "conf": float(cf),
                "cls": int(ci),
                "Xmm": X_m * 1000.0,
                "Ymm": Y_m * 1000.0,
                "Zmm": Z_m * 1000.0,
                "distmm": dist_m * 1000.0,
                "angle": float(angle),

                # 선택 정보(디버그용)
                "num_boxes": int(num_boxes),
                "chosen_center_px": (int(cx), int(cy)),
                "chosen_dist_to_img_center_px": float(chosen_dist_px),
            }

            ok_sz, *_ = size_consistency_check(poly, intr, Z_use_m, BOX_W_MM, BOX_H_MM, rel_err_max=SIZE_REL_ERR_MAX)
            if not ok_sz:
                # 프리뷰/녹화는 계속
                last_disp.update({"Xmm": cur["Xmm"], "Ymm": cur["Ymm"], "Zmm": cur["Zmm"], "angle": cur["angle"]})
                draw_overlay_xyz_angle(vis, cur["Xmm"], cur["Ymm"], cur["Zmm"], cur["angle"])
                if SHOW_PREVIEW:
                    cv2.imshow(PREVIEW_WIN_NAME, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                if writer is not None:
                    writer.write(vis)

                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0
                continue

            if is_jump(prev_valid, cur):
                last_disp.update({"Xmm": cur["Xmm"], "Ymm": cur["Ymm"], "Zmm": cur["Zmm"], "angle": cur["angle"]})
                draw_overlay_xyz_angle(vis, cur["Xmm"], cur["Ymm"], cur["Zmm"], cur["angle"])
                if SHOW_PREVIEW:
                    cv2.imshow(PREVIEW_WIN_NAME, vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return None
                if writer is not None:
                    writer.write(vis)

                consec_skips += 1
                if consec_skips >= MAX_CONSEC_SKIPS_RESET:
                    prev_valid = None
                    consec_skips = 0
                continue

            # ✅ accept
            consec_skips = 0
            prev_valid = cur
            accepted.append(cur)

            # 프리뷰 오버레이(현재 값)
            last_disp.update({"Xmm": cur["Xmm"], "Ymm": cur["Ymm"], "Zmm": cur["Zmm"], "angle": cur["angle"]})
            draw_overlay_xyz_angle(vis, cur["Xmm"], cur["Ymm"], cur["Zmm"], cur["angle"])

            if SHOW_PREVIEW:
                cv2.imshow(PREVIEW_WIN_NAME, vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    return None

            if writer is not None:
                writer.write(vis)

            if PRINT_SELECTED_EACH_ACCEPT:
                print(f"[{len(accepted)}/{AVG_N}] picked=1/{num_boxes}  dist_to_img_center={chosen_dist_px:.1f}px  conf={cf:.2f}  cls={ci}")

            if len(accepted) >= AVG_N:
                break

        # ✅ 평균
        cam_arr = np.array([[a["Xmm"], a["Ymm"], a["Zmm"], a["distmm"], a["angle"]] for a in accepted], dtype=np.float32)
        cam_mean = cam_arr.mean(axis=0)

        # ✅ gripper 평균
        g_list = []
        for a in accepted:
            gx = a["Xmm"] + OFF_X_MM
            gy = a["Ymm"] + OFF_Y_MM
            gz = a["Zmm"] + OFF_Z_MM
            g_list.append([gx, gy, gz])
        g_arr = np.array(g_list, dtype=np.float32)
        g_mean = g_arr.mean(axis=0)

        gx, gy, gz = float(g_mean[0]), float(g_mean[1]), float(g_mean[2])

        dx0 = -gx
        dy0 = +gy
        dz0 = -gz

        dx1, dy1 = rot2d_xy(dx0, dy0, BASE_YAW_OFFSET_DEG)

        if FLIP_MOVE_X:
            dx1 = -dx1
        if FLIP_MOVE_Y:
            dy1 = -dy1

        move_x_mm = float(dx1)
        move_y_mm = float(dy1)
        move_z_mm = float(dz0)

        # (원하면 여기서 파일명도 반환해줄 수 있음)
        # return dict에 video_path 추가하면, 컨트롤러에서 저장 경로 출력 가능
        ret = {
            "move_x_mm": move_x_mm,
            "move_y_mm": move_y_mm,
            "move_z_mm": move_z_mm,
            "angle_deg": float(cam_mean[4]),
        }
        if out_path is not None:
            ret["video_path"] = out_path
        return ret

    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
        if SHOW_PREVIEW:
            cv2.destroyWindow(PREVIEW_WIN_NAME)


if __name__ == "__main__":
    _ = main()
