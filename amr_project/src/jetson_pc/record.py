#!/usr/bin/env python3
# realsense_record_color.py
# - RealSense 컬러 스트림을 화면에 띄우면서 mp4로 저장
# - ESC 또는 Ctrl+C로 종료
# - 2분(기본)마다 파일 자동 분할 저장
#
# 사용 예)
#   python3 realsense_record_color.py
#   python3 realsense_record_color.py --device-sn 123456789 --w 1280 --h 720 --fps 30 --seg 120 --base color_output

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def get_next_filename(base_name="color_output", ext="mp4"):
    idx = 1
    while True:
        filename = f"{base_name}_{idx}.{ext}"
        if not os.path.exists(filename):
            return filename
        idx += 1


def pick_stream_params(usb_desc: str, w: int, h: int, fps: int):
    """
    usb_desc가 2.x면 대역폭 이슈로 고해상도 실패할 수 있어서 기본값을 낮춰 시작.
    단, 사용자가 --w/--h/--fps를 지정했으면 그 값을 우선.
    """
    if w and h and fps:
        return w, h, fps

    usb_desc = (usb_desc or "").strip()
    if usb_desc.startswith("2"):
        return 1280, 720, 30
    return 1280, 720, 30


def list_devices(ctx: rs.context):
    devs = ctx.query_devices()
    out = []
    for d in devs:
        try:
            out.append(
                {
                    "name": d.get_info(rs.camera_info.name),
                    "sn": d.get_info(rs.camera_info.serial_number),
                    "usb": d.get_info(rs.camera_info.usb_type_descriptor),
                }
            )
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-sn", default="", help="특정 장치 SN으로 고정 (여러 대 연결 시 권장)")
    ap.add_argument("--w", type=int, default=0, help="폭 (기본: USB에 따라 자동)")
    ap.add_argument("--h", type=int, default=0, help="높이 (기본: USB에 따라 자동)")
    ap.add_argument("--fps", type=int, default=0, help="FPS (기본: USB에 따라 자동)")
    ap.add_argument("--seg", type=int, default=120, help="분할 저장 길이(초) (기본: 120)")
    ap.add_argument("--base", default="color_output", help="저장 파일 베이스명")
    ap.add_argument("--ext", default="mp4", choices=["mp4", "avi"], help="저장 확장자(mp4/avi)")
    ap.add_argument("--no-preview", action="store_true", help="미리보기 창 끄기(헤드리스)")
    args = ap.parse_args()

    ctx = rs.context()
    dev_list = list_devices(ctx)
    if not dev_list:
        raise RuntimeError("No RealSense devices found.")

    # 선택 장치
    chosen = None
    if args.device_sn:
        for d in dev_list:
            if d["sn"] == args.device_sn:
                chosen = d
                break
        if chosen is None:
            raise RuntimeError(f"Device SN not found: {args.device_sn}")
    else:
        chosen = dev_list[0]

    print(f"[INFO] Device: {chosen['name']} / SN: {chosen['sn']} / USB: {chosen['usb']}")

    width, height, fps = pick_stream_params(chosen["usb"], args.w, args.h, args.fps)
    print(f"[INFO] Stream request: color {width}x{height}@{fps}")

    pipeline = rs.pipeline()
    config = rs.config()

    # 여러 대일 때도 안전하게 장치 고정
    config.enable_device(chosen["sn"])
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        pipeline.start(config)
    except Exception as e:
        print("\n[ERROR] pipeline.start(config) failed.")
        print("        Likely unsupported stream or USB bandwidth issue.")
        print("        Try lower resolution/fps or check USB3 port/cable.")
        raise

    # 코덱 설정
    if args.ext == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    save_path = get_next_filename(base_name=args.base, ext=args.ext)
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    if not out.isOpened():
        pipeline.stop()
        raise RuntimeError(f"VideoWriter open failed: {save_path}")

    win_name = "RealSense Color Preview"
    if not args.no_preview:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, width, height)

    print(f"[INFO] Recording started → {save_path}")
    print("[INFO] Press Ctrl+C or ESC to stop.\n")

    segment_sec = int(args.seg)
    start_time = time.time()
    dot_last = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            out.write(color)

            if not args.no_preview:
                cv2.imshow(win_name, color)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("\n[INFO] ESC pressed. Stopping recording...")
                    break

            # 진행 표시
            if time.time() - dot_last >= 0.1:
                print(".", end="", flush=True)
                dot_last = time.time()

            # 분할 저장
            if segment_sec > 0 and (time.time() - start_time) >= segment_sec:
                out.release()
                print("\n[INFO] Segment reached, starting new file...")

                save_path = get_next_filename(base_name=args.base, ext=args.ext)
                out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise RuntimeError(f"VideoWriter open failed: {save_path}")
                print(f"[INFO] New file → {save_path}")
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Recording stopped by user (Ctrl+C).")

    finally:
        try:
            out.release()
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        if not args.no_preview:
            cv2.destroyAllWindows()
        print(f"[INFO] Last file saved: {save_path}")


if __name__ == "__main__":
    main()
