import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import sys

def get_next_filename(base_name="color_output", ext="mp4"):
    idx = 1
    while True:
        filename = f"{base_name}_{idx}.{ext}"
        if not os.path.exists(filename):
            return filename
        idx += 1

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    width, height, fps = 1280, 720, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)

    # 프리뷰 창
    cv2.namedWindow("D405 Color", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("D405 Color", width, height)

    save_path = get_next_filename()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"[INFO] Recording started → {save_path}")
    print("[INFO] Press Ctrl+C or press ESC to stop.\n")

    start_time = time.time()
    SEGMENT_DURATION = 120  # 2분

    dot_last = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            # 영상 저장
            out.write(color)

            # 화면 표시
            cv2.imshow("D405 Color", color)
            key = cv2.waitKey(1)
            if key == 27:  # ESC 키
                print("\n[INFO] ESC pressed. Stopping recording...")
                break

            # 터미널 점 출력
            if time.time() - dot_last >= 0.1:
                print(".", end="", flush=True)
                dot_last = time.time()

            # 2분 단위 분할
            if time.time() - start_time >= SEGMENT_DURATION:
                out.release()
                print("\n[INFO] 2 minutes reached, starting new file...")

                save_path = get_next_filename()
                out = VideoWriter = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                print(f"[INFO] New file → {save_path}")

                start_time = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Recording stopped by user (Ctrl+C).")

    finally:
        out.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] Last file saved: {save_path}")

if __name__ == "__main__":
    main()

