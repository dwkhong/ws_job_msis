import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # ✅ Color 스트림만 1280x720, 30fps로 설정
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)

    # OpenCV 창 크게 보기
    cv2.namedWindow("D405 Color", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("D405 Color", 1280, 720)

    print("Press 'q' to quit")

    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())

        cv2.imshow("D405 Color", color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
