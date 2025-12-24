import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # --- 1) 스트림 활성화 (해상도/프레임레이트는 SDK가 자동 선택하게 둠) ---
    config.enable_stream(rs.stream.depth)   # depth
    config.enable_stream(rs.stream.color)   # RGB

    # 파이프라인 시작
    pipeline.start(config)

    # 컬러 프레임 기준으로 depth 정렬
    align = rs.align(rs.stream.color)

    print("Press 'q' to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # depth를 보기 좋게 컬러맵으로 변환
            depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

            # 해상도 안 맞으면 depth를 컬러 크기에 맞게 리사이즈
            if depth_colormap.shape[:2] != color_image.shape[:2]:
                depth_colormap = cv2.resize(
                    depth_colormap,
                    (color_image.shape[1], color_image.shape[0])
                )

            # 좌: 컬러 / 우: 뎁스
            stacked = np.hstack((color_image, depth_colormap))
            cv2.imshow("D405 Color (left) + Depth (right)", stacked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


