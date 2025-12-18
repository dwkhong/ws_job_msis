import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# -------------------------------------------------------
# 모델 및 클래스
# -------------------------------------------------------
MODEL_PATH = r"C:\Users\rhdeh\Desktop\ws_job_msis\src\best_cls_20251216.pt"

CLASS_NAMES = [
    "box_0", "box_1", "box_2", "box_3",
    "box_4", "unknown"
]

def main():

    # ------------------------
    # YOLO11-CLS 모델 로드
    # ------------------------
    model = YOLO(MODEL_PATH)

    # ------------------------
    # RealSense 설정
    # ------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    width, height, fps = 1280, 720, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)

    print("[INFO] RealSense + YOLO11-CLS 실시간 테스트 시작")
    print("[INFO] ESC 누르면 종료")

    # ------------------------
    # 메인 루프
    # ------------------------
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # ---------------------------------------
        # YOLO11-CLS 예측
        # ---------------------------------------
        results = model.predict(frame, verbose=False)[0]

        cls_id = int(results.probs.top1)
        cls_conf = float(results.probs.top1conf)  # 확률 값 (0~1)

        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        score_text = f"{cls_conf * 100:.1f}%"   # 퍼센트로 표시

        # ---------------------------------------
        # 화면 오른쪽 위 예측 결과 표시
        # ---------------------------------------
        text = f"Pred: {cls_name} ({score_text})"
        cv2.putText(
            frame,
            text,
            (frame.shape[1] - 420, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ---------------------------------------
        # 화면 표시
        # ---------------------------------------
        cv2.imshow("RealSense YOLO11-CLS", frame)

        # ESC 누르면 종료
        if cv2.waitKey(1) == 27:
            print("[INFO] ESC pressed. Stop.")
            break

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

