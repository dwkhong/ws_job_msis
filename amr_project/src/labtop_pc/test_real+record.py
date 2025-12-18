import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# -------------------------------------------------------
# 모델 및 클래스
# -------------------------------------------------------
MODEL_PATH = r"C:\Users\rhdeh\Desktop\ws_job_msis\src\model_cls\best_cls_20251217_2.pt"

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

    # ------------------------
    # 녹화 설정
    # ------------------------
    os.makedirs("recordings", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"recordings/cls_record_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print("[INFO] RealSense + YOLO11-CLS 실시간 테스트 시작")
    print(f"[INFO] Recording → {video_path}")
    print("[INFO] ESC 누르면 종료")

    # ------------------------
    # 표시 옵션
    # ------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    pad = 10          # 화면 모서리 여백
    bg_pad = 8        # 텍스트 배경 박스 여백 (원하면 0으로)

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
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        # ---------------------------------------
        # ✅ 오른쪽 위 "딱" 붙여서 표시 (글자 길이 자동 반영)
        # ---------------------------------------
        (tw, th), baseline = cv2.getTextSize(cls_name, font, font_scale, thickness)

        # 오른쪽 위 시작점(텍스트 기준 좌하단 좌표)
        x = frame.shape[1] - pad - tw
        y = pad + th

        # (선택) 배경 박스 그리기: 글자가 더 잘 보임
        x1 = x - bg_pad
        y1 = y - th - bg_pad
        x2 = x + tw + bg_pad
        y2 = y + baseline + bg_pad
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)  # 검정 배경

        # 텍스트
        cv2.putText(
            frame,
            cls_name,
            (x, y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA
        )

        # ---------------------------------------
        # 녹화 (오버레이 포함)
        # ---------------------------------------
        writer.write(frame)

        # ---------------------------------------
        # 화면 표시
        # ---------------------------------------
        cv2.imshow("RealSense YOLO11-CLS", frame)

        if cv2.waitKey(1) == 27:
            print("[INFO] ESC pressed. Stop.")
            break

    # ------------------------
    # 종료 처리
    # ------------------------
    writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Video saved:", video_path)


if __name__ == "__main__":
    main()
