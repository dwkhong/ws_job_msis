import cv2
import numpy as np
from ultralytics import YOLO
import time

VIDEO_PATH = r"C:\Users\rhdeh\Desktop\ws_job_msis\src\video_20251217_2\color_output_3.mp4"
MODEL_PATH = r"C:\Users\rhdeh\Desktop\ws_job_msis\src\model_cls\best_cls_20251217_2.pt"

CLASS_NAMES = [
    "box_0", "box_1", "box_2", "box_3",
    "box_4", "unknown"
]

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] 영상 열기 실패")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_jump = int(fps * 10)  # 10초

    start_wall = time.time()
    paused = False
    speed = 1.0   # 기준 속도 (최대)

    print("[INFO] SPACE: pause | a/d: ±10초 | s: 느리게 | w: 다시 빠르게 | ESC")

    while True:

        # -----------------------------
        # Pause 상태
        # -----------------------------
        if paused:
            key = cv2.waitKey(30)
            if key == 32:  # SPACE
                paused = False
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                start_wall = time.time() - (cur / fps / speed)

            elif key == ord('a') or key == ord('d'):
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if key == ord('a'):
                    cur = max(cur - frame_jump, 0)
                else:
                    cur = cur + frame_jump
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
                start_wall = time.time() - (cur / fps / speed)

            elif key == 27:
                break
            continue

        # -----------------------------
        # 재생 상태
        # -----------------------------
        ret, frame = cap.read()
        if not ret:
            break

        cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # -----------------------------
        # 시간 동기화
        # -----------------------------
        target_time = (cur_frame / fps) / speed
        while time.time() - start_wall < target_time:
            time.sleep(0.001)

        # YOLO 추론
        results = model.predict(frame, verbose=False)[0]
        cls_id = int(results.probs.top1)
        cls_score = float(results.probs.top1conf)
        cls_name = CLASS_NAMES[cls_id]

        cv2.putText(
            frame,
            f"Pred: {cls_name} ({cls_score*100:.1f}%)",
            (frame.shape[1] - 420, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Speed: {speed:.1f}x",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

        cv2.imshow("YOLO11-CLS Player", frame)

        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == 32:
            paused = True
        elif key == ord('a'):
            new_frame = max(cur_frame - frame_jump, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            start_wall = time.time() - (new_frame / fps / speed)
        elif key == ord('d'):
            new_frame = cur_frame + frame_jump
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            start_wall = time.time() - (new_frame / fps / speed)
        elif key == ord('s'):  # 느리게
            speed = max(round(speed - 0.1, 1), 0.1)
            start_wall = time.time() - (cur_frame / fps / speed)
        elif key == ord('w'):  # 다시 빠르게 (1.0까지만)
            speed = min(round(speed + 0.1, 1), 1.0)
            start_wall = time.time() - (cur_frame / fps / speed)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





