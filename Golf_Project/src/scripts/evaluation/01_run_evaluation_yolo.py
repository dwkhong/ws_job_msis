#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/run_evaluation.py

import os
import sys
from pathlib import Path

# ✅ ultralytics 로그 무음(원하면 유지)
os.environ["ULTRALYTICS_QUIET"] = "1"
# os.environ["RICH_PROGRESS_BAR"] = "0"  # 진행바도 끄고 싶으면 주석 해제

# 현재 파일이 .../src/scripts/ 아래면 parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.evaluation_yolo import run_evaluation


# ==============================
# 실행 시 옵션 입력 (인터랙티브)
# ==============================
def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """
    y / n 입력받는 헬퍼.
    엔터만 치면 default 사용.
    """
    while True:
        base = "Y/n" if default else "y/n"
        s = input(f"{prompt} [{base}]: ").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes", "1"):
            return True
        if s in ("n", "no", "0"):
            return False
        print("  ▶ y / n 중 하나로 입력해 주세요.")


def ask_float(prompt: str, default: float = 0.001, vmin: float = 0.0, vmax: float = 1.0) -> float:
    """
    float 입력받는 헬퍼. (예: CONF threshold)
    엔터만 치면 default 사용.
    """
    while True:
        s = input(f"{prompt} (기본값={default}, 범위 {vmin}~{vmax}): ").strip()
        if s == "":
            return default
        try:
            v = float(s)
        except ValueError:
            print("  ▶ 숫자로 입력해 주세요.")
            continue
        if v < vmin or v > vmax:
            print(f"  ▶ {vmin} ~ {vmax} 사이의 값을 입력해 주세요.")
            continue
        return v


if __name__ == "__main__":

    # ==============================
    # ✅ 여기 아래 “물어보는 흐름” 원복
    # ==============================
    print("============== Evaluation Options ==============")

    draw_save = ask_yes_no("1) 시각화 이미지를 파일로 저장할까요?", default=False)

    if draw_save:
        draw_only_divot = ask_yes_no("2) Divot만 그릴까요? (y: Divot만, n: 전체 클래스)", default=False)
        draw_score_min = ask_float("3) 시각화에 사용할 최소 confidence (CONF)", default=0.001, vmin=0.0, vmax=1.0)
        print(f"\n▶ 시각화 활성화: SAVE={draw_save}, ONLY_DIVOT={draw_only_divot}, DRAW_CONF≥{draw_score_min}")
    else:
        draw_only_divot = False
        draw_score_min = 0.001
        print("\n▶ 시각화 비활성화 (이미지 저장 안 함)")

    print("===============================================\n")

    # ==============================
    # ✅ 기존처럼 run_evaluation() 호출 (기능은 다 살아있음)
    # ==============================
    result = run_evaluation(
        engine_path="/home/dw/ws_job_msislab/Golf_Project/runs_yolo/20251124/20251107_data_yolov8s_img640_SGD_cls1.0_box7.5_dfl1.5_rectFalse_seed_57_20251123/weights/best_fp16_bs1_640px_static.engine",
        data_yaml="/home/dw/ws_job_msislab/Golf_Project/data/for_test/test_20251113/data.yaml",
        test_dir="/home/dw/ws_job_msislab/Golf_Project/data/for_test/test_20251113/images/test",
        gt_dir="/home/dw/ws_job_msislab/Golf_Project/data/for_test/test_20251113/labels/test",

        img_size=640,
        device=0,
        pred_conf=0.001,
        match_iou=0.5,

        # ✅ 여기만 방금 “질문으로 받은 값”을 넣음
        draw_save=draw_save,
        draw_only_divot=draw_only_divot,
        draw_score_min=draw_score_min,
        draw_dir_name="viz",

        draw_divot_roc=True,

        eval_conf_print_map=0.5,
        eval_conf_print_auroc=0.5
    )

    print("📄 Summary:", result["summary_path"])

