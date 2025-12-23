# scripts/run_evaluation.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.evaluation_yolo import run_evaluation

if __name__ == "__main__":
    result = run_evaluation(
        engine_path="yolo_model/weights/best_fp16_bs1_640px_static.engine",
        data_yaml="test_data/data.yaml",
        test_dir="test_data/images/test",
        gt_dir="test_data/labels/test",

        img_size=640,
        device=0,
        pred_conf=0.001,
        match_iou=0.5,

        draw_save=False,
        draw_only_divot=False,
        draw_score_min=0.001,
        draw_dir_name="viz",

        draw_divot_roc=True,

        eval_conf_print_map=0.5,
        eval_conf_print_auroc=0.5
    )

    print("📄 Summary:", result["summary_path"])
