# scripts/run_autolabel.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from labeling.auto_labeling_engine import auto_label

if __name__ == "__main__":

    ENGINE = "/home/dw/.../best_fp16_bs1_640px_static.engine"
    IMG_DIR = "/home/dw/ws_job_msislab/Golf_Project/data/leaves_background"

    min_conf = {0:0.40, 1:0.40, 2:0.50, 4:0.50, 5:0.50, 6:0.50, 7:0.50}

    auto_label(
        engine_path=ENGINE,
        imgsz=640,
        img_dir=IMG_DIR,
        min_conf=min_conf,
        confusion_iou=0.5,
        delta_conf=0.01,
        base_conf=0.01,
        require_minconf_for_confusion=False,
        require_both_minconf=False,
        progress_every=50
    )
