# scripts/run_autolabel.py
# -*- coding: utf-8 -*-

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
