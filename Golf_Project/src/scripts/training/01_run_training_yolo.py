# run_yolov8_train.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.training_yolo import run_training

# ===============================
# 사용자가 수정하는 부분 (단순함)
# ===============================

DATA_YAML  = "/data/DOCKER_DIRS/eddie/20251107_merge_data/data.yaml"
MODEL_NAME = "yolov8s.pt"
PROJECT_DIR = "/home/eddie/result"

IMG_SIZE = 640
EPOCHS   = 300
RECT_MODE = False

DATA_TAG = "20251107"

SEEDS    = [25, 33, 57]
CLS_LIST = [0.5]

OPTIMIZERS = [
    {"name": "SGD",   "lr0": 0.01,  "wd": 0.0005, "momentum": 0.937},
    {"name": "AdamW", "lr0": 0.001, "wd": 0.01,   "momentum": 0.937},
]

WEIGHT_SUM4 = {"precision":0.2, "recall":0.3, "map50":0.25, "map5095":0.25}

# ===============================
# 실행
# ===============================
for opt in OPTIMIZERS:
    for cls_w in CLS_LIST:
        for seed in SEEDS:
            run_training(
                data_yaml=DATA_YAML,
                model_name=MODEL_NAME,
                project_dir=PROJECT_DIR,
                img_size=IMG_SIZE,
                epochs=EPOCHS,
                box_w=7.5,
                cls_w=cls_w,
                dfl_w=1.5,
                seed=seed,
                optimizer_name=opt["name"],
                lr0=opt["lr0"],
                weight_decay=opt["wd"],
                momentum=opt["momentum"],
                rect_mode=RECT_MODE,
                save_period=1,
                data_tag=DATA_TAG,
                sum4_weights=WEIGHT_SUM4,
            )