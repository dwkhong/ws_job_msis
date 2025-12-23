# run_tensorrt_convert.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.convert_tensorRT import convert_all_to_trt

if __name__ == "__main__":
    convert_all_to_trt(
        runs_dir="/home/dw/ws_job_msislab/Golf_Project/runs_yolo/20251125",
        img_size=640,
        batch=1,
        fp16=True,
        dynamic=False,
        workspace=4096,
    )
