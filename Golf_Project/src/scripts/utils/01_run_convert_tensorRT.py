# run_tensorrt_convert.py
# -*- coding: utf-8 -*-

from utils.convert_tensorRT import convert_all_to_trt

if __name__ == "__main__":
    convert_all_to_trt(
        runs_dir="/home/dw/ws_job_msislab/Golf_Project/runs_yolo/20251125_ghulam",
        img_size=640,
        batch=1,
        fp16=True,
        dynamic=False,
        workspace=4096,
    )
