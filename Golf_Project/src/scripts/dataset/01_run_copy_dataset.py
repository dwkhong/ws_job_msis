import sys
from pathlib import Path

# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.copy_dataset import copy_folders

SRC_LIST = [
    "/home/dw/ws_job_msislab/Golf_Project/data/check_1",
    "/home/dw/ws_job_msislab/Golf_Project/data/check_2",
]

DST = "/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check"

copy_folders(SRC_LIST, DST)