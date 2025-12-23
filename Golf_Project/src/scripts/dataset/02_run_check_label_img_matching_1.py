# src/scripts/run_check_matching.py

from pathlib import Path
import sys

# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.check_label_img_matching_1 import check_label_matching

BASE_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check")

jpg_only, txt_only = check_label_matching(BASE_DIR)