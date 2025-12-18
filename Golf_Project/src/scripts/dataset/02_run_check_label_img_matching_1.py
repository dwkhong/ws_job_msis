# src/scripts/run_check_matching.py

from pathlib import Path
from dataset.check_label_img_matching_1 import check_label_matching

BASE_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")

jpg_only, txt_only = check_label_matching(BASE_DIR)