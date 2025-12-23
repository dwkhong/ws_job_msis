# run_check_merge.py
from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.report_final import check_split

BASE = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check")

if __name__ == "__main__":
    for split in ["train", "val"]:
        check_split(BASE, split)
