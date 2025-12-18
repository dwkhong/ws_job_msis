# run_check_merge.py
from pathlib import Path
from dataset.report_final import check_split

BASE = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")

if __name__ == "__main__":
    for split in ["train", "val"]:
        check_split(BASE, split)
