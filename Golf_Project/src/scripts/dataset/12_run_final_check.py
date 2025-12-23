# run_check_merge.py
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.final_check import check_split
from scripts.dataset.settings import BASE_DIR, SRC_LIST


if __name__ == "__main__":
    for split in ["train", "val"]:
        check_split(BASE_DIR, split)
