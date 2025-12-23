import sys
from pathlib import Path

# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.copy_dataset import copy_folders

from scripts.dataset.settings import SRC_LIST, DST_ROOT   # ✅ 여기만 수정

def main():
    # Path 객체 -> copy_folders는 str도 받으니까 str로 변환해서 전달
    copy_folders([str(p) for p in SRC_LIST], str(DST_ROOT))

if __name__ == "__main__":
    main()