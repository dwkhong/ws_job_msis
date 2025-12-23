#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.check_cluster_split_2 import check_cluster_split

# ================================================================
# 🧩 기본 설정
# ================================================================
BASE_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check")

SUBFOLDERS = [
    "check_1",
    "check_2",
]

FOLDERS = [BASE_DIR / name for name in SUBFOLDERS if (BASE_DIR / name).exists()]
missing = [name for name in SUBFOLDERS if not (BASE_DIR / name).exists()]

print(f"[OK] 점검 대상 폴더 {len(FOLDERS)}개:")
for f in FOLDERS:
    print("  └", f)
if missing:
    print(f"[WARN] 존재하지 않는 폴더: {missing}")

# ================================================================
# 🚀 실행
# ================================================================
if __name__ == "__main__":
    for folder in FOLDERS:
        check_cluster_split(folder)
