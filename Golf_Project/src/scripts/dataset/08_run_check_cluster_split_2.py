#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dataset.check_cluster_split_2 import check_cluster_split

# ================================================================
# 🧩 기본 설정
# ================================================================
BASE_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")

SUBFOLDERS = [
    "20250721_good_data",
    "20250725_good_data",
    "20250904_good_data",
    "20250929_good_data",
    "20250930_good_data",
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
