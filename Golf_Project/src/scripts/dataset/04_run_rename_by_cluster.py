#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.rename_by_cluster import rename_by_cluster
from scripts.dataset.settings import BASE_DIR  # ✅ 여기서 가져옴

# ================================================================
# 사용자 설정
# ================================================================
CSV_DIR = BASE_DIR 
APPLY_CHANGES = True   # 실제 이름 변경 수행

# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    rename_by_cluster(CSV_DIR, apply=APPLY_CHANGES)
