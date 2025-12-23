#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.rename_by_cluster import rename_by_cluster

# ================================================================
# 사용자 설정
# ================================================================
CSV_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check")
APPLY_CHANGES = True   # 실제 이름 변경 수행

# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    rename_by_cluster(CSV_DIR, apply=APPLY_CHANGES)
