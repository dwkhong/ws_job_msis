#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dataset.rename_by_cluster import rename_by_cluster

# ================================================================
# 사용자 설정
# ================================================================
CSV_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")
APPLY_CHANGES = True   # 실제 이름 변경 수행

# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    rename_by_cluster(CSV_DIR, apply=APPLY_CHANGES)
