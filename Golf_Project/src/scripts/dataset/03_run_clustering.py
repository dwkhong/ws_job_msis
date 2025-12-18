#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dataset.clustering import run_clustering

# ================================================================
# 데이터셋 위치
# ================================================================
BASE_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")

SUBFOLDERS = [
    "20250721_good_data",
    "20250725_good_data",
    "20250904_good_data",
    "20250929_good_data",
    "20250930_good_data",
]

FOLDERS = [BASE_DIR / name for name in SUBFOLDERS]


# ================================================================
# 기본 설정값
# ================================================================
DEFAULT_CFG = {
    "GAMMA": 0.60,
    "CHANGE_THRESH": 0.20,
    "GAUSS_BLUR": (3, 3),
    "ZNCC_SIZE": 256,
    "MAX_CLUSTER_LEN": 5,
    "DOWNSCALE": False,
    "PREPROC_MAX_SIDE": 0
}

# ================================================================
# 폴더별 개별 파라미터
# ================================================================
FOLDER_CFG = {
    "20250721_good_data": {"GAMMA": 0.60, "CHANGE_THRESH": 0.32, "MAX_CLUSTER_LEN": 5},
    "20250725_good_data": {"GAMMA": 0.60, "CHANGE_THRESH": 0.35, "MAX_CLUSTER_LEN": 5},
    "20250904_good_data": {"GAMMA": 0.55, "CHANGE_THRESH": 0.25, "MAX_CLUSTER_LEN": 25},
    "20250929_good_data": {"GAMMA": 0.55, "CHANGE_THRESH": 0.25, "MAX_CLUSTER_LEN": 7},
    "20250930_good_data": {"GAMMA": 0.60, "CHANGE_THRESH": 0.28, "MAX_CLUSTER_LEN": 15},
}

# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    run_clustering(FOLDERS, FOLDER_CFG, DEFAULT_CFG)
