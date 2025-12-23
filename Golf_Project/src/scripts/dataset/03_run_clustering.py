#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.clustering import run_clustering
from scripts.dataset.settings import BASE_DIR, SRC_LIST   # ✅ 둘 다 가져옴

# ================================================================
# 데이터셋 위치
#   SUBFOLDERS는 SRC_LIST에서 자동 생성
# ================================================================
SUBFOLDERS = [p.name for p in SRC_LIST]
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
    "PREPROC_MAX_SIDE": 0,
}

# ================================================================
# 폴더별 개별 파라미터
#   키는 폴더명(p.name) 기준으로 매칭됨
# ================================================================
FOLDER_CFG = {
    "ex": {"GAMMA": 0.60, "CHANGE_THRESH": 0.32, "MAX_CLUSTER_LEN": 5},
}

# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    run_clustering(FOLDERS, FOLDER_CFG, DEFAULT_CFG)


