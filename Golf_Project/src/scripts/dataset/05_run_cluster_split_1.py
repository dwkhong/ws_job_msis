#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
클러스터 기반 stratified train/val split 실행 스크립트
"""

from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.cluster_split_1 import (
    build_cluster_stats,
    stratified_split,
    materialize_split
)

# ================================================================
# 사용자 설정
# ================================================================
BASE_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check")

SUBFOLDERS = [
    "check_1",
    "check_2",
]

FOLDERS = [BASE_DIR / name for name in SUBFOLDERS]

# Split 설정
RATIOS = {0: 0.86, 1: 0.14}
SPLIT_NAME = {0: "train", 1: "val"}

BALANCE_CLASS_IDS = [0, 1, 2]
ALL_CLASS_IDS = list(range(8))

SEED = 33
MODE = "move"          # move | copy | symlink
REPORT_DIR = BASE_DIR / "split_reports"

# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    for folder in FOLDERS:
        if not folder.exists():
            print(f"[WARN] Skip missing folder: {folder}")
            continue

        print(f"\n=== Processing folder: {folder.name} ===")

        clusters = build_cluster_stats(folder, BALANCE_CLASS_IDS, ALL_CLASS_IDS)

        if not clusters:
            print(f"[INFO] No valid clusters in folder: {folder.name}")
            continue

        mapping = stratified_split(clusters, RATIOS, BALANCE_CLASS_IDS, seed=SEED)

        report = materialize_split(
            folder, clusters, mapping,
            split_names=SPLIT_NAME,
            all_class_ids=ALL_CLASS_IDS,
            report_dir=REPORT_DIR,
            mode=MODE
        )

        print(f"[DONE] Report saved → {report}")
