#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.cluster_split_2 import split_by_class_balance
from scripts.dataset.settings import BASE_DIR, SRC_LIST
# ======================================================================
# 🔧 사용자 설정
# ======================================================================

FOLDERS = [BASE_DIR / p.name for p in SRC_LIST]

# ---- 원하는 클래스 입력 (여기서 수정하면 됨) ----
TARGET_CLASSES = [4, 5, 6, 7]

# ---- seed도 여기서 설정 ----
SEED = 33

# ---- train/val 비율 ----
RATIOS = {0: 0.86, 1: 0.14}

EXCLUDE_012 = False
HANDLE_BG = True

MATERIALIZE_MODE = "move"
DRY_RUN = False
# ======================================================================


if __name__ == "__main__":

    for folder in FOLDERS:
        print(f"\n===== Processing {folder.name} =====")
        result = split_by_class_balance(
            folder,
            target_classes=TARGET_CLASSES,
            ratios=RATIOS,
            seed=SEED,
            exclude_if_has_012=EXCLUDE_012,
            handle_bg=HANDLE_BG,
            mode=MATERIALIZE_MODE,
            dry_run=DRY_RUN,
        )

        if result is None:
            print(f"[INFO] {folder.name}: skip")
            continue

        assignment, totals, after = result

        print("\n--- Final Results ---")
        for k in TARGET_CLASSES:
            total = totals[k]
            t_train = after[0][k]
            t_val = after[1][k]
            print(f"class {k} → total={total}, train={t_train}, val={t_val}")
