#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_merge.py

from pathlib import Path
import sys

# 현재 파일이 .../src/scripts/dataset/ 아래면 parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataset.settings import BASE_DIR, SRC_LIST
from dataset.merge_dataset import (
    ensure_dirs, merge_and_move,
    read_and_unify_class_names, write_classes_txt, write_data_yaml,
    merge_all_aug_logs, quick_summary
)

# ================================================================
# 🔧 settings 기반
# ================================================================
MERGE_ROOT = BASE_DIR
SOURCES = [MERGE_ROOT / p.name for p in SRC_LIST]

# ================================================================
# 🚀 실행
# ================================================================
if __name__ == "__main__":
    ensure_dirs(MERGE_ROOT)

    for src in SOURCES:
        if src.exists():
            print(f"→ Moving from: {src}")
            merge_and_move(MERGE_ROOT, src)
        else:
            print(f"[SKIP] not found: {src}")

    names, status = read_and_unify_class_names(MERGE_ROOT)
    write_classes_txt(MERGE_ROOT, names)
    write_data_yaml(MERGE_ROOT, names)

    merge_all_aug_logs(MERGE_ROOT)
    quick_summary(MERGE_ROOT)

    print("✅ Done. All train/val merged, logs organized, classes unified.")

