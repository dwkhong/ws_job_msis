#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# 현재 파일이 .../src/scripts/dataset/ 아래면 parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.data_augmentation_multi import augment_dataset_by_multiplier
from scripts.dataset.settings import BASE_DIR, SRC_LIST

# =========================
# 🔥 실행자가 입력하는 설정
# =========================
SEED = 33
MULTIPLIER = 3.0   # ✅ 전체 N배
NUM_CLASSES = 8

BASES = [BASE_DIR / p.name for p in SRC_LIST]

# ✅ 아래 3개만 실제로 영향 있음
BG_AUG_MULTIPLIER = 3
MAX_PER_IMAGE_HARD = 1
RECENT_COOLDOWN = 5

# =========================
# 🚀 실행
# =========================
if __name__ == "__main__":
    for base in BASES:
        augment_dataset_by_multiplier(
            base_dir=base,
            num_classes=NUM_CLASSES,
            multiplier=MULTIPLIER,
            SEED=SEED,
            BG_AUG_MULTIPLIER=BG_AUG_MULTIPLIER,
            MAX_PER_IMAGE_HARD=MAX_PER_IMAGE_HARD,
            RECENT_COOLDOWN=RECENT_COOLDOWN,
        )
        print(f"[DONE] {base.name} augmentation completed.")

