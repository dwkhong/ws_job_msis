#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_augment.py

from pathlib import Path
import sys

# 현재 파일이 .../src/scripts/dataset/ 아래면 parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.data_augmentation_class import augment_dataset

# =========================
# 🔥 실행자가 입력하는 설정
# =========================
SEED = 33

TARGET = {
    0: 0,
    1: 10,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
}

BASES = [
    Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check/check_1"),
    # Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check/check_2"),
]

# ✅ 증강 파라미터 (실제로 영향 있는 것만)
BG_AUG_MULTIPLIER = 3
MAX_PER_IMAGE_HARD = 1
RECENT_COOLDOWN = 5

# =========================
# 🚀 실행
# =========================
if __name__ == "__main__":
    for base in BASES:
        augment_dataset(
            base_dir=base,
            TARGET=TARGET,
            SEED=SEED,
            BG_AUG_MULTIPLIER=BG_AUG_MULTIPLIER,
            MAX_PER_IMAGE_HARD=MAX_PER_IMAGE_HARD,
            RECENT_COOLDOWN=RECENT_COOLDOWN,
        )
        print(f"[DONE] {base.name} augmentation completed.")

