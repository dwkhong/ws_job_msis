# run_augment.py
from pathlib import Path
from dataset.data_augmentation import augment_dataset

# =========================
# 🔥 실행자가 입력하는 설정
# =========================
SEED = 33

TARGET = {
    0: 422,
    1: 717,
    2: 38,
    3: 0,
    4: 27,
    5: 78,
    6: 18,
    7: 54,
}

BASES = [
    Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data/20250725_good_data")
]

# 증강 파라미터도 실행파일에서 조절 가능
BG_AUG_MULTIPLIER = 3
MAX_USES_BASE = 1
MAX_USES_BOOST_PER_CLASS = {0:2,1:2,2:3,3:0,4:3,5:3,6:3,7:3}
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
            MAX_USES_BASE=MAX_USES_BASE,
            MAX_USES_BOOST=MAX_USES_BOOST_PER_CLASS,
            MAX_PER_IMAGE_HARD=MAX_PER_IMAGE_HARD,
            RECENT_COOLDOWN=RECENT_COOLDOWN,
        )
        print(f"[DONE] {base.name} augmentation completed.")
