# run_augment.py
from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
    Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check/check_1"),
    #Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check/check_2")
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
