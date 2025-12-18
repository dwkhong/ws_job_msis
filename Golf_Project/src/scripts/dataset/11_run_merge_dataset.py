# run_merge.py
from pathlib import Path
from dataset.merge_dataset import (
    ensure_dirs, merge_and_move,
    read_and_unify_class_names, write_classes_txt, write_data_yaml,
    merge_all_aug_logs, quick_summary
)

# ================================================================
# 🔧 실행자가 수정하는 설정
# ================================================================
MERGE_ROOT = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")
SOURCES = [
    MERGE_ROOT / "20250721_good_data",
    MERGE_ROOT / "20250725_good_data",
    MERGE_ROOT / "20250904_good_data",
    MERGE_ROOT / "20250929_good_data",
    MERGE_ROOT / "20250930_good_data",
]

# ================================================================
# 🚀 실행
# ================================================================
if __name__ == "__main__":
    ensure_dirs(MERGE_ROOT)

    for src in SOURCES:
        if src.exists():
            print(f"→ Moving from: {src}")
            merge_and_move(MERGE_ROOT, src)

    names, status = read_and_unify_class_names(MERGE_ROOT)
    write_classes_txt(MERGE_ROOT, names)
    write_data_yaml(MERGE_ROOT, names)

    merge_all_aug_logs(MERGE_ROOT)
    quick_summary(MERGE_ROOT)

    print("✅ Done. All train/val merged, logs organized, classes unified.")
