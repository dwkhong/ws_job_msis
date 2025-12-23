# run_merge.py
from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.merge_dataset import (
    ensure_dirs, merge_and_move,
    read_and_unify_class_names, write_classes_txt, write_data_yaml,
    merge_all_aug_logs, quick_summary
)

# ================================================================
# 🔧 실행자가 수정하는 설정
# ================================================================
MERGE_ROOT = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251223_check")
SOURCES = [
    MERGE_ROOT / "check_1",
    MERGE_ROOT / "check_2",

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
