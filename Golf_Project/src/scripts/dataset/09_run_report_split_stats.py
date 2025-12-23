# scripts/run_report_split_stats.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.report_split_stats import generate_full_report
from scripts.dataset.settings import BASE_DIR, SRC_LIST

# === 설정 부분 ===
ROOT = BASE_DIR
TARGETS = [ROOT / p.name for p in SRC_LIST]

OUTPUT_FILE = ROOT / "split_class_report_for_augment.txt"

# === 실행 ===
if __name__ == "__main__":
    summary = generate_full_report(ROOT, TARGETS, OUTPUT_FILE)

    print(f"✅ 전체 결과를 {OUTPUT_FILE} 에 저장 완료!\n")
    print("[TOP SUMMARY]")
    for line in summary:
        print(line)

    print("\n[Per-dataset quick image counts]")
    from dataset.report_split_stats import list_images

    for tgt in TARGETS:
        for split in ["train", "val"]:
            img_count = len(list_images(tgt / "images" / split))
            print(f"{tgt.name} [{split}] images: {img_count}")
