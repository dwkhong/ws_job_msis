# scripts/run_report_split_stats.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dataset.report_split_stats import generate_full_report

# === 설정 부분 ===
ROOT = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")
TARGETS = [
    ROOT / "20250721_good_data",
    ROOT / "20250725_good_data",
    ROOT / "20250904_good_data",
    ROOT / "20250929_good_data",
    ROOT / "20250930_good_data",
]

OUTPUT_FILE = ROOT / "split_class_report_after_4567.txt"

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
