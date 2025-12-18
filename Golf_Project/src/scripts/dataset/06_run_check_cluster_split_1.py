# scripts/run_check_cluster_split.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dataset.check_cluster_split_1 import check_folder

# ================================================================
# 설정 (여기만 수정하면 됨)
# ================================================================
BASE_DIR = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251107_merge_data")

SUBFOLDERS = [
    "20250721_good_data",
    "20250725_good_data",
    "20250904_good_data",
    "20250929_good_data",
    "20250930_good_data",
]

# 자동 조합
FOLDERS = [BASE_DIR / name for name in SUBFOLDERS if (BASE_DIR / name).exists()]
missing = [name for name in SUBFOLDERS if not (BASE_DIR / name).exists()]

print(f"[OK] 점검할 폴더 {len(FOLDERS)}개:")
for f in FOLDERS:
    print("  └", f)

if missing:
    print(f"[WARN] 존재하지 않는 폴더: {missing}")


# ================================================================
# 실행
# ================================================================
def main():
    for folder in FOLDERS:
        print(f"\n=== Checking {folder.name} ===")
        bad_clusters = check_folder(folder)

        if not bad_clusters:
            print("✅ 모든 클러스터가 단일 split에만 존재 (문제 없음)")
        else:
            print(f"❌ {len(bad_clusters)}개 클러스터가 train/val/test에 섞여 있음")
            for cid, splits in bad_clusters.items():
                print(f"  cluster {cid}: {', '.join(splits)}")


if __name__ == "__main__":
    main()
