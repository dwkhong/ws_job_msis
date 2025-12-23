# scripts/run_check_cluster_split.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
# 현재 파일: .../src/scripts/dataset/01_run_copy_dataset.py
# parents[2] => .../src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.check_cluster_split_1 import check_folder

# ================================================================
# 설정 (여기만 수정하면 됨)
# ================================================================
from scripts.dataset.settings import BASE_DIR, SRC_LIST   # ✅ 둘 다 가져옴

SUBFOLDERS = [p.name for p in SRC_LIST]

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
