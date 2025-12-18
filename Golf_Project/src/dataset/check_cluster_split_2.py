#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

CLUSTER_RE = re.compile(r"_cluster_(\d+)", re.IGNORECASE)


def check_cluster_split(folder: Path):
    """
    같은 cluster_id가 train/val/test 여러 split에 섞여 있는지 검사하는 함수.
    """
    print(f"\n=== Checking {folder.name} ===")

    cluster_splits = defaultdict(set)

    for split in ("train", "val", "test"):
        img_dir = folder / "images" / split
        if not img_dir.exists():
            continue

        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix in IMG_EXTS:
                m = CLUSTER_RE.search(p.stem)
                if not m:
                    continue
                cluster_splits[m.group(1)].add(split)

    # 교차 split 탐지
    bad = {cid: sorted(splits) for cid, splits in cluster_splits.items() if len(splits) > 1}

    if not bad:
        print("✅ 모든 클러스터가 한 split에만 존재 (문제 없음)")
        return True, {}

    print(f"❌ {len(bad)}개의 클러스터가 여러 split에 섞여 있음")
    for cid, splits in bad.items():
        print(f"  cluster {cid}: {', '.join(splits)}")

    return False, bad
