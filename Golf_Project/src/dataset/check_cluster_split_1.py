# src/check/check_cluster_split_1.py
from pathlib import Path
import re
from collections import defaultdict

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"
}

CLUSTER_RE = re.compile(r"_cluster_(\d+)", re.IGNORECASE)


def check_folder(folder: Path):
    """
    하나의 merge_data 하위 폴더에서
    동일 cluster_id가 train/val/test 여러 split에 걸쳐 존재하는지 검사.
    
    Returns:
        dict: { cluster_id: ['train','val'] } 와 같이 섞여 있는 클러스터 목록
    """
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
                cluster_id = m.group(1)
                cluster_splits[cluster_id].add(split)

    # 두 개 이상의 split에 등장한 클러스터만 반환
    bad = {cid: sorted(splits)
           for cid, splits in cluster_splits.items()
           if len(splits) > 1}

    return bad
