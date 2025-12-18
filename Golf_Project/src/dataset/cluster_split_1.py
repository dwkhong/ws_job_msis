#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클러스터 기반 stratified train/val split 모듈.

- 파일명은 반드시 *_cluster_<id>.jpg 형태여야 함
- cluster 단위로 split
- BALANCE_CLASS_IDS 기준으로 클래스 균형을 맞춰서 split 수행
- materialize(move/copy/symlink) 및 리포트 생성 기능 포함
"""

from pathlib import Path
import re
import random
import shutil

# ================================================================
# 공통 설정 (외부에서 override 가능)
# ================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

CLUSTER_RE = re.compile(r"^(?P<stem>.+)_cluster_(?P<cid>\d+)(?:__\d+)?$", re.IGNORECASE)

# 가중치 (이미지 개수 / 클래스 균형 / 배경 비율 중요도)
IMG_W = 1.0
CLS_W = 1.0
BG_W  = 1.0


# ================================================================
# 유틸 함수
# ================================================================
def find_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix in IMG_EXTS]


def parse_cluster_id(stem: str):
    m = CLUSTER_RE.match(stem)
    return (m.group("stem"), int(m.group("cid"))) if m else (None, None)


def read_label_lines(txt_path: Path):
    if not txt_path.exists():
        return []
    try:
        with txt_path.open("r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except:
        return []


def count_classes_from_txt_lines(lines, class_ids):
    counts = {k: 0 for k in class_ids}
    for s in lines:
        parts = s.split()
        try:
            cid = int(parts[0])
        except:
            continue
        if cid in counts:
            counts[cid] += 1
    return counts


# ================================================================
# 클러스터 통계 생성
# ================================================================
def build_cluster_stats(folder: Path, balance_class_ids, all_class_ids):
    clusters = {}

    for img in find_images(folder):
        _, cid = parse_cluster_id(img.stem)
        if cid is None:
            continue

        txt = img.with_suffix(".txt")
        lines = read_label_lines(txt)

        counts_all = count_classes_from_txt_lines(lines, all_class_ids)
        counts_bal = {k: counts_all.get(k, 0) for k in balance_class_ids}
        is_bg = (len(lines) == 0)

        if cid not in clusters:
            clusters[cid] = {
                "images": [],
                "class_counts_bal": {k: 0 for k in balance_class_ids},
                "class_counts_all": {k: 0 for k in all_class_ids},
                "bg_count": 0,
            }

        clusters[cid]["images"].append(img)

        for k in balance_class_ids:
            clusters[cid]["class_counts_bal"][k] += counts_bal[k]

        for k in all_class_ids:
            clusters[cid]["class_counts_all"][k] += counts_all[k]

        clusters[cid]["bg_count"] += 1 if is_bg else 0

    return clusters


# ================================================================
# Stratified Cluster Split
# ================================================================
def stratified_split(clusters: dict, ratios: dict, balance_ids: list, seed=42):
    random.seed(seed)

    cluster_items = []
    total_imgs, total_cls, total_bg = 0, {k: 0 for k in balance_ids}, 0

    for cid, info in clusters.items():
        sz = len(info["images"])
        total_imgs += sz

        vec = {"imgs": sz, "bg": info.get("bg_count", 0)}
        total_bg += vec["bg"]

        for k in balance_ids:
            vec[k] = info["class_counts_bal"][k]
            total_cls[k] += vec[k]

        cluster_items.append((cid, vec))

    if total_imgs == 0:
        return {}

    targets_imgs = {s: ratios[s] * total_imgs for s in ratios}
    targets_bg   = {s: ratios[s] * total_bg   for s in ratios}
    targets_class = {
        k: {s: ratios[s] * total_cls[k] for s in ratios}
        for k in balance_ids
    }

    cluster_items.sort(key=lambda x: (-x[1]["imgs"], x[0]))
    assigned_imgs = {s: 0.0 for s in ratios}
    assigned_bg   = {s: 0.0 for s in ratios}
    assigned_class = {k: {s: 0.0 for s in ratios} for k in balance_ids}

    mapping = {}

    def score(ai_imgs, ai_bg, ai_cls):
        img_resid = sum((ai_imgs[s] - targets_imgs[s])**2 for s in ratios)
        bg_resid  = sum((ai_bg[s]   - targets_bg[s])**2   for s in ratios)
        cls_resid = sum(
            (ai_cls[k][s] - targets_class[k][s])**2
            for k in balance_ids
            for s in ratios
        )
        return IMG_W*img_resid + BG_W*bg_resid + CLS_W*cls_resid

    def score_after(split, vec):
        ai_imgs = {s: assigned_imgs[s] + (vec["imgs"] if s == split else 0) for s in ratios}
        ai_bg   = {s: assigned_bg[s]   + (vec["bg"]   if s == split else 0) for s in ratios}
        ai_cls  = {
            k: {s: assigned_class[k][s] + (vec[k] if s == split else 0) for s in ratios}
            for k in balance_ids
        }
        return score(ai_imgs, ai_bg, ai_cls)

    for cid, vec in cluster_items:
        best_split, best_sc = None, None

        for s in ratios:
            sc = score_after(s, vec)
            if best_sc is None or sc < best_sc:
                best_sc, best_split = sc, s

        mapping[cid] = best_split

        assigned_imgs[best_split] += vec["imgs"]
        assigned_bg[best_split]   += vec["bg"]
        for k in balance_ids:
            assigned_class[k][best_split] += vec[k]

    return mapping


# ================================================================
# Materialize split 결과
# ================================================================
def ensure_clean_split_dirs(base: Path, clean_previous=True):
    images_root = base / "images"
    labels_root = base / "labels"

    if clean_previous:
        shutil.rmtree(images_root, ignore_errors=True)
        shutil.rmtree(labels_root, ignore_errors=True)

    for split in ("train", "val"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode="move"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copy2(str(src), str(dst))
    elif mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except FileExistsError:
            dst.unlink()
            dst.symlink_to(src.resolve())


def materialize_split(folder: Path, clusters: dict, mapping: dict,
                      split_names, all_class_ids,
                      report_dir: Path, mode="move",
                      create_empty_label=False, clean_previous=True):

    ensure_clean_split_dirs(folder, clean_previous)

    split_img_counts = {0: 0, 1: 0}
    split_bg_counts  = {0: 0, 1: 0}
    split_class_counts = {
        s: {k: 0 for k in all_class_ids}
        for s in (0,1)
    }

    for cid, info in clusters.items():
        split = mapping[cid]
        img_dir = folder / "images" / split_names[split]
        lbl_dir = folder / "labels" / split_names[split]

        for k in all_class_ids:
            split_class_counts[split][k] += info["class_counts_all"][k]
        split_bg_counts[split] += info.get("bg_count", 0)

        for img in info["images"]:
            split_img_counts[split] += 1

            place_file(img, img_dir / img.name, mode=mode)

            txt_src = img.with_suffix(".txt")
            txt_dst = lbl_dir / txt_src.name

            if txt_src.exists():
                place_file(txt_src, txt_dst, mode=mode)
            elif create_empty_label:
                txt_dst.touch(exist_ok=True)

    # 리포트 저장
    report_dir.mkdir(parents=True, exist_ok=True)
    out_txt = report_dir / f"split_report_{folder.name}.txt"

    lines = []
    total = sum(split_img_counts.values())

    lines.append(f"[OK] {folder.name} split completed (total {total} images)")

    for s in (0,1):
        cnt = split_img_counts[s]
        pct = cnt / total * 100 if total else 0
        lines.append(f"  {split_names[s]}: {cnt} imgs ({pct:.1f}%)")

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_txt
