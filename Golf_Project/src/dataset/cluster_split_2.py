#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클러스터 무시 / 순수 class-based 재분할 모듈
"""

from pathlib import Path
import shutil
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}


# =========================================================
# 기본 유틸
# =========================================================
def split_dirs(base: Path):
    return {
        0: (base / "images" / "train", base / "labels" / "train"),
        1: (base / "images" / "val",   base / "labels" / "val"),
    }


def iter_split_images(base: Path):
    for sid, (img_dir, _) in split_dirs(base).items():
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.is_file() and p.suffix in IMG_EXTS:
                yield sid, p


def read_label_lines(txt: Path):
    if not txt.exists():
        return []
    try:
        return [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except:
        return []


def get_class_info(txt: Path, target_classes, exclude_012):
    """YOLO txt에서 타겟 클래스 카운트 및 배제 여부 판단"""
    lines = read_label_lines(txt)
    is_bg = (len(lines) == 0)

    counts = {k: 0 for k in target_classes}
    has_012 = False

    for s in lines:
        parts = s.split()
        try:
            cid = int(parts[0])
        except:
            continue

        if cid in counts:
            counts[cid] += 1
        if cid in (0, 1, 2):
            has_012 = True

    if exclude_012 and has_012:
        return None

    has_target = sum(counts.values()) > 0
    return counts, is_bg, has_012, has_target


def place_file(src: Path, dst: Path, mode="move"):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "move":
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))

    elif mode == "copy":
        shutil.copy2(str(src), str(dst))

    elif mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except FileExistsError:
            dst.unlink(missing_ok=True)
            dst.symlink_to(src.resolve())

    else:
        raise ValueError("Invalid materialize mode")


# =========================================================
# 재분할 메인 함수
# =========================================================
def split_by_class_balance(
    folder: Path,
    target_classes,
    ratios,                    # {0: train_ratio, 1: val_ratio}
    seed,
    exclude_if_has_012=False,
    handle_bg=True,
    mode="move",
    dry_run=False,
):
    """
    특정 클래스 기반의 train/val 재분할 (클러스터 무시)
    """

    random.seed(seed)

    items = []
    class_totals = {k: 0 for k in target_classes}

    # -----------------------
    # 1) 이미지 수집
    # -----------------------
    for sid, img in iter_split_images(folder):
        lbl = folder / "labels" / ("train" if sid == 0 else "val") / f"{img.stem}.txt"
        info = get_class_info(lbl, target_classes, exclude_if_has_012)

        if info is None:
            continue

        counts, is_bg, has_012, has_target = info

        if not has_target and not (handle_bg and is_bg):
            continue

        vec = {"imgs": 1, "bg": 1 if (handle_bg and is_bg) else 0}
        vec.update(counts)

        items.append((img, lbl, sid, vec))

        for k in target_classes:
            class_totals[k] += counts[k]

    if not items:
        print(f"[INFO] {folder.name}: No valid items found.")
        return None

    # -----------------------
    # 2) 목표 개수 계산
    # -----------------------
    target_objs = {
        split: {k: class_totals[k] * ratios[split] for k in target_classes}
        for split in ratios
    }

    obj_counts = {s: {k: 0 for k in target_classes} for s in ratios}

    random.shuffle(items)

    assignment = []

    # -----------------------
    # 3) 부족 split 선택
    # -----------------------
    for img, lbl, from_split, vec in items:
        scores = {}

        for split in ratios:
            diff_sum = 0
            for k in target_classes:
                diff = obj_counts[split][k] - target_objs[split][k]
                diff_sum += diff / max(class_totals[k], 1)
            scores[split] = diff_sum

        best_split = min(scores, key=scores.get)
        assignment.append((img, lbl, from_split, best_split, vec))

        for k in target_classes:
            obj_counts[best_split][k] += vec[k]

    # -----------------------
    # 4) 반영
    # -----------------------
    if not dry_run:
        for img, lbl, from_split, to_split, vec in assignment:
            if from_split == to_split:
                continue

            img_dst = folder / "images" / ("train" if to_split == 0 else "val") / img.name
            lbl_dst = folder / "labels" / ("train" if to_split == 0 else "val") / f"{img.stem}.txt"

            place_file(img, img_dst, mode)
            if lbl.exists():
                place_file(lbl, lbl_dst, mode)

    return assignment, class_totals, obj_counts
