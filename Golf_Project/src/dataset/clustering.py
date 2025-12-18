#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 불변 특징 기반 클러스터링
LBP + ZNCC 기반 → 조명/밝기 변화에 둔감
"""

from pathlib import Path
import cv2
import numpy as np
import csv, re
from typing import Dict, Tuple, Optional, List


# ================================================================
# 기본 설정
# ================================================================
IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"
}

LBP_HIST_BINS = 256
NATURAL_SORT = True


# ================================================================
# 유틸 함수
# ================================================================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: Path, max_images: int = None) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files[:max_images] if max_images else files


def resize_longest(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)


def load_bgr(path: Path, downscale: bool, max_side: int) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if downscale and max_side > 0:
        img = resize_longest(img, max_side)
    return img


def gray_of(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ================================================================
# LBP / ZNCC
# ================================================================
def lbp_image(gray: np.ndarray) -> np.ndarray:
    g = gray
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)

    nbrs = [
        (0, 0), (0, 1), (0, 2),
        (1, 2), (2, 2), (2, 1),
        (2, 0), (1, 0),
    ]

    for bit, (dy, dx) in enumerate(nbrs):
        n = g[dy:dy + c.shape[0], dx:dx + c.shape[1]]
        code |= ((n >= c) << (7 - bit)).astype(np.uint8)

    return code


def lbp_hist(gray: np.ndarray, blur_kernel: Tuple[int, int]) -> np.ndarray:
    if blur_kernel and blur_kernel[0] > 0:
        gray = cv2.GaussianBlur(gray, blur_kernel, 0)

    lbp = lbp_image(gray)
    hist = cv2.calcHist([lbp], [0], None, [LBP_HIST_BINS], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def d_bhat(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def zncc(gray1: np.ndarray, gray2: np.ndarray,
         size: int, blur_kernel: Tuple[int, int]) -> float:

    g1 = cv2.resize(gray1, (size, size))
    g2 = cv2.resize(gray2, (size, size))

    if blur_kernel and blur_kernel[0] > 0:
        g1 = cv2.GaussianBlur(g1, blur_kernel, 0)
        g2 = cv2.GaussianBlur(g2, blur_kernel, 0)

    g1 = g1.astype(np.float32) - g1.mean()
    g2 = g2.astype(np.float32) - g2.mean()

    g1 /= (g1.std() + 1e-6)
    g2 /= (g2.std() + 1e-6)

    num = float(np.sum(g1 * g2))
    den = float(np.sqrt(np.sum(g1*g1)) * np.sqrt(np.sum(g2*g2)) + 1e-6)

    return float(np.clip(num / den, -1, 1))


# ================================================================
# 클러스터링 메인 기능
# ================================================================
def cluster_folder(folder: Path, cfg: Dict):
    """
    folder 내부 이미지들을 순차적으로 분석하여 클러스터링 CSV 파일 생성.

    cfg = {
        "GAMMA": float,
        "CHANGE_THRESH": float,
        "GAUSS_BLUR": (3,3),
        "ZNCC_SIZE": int,
        "MAX_CLUSTER_LEN": int,
        "DOWNSCALE": bool,
        "PREPROC_MAX_SIDE": int,
    }
    """

    GAMMA = cfg["GAMMA"]
    CHANGE_THRESH = cfg["CHANGE_THRESH"]
    GAUSS_BLUR = tuple(cfg["GAUSS_BLUR"])
    ZNCC_SIZE = int(cfg["ZNCC_SIZE"])
    MAXLEN = int(cfg["MAX_CLUSTER_LEN"])
    DOWNSCALE = cfg.get("DOWNSCALE", False)
    PREPROC_MAX_SIDE = cfg.get("PREPROC_MAX_SIDE", 0)

    imgs = list_images(folder)
    if not imgs:
        print(f"[WARN] No images in folder: {folder}")
        return

    # 초기 프레임 설정
    img_prev = load_bgr(imgs[0], DOWNSCALE, PREPROC_MAX_SIDE)
    if img_prev is None:
        print(f"[WARN] Can't read {imgs[0]}")
        return

    g_prev = gray_of(img_prev)
    lbp_prev = lbp_hist(g_prev, GAUSS_BLUR)

    # Anchor 초기화
    anchor_img, g_anchor, lbp_anchor = img_prev, g_prev, lbp_prev

    cluster_id = 0
    curr_len = 1

    rows = [{
        "filename": imgs[0].name,
        "cluster_id": 0,
        "index_in_folder": 0,
        "d_lbp_prev": 0.0,
        "zncc_prev": 1.0,
        "change_prev": 0.0,
        "d_lbp_anchor": 0.0,
        "zncc_anchor": 1.0,
        "change_anchor": 0.0,
        "is_cluster_start": 1,
        "cluster_len_so_far": 1,
        "reason": "start"
    }]

    # 프레임 반복
    for idx in range(1, len(imgs)):
        path = imgs[idx]
        img = load_bgr(path, DOWNSCALE, PREPROC_MAX_SIDE)
        if img is None:
            print(f"[WARN] Can't read {path}")
            continue

        g_cur = gray_of(img)
        lbp_cur = lbp_hist(g_cur, GAUSS_BLUR)

        # prev 비교
        d_prev = d_bhat(lbp_prev, lbp_cur)
        z_prev = zncc(g_prev, g_cur, ZNCC_SIZE, GAUSS_BLUR)
        change_prev = GAMMA * d_prev + (1 - GAMMA) * (1 - z_prev)

        # anchor 비교
        d_anchor = d_bhat(lbp_anchor, lbp_cur)
        z_anchor = zncc(g_anchor, g_cur, ZNCC_SIZE, GAUSS_BLUR)
        change_anchor = GAMMA * d_anchor + (1 - GAMMA) * (1 - z_anchor)

        # 클러스터 분기 조건
        limit_hit = (MAXLEN > 0) and (curr_len >= MAXLEN)
        is_new_cluster = (
            limit_hit or
            (change_prev >= CHANGE_THRESH) or
            (change_anchor >= CHANGE_THRESH)
        )

        if is_new_cluster:
            cluster_id += 1
            is_start = 1
            reason = "limit" if limit_hit else "new"
            curr_len = 1
            anchor_img, g_anchor, lbp_anchor = img, g_cur, lbp_cur
        else:
            is_start = 0
            reason = "similar"
            curr_len += 1

        rows.append({
            "filename": path.name,
            "cluster_id": cluster_id,
            "index_in_folder": idx,
            "d_lbp_prev": round(d_prev, 6),
            "zncc_prev": round(z_prev, 6),
            "change_prev": round(change_prev, 6),
            "d_lbp_anchor": round(d_anchor, 6),
            "zncc_anchor": round(z_anchor, 6),
            "change_anchor": round(change_anchor, 6),
            "is_cluster_start": is_start,
            "cluster_len_so_far": curr_len,
            "reason": reason
        })

        img_prev, g_prev, lbp_prev = img, g_cur, lbp_cur

    # 결과 저장
    out_csv = folder.parent / f"clusters_invariant_{folder.name}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {folder.name}: {len(rows)} frames → {cluster_id + 1} clusters → {out_csv}")


# ================================================================
# 여러 폴더에 일괄 적용
# ================================================================
def run_clustering(folders: List[Path], folder_cfg: Dict, default_cfg: Dict):
    for folder in folders:
        cfg = {**default_cfg, **folder_cfg.get(folder.name, {})}
        cluster_folder(folder, cfg)
