#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_folder()로 생성된 CSV(clusters_invariant_*.csv)를 읽어서
이미지 및 라벨 파일명을 `_cluster_<id>` 형식으로 일괄 변경하는 유틸리티 모듈.
"""

from pathlib import Path
import csv
import re

# 파일명 패턴 (_cluster_<id>)
SUFFIX_RE = re.compile(r"_cluster_(\d+)$", re.IGNORECASE)

# cluster suffix 생성
def make_suffix(cluster_id: int) -> str:
    return f"_cluster_{cluster_id}"

def with_seq(p: Path, k: int) -> Path:
    """파일 충돌 시 __k 숫자를 붙여 충돌을 피함."""
    return p.with_name(f"{p.stem}__{k}{p.suffix}")

def ensure_nonconflicting(target: Path) -> Path:
    """이미 타겟 파일이 존재하면 __2, __3 형태로 안전한 새 파일명 생성."""
    if not target.exists():
        return target
    k = 2
    cand = with_seq(target, k)
    while cand.exists():
        k += 1
        cand = with_seq(target, k)
    return cand

def parse_folder_from_csv(csv_path: Path, img_root: Path) -> Path:
    """
    clusters_invariant_폴더명.csv → 폴더명 복구
    예: clusters_invariant_20250721_good_data.csv
    """
    m = re.match(r"clusters_invariant_(.+)\.csv$", csv_path.name)
    if not m:
        raise ValueError(f"CSV 이름 형식 오류: {csv_path.name}")
    return img_root / m.group(1)

def plan_changes(csv_path: Path, img_root: Path):
    """
    CSV 하나를 기반으로 파일 변경 계획 생성.
    returns: list[{old_jpg, new_jpg, old_txt, new_txt}]
    """
    folder = parse_folder_from_csv(csv_path, img_root)
    if not folder.exists():
        print(f"[WARN] 폴더 없음: {folder}")
        return []

    plans = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "cluster_id" not in reader.fieldnames:
            raise ValueError(f"[ERR] CSV 필수 컬럼 없음: {csv_path}")

        for row in reader:
            fn = row["filename"]
            cid = int(row["cluster_id"])

            if not fn.lower().endswith(".jpg"):
                continue

            jpg_path = folder / fn
            if not jpg_path.exists():
                continue

            stem = jpg_path.stem

            # 이미 붙어있으면 스킵
            if SUFFIX_RE.search(stem):
                continue

            txt_path = jpg_path.with_suffix(".txt")
            new_stem = f"{stem}{make_suffix(cid)}"

            new_jpg = jpg_path.with_name(f"{new_stem}{jpg_path.suffix}")
            new_txt = txt_path.with_name(f"{new_stem}{txt_path.suffix}") if txt_path.exists() else None

            plans.append({
                "old_jpg": jpg_path,
                "new_jpg": new_jpg,
                "old_txt": txt_path if txt_path.exists() else None,
                "new_txt": new_txt,
            })

    return plans


def apply_changes(plans, apply=True):
    """계획(plan)을 실제로 수행."""
    processed = 0

    for p in plans:
        old_jpg = p["old_jpg"]
        new_jpg = ensure_nonconflicting(p["new_jpg"])

        old_txt = p["old_txt"]
        new_txt = ensure_nonconflicting(p["new_txt"]) if p["new_txt"] else None

        if apply:
            if old_txt and new_txt:
                old_txt.rename(new_txt)
                new_jpg = ensure_nonconflicting(new_jpg)  # TXT rename 후 다시 확인
                old_jpg.rename(new_jpg)
            else:
                old_jpg.rename(new_jpg)

        processed += 1

        # 계획 출력
        if old_txt and new_txt:
            print(f"[JPG] {old_jpg.name} → {new_jpg.name}")
            print(f"[TXT] {old_txt.name} → {new_txt.name}")
        else:
            print(f"[JPG] {old_jpg.name} → {new_jpg.name} (txt 없음)")

    return processed


def rename_by_cluster(csv_dir: Path, apply=True):
    """
    CSV_DIR 내부의 clusters_invariant_*.csv 를 모두 읽어
    각 이미지에 `_cluster_<id>` suffix 를 붙임.
    """
    csv_files = sorted(csv_dir.glob("clusters_invariant_*.csv"))
    if not csv_files:
        print(f"[WARN] CSV 없음: {csv_dir}")
        return

    total = 0
    for csv_path in csv_files:
        print(f"\n[CSV] 처리 중 → {csv_path.name}")
        folder = parse_folder_from_csv(csv_path, csv_dir)
        print(f" → 대상 폴더: {folder}")

        plans = plan_changes(csv_path, csv_dir)
        if not plans:
            print(f"[INFO] 변경 사항 없음.")
            continue

        count = apply_changes(plans, apply=apply)
        total += count

    print(f"\n[OK] 총 {total}개 파일 이름 변경 완료" if apply else "\n[DRY] 적용하지 않고 계획만 출력함")
