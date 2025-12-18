# src/report_split_stats.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

def read_classes_txt(folder: Path):
    f = folder / "classes.txt"
    if f.exists():
        names = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {i: names[i] for i in range(len(names))}
    return {i: f"class_{i}" for i in range(8)}

def list_images(dirpath: Path):
    return [p for p in dirpath.rglob("*") if p.suffix in IMG_EXTS]

def count_split(folder: Path, split: str, id2name: dict):
    img_dir = folder / "images" / split
    lbl_dir = folder / "labels" / split
    cls_counter = Counter()
    total_labels = 0

    # 클래스 카운트
    if lbl_dir.exists():
        for txt in lbl_dir.rglob("*.txt"):
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    cid = int(float(parts[0]))
                except:
                    continue
                cls_counter[cid] += 1
                total_labels += 1

    # BG/이미지 수 계산
    bg = 0
    if img_dir.exists():
        imgs = list_images(img_dir)
        stems_img = {p.stem for p in imgs}
        stems_lbl = {p.stem for p in lbl_dir.rglob("*.txt")} if lbl_dir.exists() else set()
        bg = len(stems_img - stems_lbl)
    else:
        imgs = []

    total_images = len(imgs)
    labeled_images = len(stems_lbl) if lbl_dir.exists() else 0

    per_class = {id2name.get(k, f"class_{k}"): v for k, v in sorted(cls_counter.items())}
    return per_class, bg, total_images, labeled_images, total_labels


def generate_full_report(root: Path, targets: list, output_file: Path):
    SPLITS = ["train", "val"]
    global_total_images = 0
    global_total_labeled = 0
    global_total_bg = 0
    global_total_objects = 0
    global_class_totals = Counter()
    all_class_names_seen = set()
    report_lines = []

    PREFERRED_ORDER = [
        "Divot", "Fixed_Divot", "Diseased_Grass", "Confused_Object",
        "Pole", "Sprinkler", "Drain", "Golf ball"
    ]

    # ===== per-folder report
    for tgt in targets:
        id2name = read_classes_txt(tgt)
        all_class_names_seen.update(id2name.values())

        report_lines.append(f"\n=== {tgt.name} ===")
        for split in SPLITS:
            per_class, bg, total_images, labeled_images, total_labels = count_split(tgt, split, id2name)

            report_lines.append(
                f"[{split}] images={total_images}, labeled={labeled_images}, "
                f"BG(no-label)={bg}, total_objects={total_labels}"
            )

            for i in sorted(id2name.keys()):
                cname = id2name[i]
                report_lines.append(f"  {cname:16s}: {per_class.get(cname, 0)}")

            report_lines.append(f"  {'BG':16s}: {bg}")

            # accumulate totals
            global_total_images += total_images
            global_total_labeled += labeled_images
            global_total_bg += bg
            global_total_objects += total_labels
            for cname, cnt in per_class.items():
                global_class_totals[cname] += cnt

    # ===== TOTAL SECTION
    class_order = [c for c in PREFERRED_ORDER if c in all_class_names_seen] + \
                  [c for c in sorted(all_class_names_seen) if c not in PREFERRED_ORDER]

    total_lines = []
    total_lines.append("=== TOTAL (ALL DATASETS, train/val only) ===")
    total_lines.append(
        f"images={global_total_images}, labeled={global_total_labeled}, "
        f"BG(no-label)={global_total_bg}, total_objects={global_total_objects}"
    )

    for cname in class_order:
        total_lines.append(f"  {cname:16s}: {global_class_totals.get(cname, 0)}")

    total_lines.append(f"  {'BG':16s}: {global_total_bg}")

    # ===== SAVE FILE
    final_text = "\n".join(total_lines + [""] + report_lines)
    output_file.write_text(final_text, encoding="utf-8")

    return total_lines   # 실행 파일에서 summary 출력용
