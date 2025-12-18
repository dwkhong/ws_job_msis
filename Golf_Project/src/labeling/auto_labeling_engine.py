# utils/autolabel_engine.py
# -*- coding: utf-8 -*-

from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 유틸
# ============================================================
def collect_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def ensure_label_path(img_path: Path) -> Path:
    """images → labels 경로 자동 변환"""
    parts = list(img_path.parts)
    if "images" in parts:
        i = parts.index("images")
        parts[i] = "labels"
        label_dir = Path(*parts[:-1])
    else:
        label_dir = img_path.parent / "labels"

    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / (img_path.stem + ".txt")


def write_yolo_txt(label_path: Path, rows: List[Tuple[int, float, float, float, float]]):
    if not rows:
        if label_path.exists():
            label_path.unlink()
        return False

    with open(label_path, "w") as f:
        for cls, cx, cy, w, h in rows:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return True


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _xyxy_to_xywhn(xyxy, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    return (cx/img_w, cy/img_h, w/img_w, h/img_h)


# ============================================================
# 모델 inference
# ============================================================
def detect(model: YOLO, img: Path, imgsz: int, conf: float, classes=None):
    r = model.predict(
        source=str(img),
        classes=classes,
        conf=conf,
        imgsz=imgsz,
        device=0,
        verbose=False
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy  = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)

    try:
        xywhn = r.boxes.xywhn.cpu().numpy()
        has_xywhn = True
    except:
        has_xywhn = False

    H, W = r.orig_shape
    out = []
    for i in range(len(clses)):
        if has_xywhn:
            cx, cy, w, h = map(float, xywhn[i])
        else:
            cx, cy, w, h = _xyxy_to_xywhn(xyxy[i], W, H)

        out.append({
            "cls":  int(clses[i]),
            "conf": float(confs[i]),
            "xyxy": tuple(map(float, xyxy[i])),
            "xywhn": (cx, cy, w, h)
        })
    return out


# ============================================================
# 메인 자동 라벨링 기능
# ============================================================
def auto_label(
    engine_path: str,
    imgsz: int,
    img_dir: str,
    min_conf: dict,
    confusion_iou: float = 0.5,
    delta_conf: float = 0.01,
    base_conf: float = 0.01,
    require_minconf_for_confusion: bool = False,
    require_both_minconf: bool = False,
    progress_every: int = 50
):

    model = YOLO(engine_path)
    img_dir = Path(img_dir)
    images = collect_images(img_dir)

    per_class_counts = defaultdict(int)
    images_labeled = 0
    total_labels = 0

    total = len(images)
    print(f"[Start] Total images: {total} (progress every {progress_every})")
    sys.stdout.flush()

    for idx, img in enumerate(images, start=1):
        try:
            # 클래스별 detection
            other_boxes = detect(model, img, imgsz, base_conf, classes=[2,4,5,6,7])
            divot_boxes = detect(model, img, imgsz, base_conf, classes=[0])
            fixed_boxes = detect(model, img, imgsz, base_conf, classes=[1])

            used0, used1 = set(), set()
            keep_01, confused_rows = [], []

            # -------------------------
            # Divot(0) ↔ Fixed(1) confusion 처리
            # -------------------------
            for i, b0 in enumerate(divot_boxes):
                best_j, best_iou = -1, 0.0
                for j, b1 in enumerate(fixed_boxes):
                    if j in used1:
                        continue
                    iou = iou_xyxy(b0["xyxy"], b1["xyxy"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_j >= 0 and best_iou >= confusion_iou:
                    c0, c1 = b0["conf"], fixed_boxes[best_j]["conf"]

                    allow = True
                    if require_minconf_for_confusion:
                        if require_both_minconf:
                            allow = (c0 >= min_conf[0] and c1 >= min_conf[1])
                        else:
                            allow = (c0 >= min_conf[0] or c1 >= min_conf[1])

                    if abs(c0 - c1) < delta_conf and allow:
                        winner = b0 if c0 >= c1 else fixed_boxes[best_j]
                        cx, cy, w, h = winner["xywhn"]
                        confused_rows.append((3, cx, cy, w, h))
                        used0.add(i); used1.add(best_j)
                    else:
                        winner = b0 if c0 >= c1 else fixed_boxes[best_j]
                        if winner["conf"] >= min_conf.get(winner["cls"], 0.0):
                            keep_01.append(winner)
                        used0.add(i); used1.add(best_j)

            # 남은 Divot, Fixed
            for i, b0 in enumerate(divot_boxes):
                if i not in used0 and b0["conf"] >= min_conf[0]:
                    keep_01.append(b0)
            for j, b1 in enumerate(fixed_boxes):
                if j not in used1 and b1["conf"] >= min_conf[1]:
                    keep_01.append(b1)

            # -------------------------
            # 결과 rows 생성
            # -------------------------
            rows = []

            for b in keep_01:
                cx, cy, w, h = b["xywhn"]
                rows.append((b["cls"], cx, cy, w, h))

            rows.extend(confused_rows)

            for b in other_boxes:
                if b["conf"] >= min_conf.get(b["cls"], 0.0):
                    cx, cy, w, h = b["xywhn"]
                    rows.append((b["cls"], cx, cy, w, h))

            # -------------------------
            # YOLO txt 저장
            # -------------------------
            label_path = ensure_label_path(img)
            wrote = write_yolo_txt(label_path, rows)

            if wrote:
                images_labeled += 1
                total_labels += len(rows)
                for cls, *_ in rows:
                    per_class_counts[cls] += 1

        except Exception as e:
            sys.stdout.write(f"\n[Skip] #{idx}: {img.name} ({e})\n")
            sys.stdout.flush()

        # 진행 출력
        if idx % progress_every == 0 or idx == total:
            sys.stdout.write(f"[Progress] {idx}/{total}\n")
            sys.stdout.flush()

    # ============================================================
    # 최종 요약
    # ============================================================
    print("\n========== Auto-Labeling Summary ==========")
    print(f"Images scanned     : {total}")
    print(f"Images labeled     : {images_labeled}")
    print(f"Total labels saved : {total_labels}")
    print("-------------------------------------------")

    class_names = {
        0:"Divot", 1:"Fixed_Divot", 2:"Diseased_Grass",
        3:"Confused_Object", 4:"Pole", 5:"Sprinkler",
        6:"Drain", 7:"Golf ball"
    }

    for cid in range(8):
        print(f"{cid:>1} ({class_names[cid]:>16}): {per_class_counts.get(cid,0)}")

    print("===========================================\n")
