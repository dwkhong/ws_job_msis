#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/evaluation_yolo.py
- TensorRT 엔진(.engine) YOLO 예측 저장
- mAP 계산(커스텀 ap_per_class 기반) + per-class
- AUROC 계산(FN을 score=0 양성으로 반영)
- (옵션) Divot ROC 테이블/곡선 저장
- (옵션) 예측/GT 시각화 저장 (GT=green, TP=blue, FP=red)
- summary.txt 저장
- (옵션) 터미널 출력(print_terminal=True)
"""

# ── [ultralytics 로그 무음: import 전에] ───────────────────────────────
import os, logging
os.environ["ULTRALYTICS_QUIET"] = "1"

from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import yaml
import math
import csv
import shutil
from sklearn.metrics import roc_auc_score, roc_curve, auc
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ap_per_class, box_iou

LOGGER.setLevel(logging.ERROR)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP")


# ==============================
# Utils
# ==============================
def _find_image_by_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_yolo_txt(path: Path | None):
    """Return Nx6 [cls, x, y, w, h, conf]. GT는 conf=1.0으로 들어올 수 있음."""
    if (path is None) or (not path.exists()):
        return np.zeros((0, 6), dtype=np.float32)

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split()))
            if len(vals) == 5:
                c, x, y, w, h = vals
                conf = 1.0
            else:
                c, x, y, w, h, conf = vals[:6]
            rows.append([c, x, y, w, h, conf])

    return np.asarray(rows, dtype=np.float32)


def _xywhn2xyxy_norm(xywhn: np.ndarray):
    """
    정규화 xywh (0~1) -> 정규화 xyxy (0~1)
    ✅ 이미지 W/H 필요 없음 (속도↑)
    """
    if len(xywhn) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x, y, w, h = xywhn.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _stable_conf_sort(conf: np.ndarray):
    idx = np.arange(len(conf))
    return np.lexsort((idx, -conf))


def _cat_or_empty(arrs, axis=0, empty_shape=None, dtype=None):
    arrs = [a for a in arrs if a is not None]
    if len(arrs) == 0:
        return np.zeros(empty_shape, dtype=dtype)
    try:
        return np.concatenate(arrs, axis=axis)
    except Exception:
        return np.zeros(empty_shape, dtype=dtype)


def _iou_xywhn(box1, box2):
    """YOLO xywhn(정규화) 형태로 IoU 계산 (viz용 간단 매칭)"""
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return (inter / union) if union > 0 else 0.0


# ==============================
# Main
# ==============================
def run_evaluation(
    engine_path: str,
    data_yaml: str,
    test_dir: str,
    gt_dir: str,
    img_size: int = 640,
    device: int = 0,
    pred_conf: float = 0.001,
    match_iou: float = 0.5,

    # viz
    draw_save: bool = False,
    draw_only_divot: bool = False,
    draw_score_min: float = 0.001,
    draw_dir_name: str = "viz",

    # divot roc
    draw_divot_roc: bool = True,
    target_class_name: str = "Divot",
    roc_thresh_grid: np.ndarray | None = None,
    roc_outdir_name: str = "roc_divot",

    # print/summary 기준 conf
    eval_conf_print_map: float = 0.5,
    eval_conf_print_auroc: float = 0.5,

    # output control
    out_root: str | None = None,
    run_name: str = "pred_latest",

    # predict options
    augment: bool = False,
    agnostic_nms: bool = True,
    rect: bool = False,

    # ✅ 터미널 출력 토글
    print_terminal: bool = True,
):
    """
    - pred_conf: 예측 저장 conf (보통 0.001 권장)
    - eval_conf_print_map: mAP 계산에 사용할 추가 필터(conf>=)
    - eval_conf_print_auroc: AUROC에서 TP/FP/TN/FN 요약을 위한 임계값
    """

    engine_path = str(engine_path)
    data_yaml = str(data_yaml)
    TEST_DIR = Path(test_dir)
    GT_LABEL_DIR = Path(gt_dir)

    # ===== class names
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    CLASS_NAMES = data_cfg.get("names", [])
    if isinstance(CLASS_NAMES, dict):  # {0:"Divot",...}
        CLASS_NAMES = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES.keys())]
    NUM_CLASSES = len(CLASS_NAMES)

    # ===== output dir
    project_root = (Path.cwd() / "evaluation_results") if out_root is None else Path(out_root)
    project_root.mkdir(parents=True, exist_ok=True)

    run_dir = project_root / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ===== inference
    model = YOLO(engine_path)
    print(f"🚀 Predicting (conf={pred_conf}) → {run_dir}")
    model.predict(
        source=str(TEST_DIR),
        imgsz=img_size,
        conf=pred_conf,
        iou=match_iou,
        device=device,
        rect=rect,
        agnostic_nms=agnostic_nms,
        augment=augment,
        deterministic=True,
        seed=1,
        save_txt=True,
        save_conf=True,
        project=project_root,
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
    PRED_LABEL_DIR = run_dir / "labels"
    print(f"✅ Predictions saved: {PRED_LABEL_DIR}")

    # ===== common
    IOU_THRS = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    gt_files = sorted(GT_LABEL_DIR.glob("*.txt"))
    pred_map = {p.stem: p for p in PRED_LABEL_DIR.glob("*.txt")}

    # ==============================
    # 1) mAP
    # ==============================
    tp_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []

    for gf in tqdm(gt_files, desc=f"Collect mAP stats (conf≥{eval_conf_print_map})"):
        stem = gf.stem
        preds = _load_yolo_txt(pred_map.get(stem))
        if len(preds):
            preds = preds[preds[:, 5] >= float(eval_conf_print_map)]
        gts = _load_yolo_txt(gf)

        gt_boxes = _xywhn2xyxy_norm(gts[:, 1:5]) if len(gts) else np.zeros((0, 4), dtype=np.float32)
        gt_cls = gts[:, 0].astype(int) if len(gts) else np.zeros((0,), dtype=int)

        pred_boxes = _xywhn2xyxy_norm(preds[:, 1:5]) if len(preds) else np.zeros((0, 4), dtype=np.float32)
        p_cls = preds[:, 0].astype(int) if len(preds) else np.zeros((0,), dtype=int)
        conf = preds[:, 5].astype(np.float32) if len(preds) else np.zeros((0,), dtype=np.float32)

        correct = np.zeros((len(preds), len(IOU_THRS)), dtype=bool)

        if len(preds) and len(gts):
            ious = box_iou(torch.tensor(pred_boxes, dtype=torch.float32),
                           torch.tensor(gt_boxes, dtype=torch.float32)).cpu().numpy()

            order = _stable_conf_sort(conf)
            pred_boxes, p_cls, conf = pred_boxes[order], p_cls[order], conf[order]
            ious = ious[order]

            for j, thr in enumerate(IOU_THRS):
                detected = []
                for k, pc in enumerate(p_cls):
                    best_iou, best_gt = 0.0, -1
                    for gi, gc in enumerate(gt_cls):
                        if gc != pc or gi in detected:
                            continue
                        iou_val = ious[k, gi]
                        if iou_val > best_iou:
                            best_iou, best_gt = iou_val, gi
                    if best_iou > thr:  # YOLO val과 동일 '>'
                        correct[k, j] = True
                        detected.append(best_gt)

        tp_list.append(correct)
        conf_list.append(conf)
        pred_cls_list.append(p_cls)
        target_cls_list.append(gt_cls)

    tp = _cat_or_empty(tp_list, axis=0, empty_shape=(0, len(IOU_THRS)), dtype=bool)
    conf_all = _cat_or_empty(conf_list, axis=0, empty_shape=(0,), dtype=np.float32)
    pred_cls_all = _cat_or_empty(pred_cls_list, axis=0, empty_shape=(0,), dtype=int)
    target_cls_all = _cat_or_empty(target_cls_list, axis=0, empty_shape=(0,), dtype=int)

    # ultralytics 버전 호환
    try:
        tp_c, fp_c, p, r, f1, ap, ap_class, *_ = ap_per_class(tp, conf_all, pred_cls_all, target_cls_all, iouv=IOU_THRS)
    except TypeError:
        tp_c, fp_c, p, r, f1, ap, ap_class, *_ = ap_per_class(tp, conf_all, pred_cls_all, target_cls_all)

    p_arr = np.asarray(p, dtype=float).reshape(-1)
    r_arr = np.asarray(r, dtype=float).reshape(-1)
    ap_np = np.asarray(ap, dtype=float)

    # ✅ ap shape 방어 (버전/설정에 따라 ap가 (nc,)만 오는 경우가 있음)
    # - 그 경우는 "AP50만" 온 것으로 간주
    if ap_np.ndim == 1:
        ap_np = ap_np.reshape(-1, 1)

    p_mean = float(p_arr.mean()) if p_arr.size else 0.0
    r_mean = float(r_arr.mean()) if r_arr.size else 0.0
    mAP50 = float(ap_np[:, 0].mean()) if ap_np.size else 0.0
    mAP5095 = float(ap_np.mean()) if (ap_np.size and ap_np.shape[1] > 1) else mAP50

    ap_dict = {}
    for i, c in enumerate(ap_class):
        cid = int(c)
        pi = float(p_arr[i]) if i < p_arr.size else 0.0
        ri = float(r_arr[i]) if i < r_arr.size else 0.0
        ap50_i = float(ap_np[i, 0]) if i < ap_np.shape[0] else 0.0
        ap5095_i = float(ap_np[i].mean()) if (i < ap_np.shape[0] and ap_np.shape[1] > 1) else ap50_i
        ap_dict[cid] = (pi, ri, ap50_i, ap5095_i)

    # ==============================
    # 2) AUROC (FN score=0 양성 포함)
    # ==============================
    per_true = {i: [] for i in range(NUM_CLASSES)}
    per_score = {i: [] for i in range(NUM_CLASSES)}

    for gf in tqdm(gt_files, desc="Matching for AUROC (GT-based)"):
        stem = gf.stem
        preds = _load_yolo_txt(pred_map.get(stem))
        gts = _load_yolo_txt(gf)

        # 예측 없음 + GT 존재 => 전부 FN (score=0 양성)
        if len(preds) == 0 and len(gts) > 0:
            for c in gts[:, 0].astype(int):
                if 0 <= c < NUM_CLASSES:
                    per_true[c].append(1)
                    per_score[c].append(0.0)
            continue

        gt_boxes = _xywhn2xyxy_norm(gts[:, 1:5]) if len(gts) else np.zeros((0, 4), dtype=np.float32)
        gt_cls = gts[:, 0].astype(int) if len(gts) else np.zeros((0,), dtype=int)

        pred_boxes = _xywhn2xyxy_norm(preds[:, 1:5]) if len(preds) else np.zeros((0, 4), dtype=np.float32)
        pred_cls = preds[:, 0].astype(int) if len(preds) else np.zeros((0,), dtype=int)
        conf = preds[:, 5].astype(np.float32) if len(preds) else np.zeros((0,), dtype=np.float32)

        if len(preds) and len(gts):
            ious = box_iou(torch.tensor(pred_boxes, dtype=torch.float32),
                           torch.tensor(gt_boxes, dtype=torch.float32)).cpu().numpy()

            order = _stable_conf_sort(conf)
            pred_boxes, pred_cls, conf = pred_boxes[order], pred_cls[order], conf[order]
            ious = ious[order]

            used = set()
            for k, (pc, cf) in enumerate(zip(pred_cls, conf)):
                if not (0 <= pc < NUM_CLASSES):
                    continue

                best_iou, best_gt = 0.0, -1
                for gi, gc in enumerate(gt_cls):
                    if gc != pc or gi in used:
                        continue
                    iou_val = ious[k, gi]
                    if iou_val > best_iou:
                        best_iou, best_gt = iou_val, gi

                if best_iou > match_iou and best_gt != -1:
                    per_true[pc].append(1)
                    per_score[pc].append(float(cf))
                    used.add(best_gt)
                else:
                    per_true[pc].append(0)
                    per_score[pc].append(float(cf))

            # unmatched GT => FN (score=0 양성)
            for gi, gc in enumerate(gt_cls):
                if not (0 <= gc < NUM_CLASSES):
                    continue
                if gi not in used:
                    per_true[gc].append(1)
                    per_score[gc].append(0.0)

        elif len(preds) > 0 and len(gts) == 0:
            for pc, cf in zip(pred_cls, conf):
                if not (0 <= pc < NUM_CLASSES):
                    continue
                per_true[pc].append(0)
                per_score[pc].append(float(cf))

    aurocs = []
    for c in range(NUM_CLASSES):
        y_true = np.asarray(per_true[c], dtype=np.float32)
        y_score = np.asarray(per_score[c], dtype=np.float32)
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            aurocs.append(float("nan"))
        else:
            try:
                aurocs.append(float(roc_auc_score(y_true, y_score)))
            except Exception:
                aurocs.append(float("nan"))

    # thr 기반 TP/FP/TN/FN/GT (print/summary용)
    tp_thr = np.zeros(NUM_CLASSES, dtype=int)
    fp_thr = np.zeros(NUM_CLASSES, dtype=int)
    tn_thr = np.zeros(NUM_CLASSES, dtype=int)
    fn_thr = np.zeros(NUM_CLASSES, dtype=int)
    gt_thr = np.zeros(NUM_CLASSES, dtype=int)

    for c in range(NUM_CLASSES):
        y_true = np.asarray(per_true[c], dtype=np.int32)
        y_score = np.asarray(per_score[c], dtype=np.float32)
        if y_true.size == 0:
            continue
        y_pred = (y_score >= float(eval_conf_print_auroc)).astype(np.int32)
        tp_thr[c] = int(((y_pred == 1) & (y_true == 1)).sum())
        fp_thr[c] = int(((y_pred == 1) & (y_true == 0)).sum())
        tn_thr[c] = int(((y_pred == 0) & (y_true == 0)).sum())
        fn_thr[c] = int(((y_pred == 0) & (y_true == 1)).sum())
        gt_thr[c] = int((y_true == 1).sum())

    # ==============================
    # 3) Divot ROC (옵션)
    # ==============================
    rows_preview = None
    roc_auc_val = None

    if roc_thresh_grid is None:
        roc_thresh_grid = np.concatenate(([0.001], np.linspace(0.0, 1.0, 11)))

    if draw_divot_roc and NUM_CLASSES > 0:
        try:
            CID = CLASS_NAMES.index(target_class_name)
        except ValueError:
            CID = None

        if CID is not None:
            y_true_div = np.asarray(per_true[CID], dtype=np.int32)
            y_score_div = np.asarray(per_score[CID], dtype=np.float32)

            if y_true_div.size >= 2 and len(np.unique(y_true_div)) >= 2:
                roc_dir = run_dir / roc_outdir_name
                roc_dir.mkdir(parents=True, exist_ok=True)

                rows = []
                P = int((y_true_div == 1).sum())
                N = int((y_true_div == 0).sum())

                for thr in roc_thresh_grid:
                    y_pred = (y_score_div >= float(thr)).astype(np.int32)
                    TP = int(((y_pred == 1) & (y_true_div == 1)).sum())
                    FP = int(((y_pred == 1) & (y_true_div == 0)).sum())
                    TN = int(((y_pred == 0) & (y_true_div == 0)).sum())
                    FN = int(((y_pred == 0) & (y_true_div == 1)).sum())
                    TPR = (TP / P) if P > 0 else 0.0
                    FPR = (FP / N) if N > 0 else 0.0
                    rows.append([float(thr), TP, FP, TN, FN, float(TPR), float(FPR)])
                rows.sort(key=lambda x: x[0])
                rows_preview = rows[:11]

                csv_path = roc_dir / "divot_roc_table.csv"
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["threshold", "TP", "FP", "TN", "FN", "TPR", "FPR"])
                    w.writerows(rows)

                fpr, tpr, thr = roc_curve(y_true_div, y_score_div)
                roc_auc_val = float(auc(fpr, tpr))

                full_csv_path = roc_dir / "divot_roc_full_points.csv"
                with open(full_csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["threshold", "TPR", "FPR"])
                    for th, tpr_v, fpr_v in zip(thr, tpr, fpr):
                        w.writerow([float(th), float(tpr_v), float(fpr_v)])

                plt.figure(figsize=(5, 5))
                plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={roc_auc_val:.4f})")
                plt.fill_between(fpr, tpr, 0, alpha=0.15)
                plt.plot([0, 1], [0, 1], "--", linewidth=1, label="chance")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title(f"ROC — {target_class_name}")
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                fig_path = roc_dir / "divot_roc_curve.png"
                plt.savefig(fig_path, bbox_inches="tight", dpi=150)
                plt.close()

    # ==============================
    # 4) Visualization (옵션)
    # ==============================
    viz_dir = None
    if draw_save:
        viz_dir = run_dir / draw_dir_name
        viz_dir.mkdir(parents=True, exist_ok=True)

        TARGET_CID = None
        if draw_only_divot:
            try:
                TARGET_CID = CLASS_NAMES.index("Divot")
            except ValueError:
                TARGET_CID = None

        img_paths = []
        for ext in IMG_EXTS:
            img_paths.extend(sorted(TEST_DIR.glob(f"*{ext}")))
        img_paths = sorted(set(img_paths))

        print(f"🖼  Drawing viz → {viz_dir} (conf≥{draw_score_min}, only_divot={draw_only_divot})")

        for img_path in tqdm(img_paths, desc="Drawing"):
            name = img_path.stem
            gt_path = GT_LABEL_DIR / f"{name}.txt"
            pred_path = PRED_LABEL_DIR / f"{name}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            # GT
            gt_boxes = []
            if gt_path.exists():
                gts = _load_yolo_txt(gt_path)
                for row in gts:
                    cls = int(row[0])
                    box = row[1:5].tolist()
                    if TARGET_CID is not None and cls != TARGET_CID:
                        continue
                    gt_boxes.append((cls, box))

            # preds
            preds = []
            if pred_path.exists():
                pr = _load_yolo_txt(pred_path)
                for row in pr:
                    cls = int(row[0])
                    box = row[1:5].tolist()
                    confv = float(row[5])
                    if confv < float(draw_score_min):
                        continue
                    if TARGET_CID is not None and cls != TARGET_CID:
                        continue
                    preds.append((cls, box, confv))

            # draw GT (green)
            for cls, box in gt_boxes:
                cx, cy, bw, bh = box
                x1, y1 = int((cx - bw/2) * W), int((cy - bh/2) * H)
                x2, y2 = int((cx + bw/2) * W), int((cy + bh/2) * H)
                color = (0, 255, 0)
                label = f"GT-{CLASS_NAMES[cls] if 0 <= cls < NUM_CLASSES else cls}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # draw preds (TP=blue, FP=red)
            for cls, box, confv in preds:
                cx, cy, bw, bh = box
                x1, y1 = int((cx - bw/2) * W), int((cy - bh/2) * H)
                x2, y2 = int((cx + bw/2) * W), int((cy + bh/2) * H)

                matched = any((cls == g_cls and _iou_xywhn(box, g_box) >= match_iou) for g_cls, g_box in gt_boxes)
                if matched:
                    color = (255, 0, 0)  # blue
                    tag = "TP"
                else:
                    color = (0, 0, 255)  # red
                    tag = "FP"

                label = f"{tag}-{CLASS_NAMES[cls] if 0 <= cls < NUM_CLASSES else cls} {confv:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out_path = viz_dir / f"{name}.jpg"
            cv2.imwrite(str(out_path), img)

    # ==============================
    # 5) Summary 저장 + (옵션) 터미널 출력
    # ==============================
    summary_lines = []

    # mAP
    summary_lines.append(f"===== MANUAL EVALUATION (mAP@conf>= {eval_conf_print_map}) =====")
    summary_lines.append(f"all              mP={p_mean:.3f}  mR={r_mean:.3f}  mAP50={mAP50:.3f}  mAP50-95={mAP5095:.3f}")
    for i in range(NUM_CLASSES):
        pi, ri, ap50_i, ap5095_i = ap_dict.get(i, (0.0, 0.0, 0.0, 0.0))
        summary_lines.append(
            f"{CLASS_NAMES[i]:<15}  P={pi:.3f}  R={ri:.3f}  mAP50={ap50_i:.3f}  mAP50-95={ap5095_i:.3f}"
        )
    summary_lines.append("")

    # AUROC + thr confusion
    summary_lines.append(f"===== AUROC @ conf >= {eval_conf_print_auroc} =====")
    for i in range(NUM_CLASSES):
        auc_str = f"{aurocs[i]:.4f}" if not math.isnan(aurocs[i]) else "NaN"
        summary_lines.append(
            f"{CLASS_NAMES[i]:<15s}: AUROC={auc_str:<8s} | "
            f"TP={tp_thr[i]:<4d} FP={fp_thr[i]:<4d} TN={tn_thr[i]:<4d} "
            f"FN={fn_thr[i]:<4d} GT={gt_thr[i]:<4d}"
        )

    # Divot ROC preview (있으면)
    if draw_divot_roc:
        summary_lines.append("")
        summary_lines.append("===== Divot ROC Threshold Preview (first 11) =====")
        if rows_preview is None:
            summary_lines.append("Divot ROC 미생성(양/음 부족 또는 class not found)")
        else:
            summary_lines.append("thr     TP    FP    TN    FN    TPR      FPR")
            for r in rows_preview:
                summary_lines.append(
                    f"{r[0]:.3f}  {r[1]:5d} {r[2]:5d} {r[3]:5d} {r[4]:5d}  {r[5]:.4f}  {r[6]:.4f}"
                )

    summary_path = run_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if print_terminal:
        RESET = "\033[0m"
        BOLD  = "\033[1m"
        YEL   = "\033[33m"

        print(f"\n================= ✅ MANUAL EVALUATION (mAP@conf≥{eval_conf_print_map}) =================")
        print(f"{'all':<15}  mP={p_mean:.3f}  mR={r_mean:.3f}  mAP50={mAP50:.3f}  mAP50-95={mAP5095:.3f}")
        for i in range(NUM_CLASSES):
            pi, ri, ap50_i, ap5095_i = ap_dict.get(i, (0.0, 0.0, 0.0, 0.0))
            print(f"{CLASS_NAMES[i]:<15}  P={pi:.3f}  R={ri:.3f}  mAP50={ap50_i:.3f}  mAP50-95={ap5095_i:.3f}")
        print("======================================================================")

        print(f"\n📈 AUROC Evaluation per class (TP/FP/TN/FN/GT @ conf ≥ {eval_conf_print_auroc})")
        print("─────────────────────────────────────────────────────────────")
        for i in range(NUM_CLASSES):
            auc_str = f"{aurocs[i]:.4f}" if not math.isnan(aurocs[i]) else "NaN"
            line = (f"{CLASS_NAMES[i]:<18s}: AUROC={auc_str:<8s} | "
                    f"TP={tp_thr[i]:<4d} FP={fp_thr[i]:<4d} TN={tn_thr[i]:<4d} "
                    f"FN={fn_thr[i]:<4d} GT={gt_thr[i]:<4d}")

            # ✅ Divot 줄만 강조
            if CLASS_NAMES[i] == target_class_name:
                print(f"{BOLD}{YEL}{line}{RESET}")
            else:
                print(line)
        print("─────────────────────────────────────────────────────────────")

        if draw_divot_roc and rows_preview is not None:
            print("\n[ROC] Divot — threshold sweep preview")
            print("========================================================")
            print("thr     TP    FP    TN    FN    TPR      FPR")
            for r in rows_preview:
                print(f"{r[0]:.3f}  {r[1]:5d} {r[2]:5d} {r[3]:5d} {r[4]:5d}  {r[5]:.4f}  {r[6]:.4f}")
            print("========================================================")

        print(f"\n📄 Summary saved → {summary_path}\n")

    return {
        "mP": p_mean,
        "mR": r_mean,
        "mAP50": mAP50,
        "mAP5095": mAP5095,
        "summary_path": summary_path,
        "save_dir": run_dir,
        "summary_lines": summary_lines,
    }
