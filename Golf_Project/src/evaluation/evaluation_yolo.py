# utils/evaluator.py
# -*- coding: utf-8 -*-

import os, logging, shutil, csv, math, yaml, cv2, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ap_per_class, box_iou
from datetime import datetime

LOGGER.setLevel(logging.ERROR)

# -------------------------------------------------------------
# 메인 실행 함수 (run_evaluation.py에서 호출)
# -------------------------------------------------------------
def run_evaluation(
    engine_path: str,
    data_yaml: str,
    test_dir: str,
    gt_dir: str,
    img_size: int = 640,
    device: int = 0,
    pred_conf: float = 0.001,
    match_iou: float = 0.5,
    draw_save: bool = False,
    draw_only_divot: bool = False,
    draw_score_min: float = 0.001,
    draw_dir_name: str = "viz",
    draw_divot_roc: bool = True,
    eval_conf_print_map: float = 0.5,
    eval_conf_print_auroc: float = 0.5
):

    # =====================================================
    # 모델 / 데이터 로드
    # =====================================================
    model = YOLO(engine_path)

    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    CLASS_NAMES = data_cfg["names"]
    NUM_CLASSES = len(CLASS_NAMES)

    TEST_DIR     = Path(test_dir)
    GT_LABEL_DIR = Path(gt_dir)
    project_root = Path.cwd() / "evaluation_results"
    project_root.mkdir(parents=True, exist_ok=True)

    run_dir = project_root / "pred_latest"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 1) TensorRT inference → prediction txt 저장
    # =====================================================
    print(f"🚀 Predicting (conf={pred_conf}) → {run_dir}")
    model.predict(
        source=str(TEST_DIR),
        imgsz=img_size,
        conf=pred_conf,
        iou=match_iou,
        device=device,
        save_txt=True,
        save_conf=True,
        project=project_root,
        name="pred_latest",
        exist_ok=True,
        verbose=False
    )
    PRED_LABEL_DIR = run_dir / "labels"

    # =====================================================
    # 2) 내부 기능 유틸
    # =====================================================
    def load_yolo_txt(p: Path):
        if not p or not p.exists():
            return np.zeros((0, 6), dtype=np.float32)
        rows = []
        for line in open(p):
            vals = list(map(float, line.split()))
            if len(vals) == 5:
                c, x, y, w, h = vals; conf = 1.0
            else:
                c, x, y, w, h, conf = vals
            rows.append([c, x, y, w, h, conf])
        return np.asarray(rows, dtype=np.float32)

    def xywhn2xyxy_pixels(xywhn, W, H):
        if len(xywhn)==0:
            return np.zeros((0,4),dtype=np.float32)
        x,y,w,h = xywhn.T
        return np.stack([
            (x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H
        ],axis=1).astype(np.float32)

    def stable_conf_sort(conf):
        idx = np.arange(len(conf))
        return np.lexsort((idx, -conf))

    # =====================================================
    # 3) mAP 계산
    # =====================================================
    IOU_THRS = np.linspace(0.5, 0.95, 10)
    gt_files  = sorted(GT_LABEL_DIR.glob("*.txt"))
    pred_map  = {p.stem: p for p in PRED_LABEL_DIR.glob("*.txt")}

    tp_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []

    for gf in tqdm(gt_files, desc="Collect mAP stats…"):
        stem = gf.stem
        preds = load_yolo_txt(pred_map.get(stem))
        preds = preds[preds[:,5] >= eval_conf_print_map] if len(preds) else preds
        gts   = load_yolo_txt(gf)

        gt_boxes = xywhn2xyxy_pixels(gts[:,1:5],1920,1536) if len(gts) else np.zeros((0,4))
        gt_cls   = gts[:,0].astype(int) if len(gts) else np.zeros((0,))
        pred_box = xywhn2xyxy_pixels(preds[:,1:5],1920,1536) if len(preds) else np.zeros((0,4))
        p_cls    = preds[:,0].astype(int) if len(preds) else np.zeros((0,))
        conf     = preds[:,5].astype(np.float32) if len(preds) else np.zeros((0,))

        correct = np.zeros((len(preds), len(IOU_THRS)), dtype=bool)
        if len(preds) and len(gts):
            ious = box_iou(torch.tensor(pred_box), torch.tensor(gt_boxes)).numpy()
            order = stable_conf_sort(conf)
            pred_box, p_cls, conf = pred_box[order], p_cls[order], conf[order]
            ious = ious[order]

            for j,thr in enumerate(IOU_THRS):
                used=[]
                for k,pc in enumerate(p_cls):
                    best_iou=0; best_gt=-1
                    for gi,gc in enumerate(gt_cls):
                        if gc!=pc or gi in used: continue
                        if ious[k,gi] > best_iou:
                            best_iou, best_gt = ious[k,gi], gi
                    if best_iou > thr:
                        correct[k,j] = True
                        used.append(best_gt)

        tp_list.append(correct)
        conf_list.append(conf)
        pred_cls_list.append(p_cls)
        target_cls_list.append(gt_cls)

    # cat
    def cat_or_empty(lst, axis=0, empty_shape=(0,), dtype=float):
        lst = [x for x in lst if x is not None]
        if not lst:
            return np.zeros(empty_shape,dtype=dtype)
        try: return np.concatenate(lst,axis=axis)
        except: return np.zeros(empty_shape,dtype=dtype)

    tp = cat_or_empty(tp_list, axis=0, empty_shape=(0,len(IOU_THRS)),dtype=bool)
    conf_all = cat_or_empty(conf_list, axis=0)
    pred_cls_all = cat_or_empty(pred_cls_list,axis=0,dtype=int)
    target_cls_all = cat_or_empty(target_cls_list,axis=0,dtype=int)

    p,r,f1,ap,ap_class,*_ = ap_per_class(tp, conf_all, pred_cls_all, target_cls_all)
    ap=np.asarray(ap)
    if ap.ndim==1: ap=ap.reshape(-1,len(IOU_THRS))

    p_mean=float(np.mean(p)) if len(p) else 0
    r_mean=float(np.mean(r)) if len(r) else 0
    mAP50=float(np.mean(ap[:,0])) if ap.size else 0
    mAP5095=float(np.mean(ap)) if ap.size else 0

    # =====================================================
    # 4) AUROC 계산 + FN 반영
    # =====================================================
    # 생략: (전체 코드와 동일)
    # → 결과 변수 aurocs[], tp_thr[], fp_thr[], ... 등 반환 가능하게 작성

    # -----------------------------------------------------
    # 최종 결과 summary 저장
    # -----------------------------------------------------
    summary = []
    summary.append("===== mAP =====")
    summary.append(f"mP={p_mean:.3f} mR={r_mean:.3f} mAP50={mAP50:.3f} mAP50-95={mAP5095:.3f}")

    summary_path = run_dir / "summary.txt"
    summary_path.write_text("\n".join(summary),encoding="utf-8")

    return {
        "mP":p_mean,
        "mR":r_mean,
        "mAP50":mAP50,
        "mAP5095":mAP5095,
        "summary_path":summary_path,
        "save_dir":run_dir
    }
