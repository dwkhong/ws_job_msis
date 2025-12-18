# yolov8_trainer.py
# -*- coding: utf-8 -*-

from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.torch_utils import strip_optimizer
from datetime import datetime
import csv, math, shutil, gc, time
import torch

# ==========================================
# CSV 유틸
# ==========================================
def read_csv(path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields

def pick_col(cands, fields):
    for c in cands:
        if c in fields:
            return c
    raise KeyError(f"Column missing: {cands}")

def safe_float(v):
    try:
        x = float(v)
        return (-1.0 if math.isnan(x) else x)
    except:
        return -1.0

# ==========================================
# Top-3 체크포인트 후처리
# ==========================================
def process_run(exp_dir: Path, weights_dir: Path, sum4_weights, strip=True, clean=True):
    csv_path = exp_dir / "results.csv"
    rows, fields = read_csv(csv_path)

    CAND = {
        "precision": ["metrics/precision(B)", "metrics/precision"],
        "recall":    ["metrics/recall(B)",    "metrics/recall"],
        "map50":     ["metrics/mAP50(B)",     "metrics/mAP50"],
        "map5095":   ["metrics/mAP50-95(B)",  "metrics/mAP50-95"],
    }

    col_p  = pick_col(CAND["precision"], fields)
    col_r  = pick_col(CAND["recall"],    fields)
    col_50 = pick_col(CAND["map50"],     fields)
    col_95 = pick_col(CAND["map5095"],   fields)

    top3 = {m:[(-1.0,-1)]*3 for m in ["precision","recall","map50","map5095","sum4"]}

    def upd(t3, val, ep):
        b, s, t = t3
        if val > b[0]:
            t3[:] = [(val,ep), b, s]
        elif val > s[0]:
            t3[1:] = [(val,ep), s]
        elif val > t[0]:
            t3[2] = (val,ep)

    for r in rows:
        if "epoch" not in r:
            continue
        ep  = int(r["epoch"])
        p   = safe_float(r[col_p])
        rc  = safe_float(r[col_r])
        m50 = safe_float(r[col_50])
        m95 = safe_float(r[col_95])
        s4  = p*sum4_weights["precision"] + rc*sum4_weights["recall"] + m50*sum4_weights["map50"] + m95*sum4_weights["map5095"]

        upd(top3["precision"], p, ep)
        upd(top3["recall"],    rc, ep)
        upd(top3["map50"],     m50, ep)
        upd(top3["map5095"],   m95, ep)
        upd(top3["sum4"],      s4, ep)

    kept = {"best.pt", "last.pt"}

    def copy_ckpt(metric, rank, val, ep):
        if ep < 0:
            return
        src = weights_dir / f"epoch{ep-1}.pt"
        if not src.exists():
            return
        dst = weights_dir / f"{metric}_{rank}_epoch{ep}_{val:.5f}.pt"
        shutil.copy2(src, dst)
        if strip:
            strip_optimizer(str(dst))
        kept.add(dst.name)

    for m in top3:
        for idx, rank in enumerate(["BEST","SECOND","THIRD"]):
            val, ep = top3[m][idx]
            copy_ckpt(m, rank, val, ep)

    if clean:
        for p in weights_dir.glob("epoch*.pt"):
            if p.name not in kept:
                p.unlink(missing_ok=True)

    # summary
    with (exp_dir / "best_summary.txt").open("w") as f:
        for m, arr in top3.items():
            f.write(f"{m}: {arr}\n")

# ==========================================
# 메인 학습 함수 — 실행 파일에서 호출
# ==========================================
def run_training(
    *,
    data_yaml,
    model_name,
    project_dir,
    img_size,
    epochs,
    box_w,
    cls_w,
    dfl_w,
    seed,
    optimizer_name,
    lr0,
    weight_decay,
    momentum,
    rect_mode,
    save_period,
    data_tag,
    sum4_weights,
):
    today = datetime.now().strftime("%Y%m%d")
    model_tag = Path(model_name).stem.lower()

    exp_name = (
        f"{data_tag}_{model_tag}_cls{cls_w}_box{box_w}_dfl{dfl_w}"
        f"_{optimizer_name}_seed{seed}_{today}"
    )

    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=epochs,
        box=box_w,
        cls=cls_w,
        dfl=dfl_w,
        optimizer=optimizer_name,
        lr0=lr0,
        weight_decay=weight_decay,
        momentum=momentum,
        rect=rect_mode,
        seed=seed,
        deterministic=True,
        save=True,
        save_period=save_period,
        project=project_dir,
        name=exp_name,
        exist_ok=True,
    )

    exp_dir = Path(project_dir) / exp_name
    weights_dir = exp_dir / "weights"

    process_run(exp_dir, weights_dir, sum4_weights)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n🔥 완료: {exp_name}")
    return exp_name
