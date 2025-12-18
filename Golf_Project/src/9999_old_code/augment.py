#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict, deque
import random, csv, re
import cv2
import albumentations as A

# 🔥 추가: dup/bg 증강본 필터용 정규식
EXCLUDE_RE = re.compile(r'_(dup|bg)\d+$', re.IGNORECASE)

# ========== 대상 폴더들 ==========
BASE = Path("/home/dw/ws_job_msislab/Golf_Project/data/for_study/20251024_merge_data")

BASES = [
    #BASE / "20250721_good_data",
    BASE / "20250725_good_data",
    #BASE / "20250904_good_data",
    #BASE / "20250929_good_data",
    #BASE / "20250930_good_data",
]

# ========== 공통 설정 ==========
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"]
SEED = 33
random.seed(SEED)

# 클래스 매핑(8클래스, 필요시 변경)
# 0 Divot, 1 Fixed_Divot, 2 Diseased_Grass, 3 Confused_Object, 4 Pole, 5 Sprinkler, 6 Drain, 7 Golf ball
TARGET = {
    0: 346,  # Divot
    1: 956,  # Fixed_Divot
    2: 32,   # Diseased_Grass
    3: 0,    # Confused_Object
    4: 11,   # Pole
    5: 21,   # Sprinkler
    6: 6,   # Drain
    7: 22,   # Golf ball
}
ALL_CLASS_IDS = list(range(8))
ALLOW_IDS = set(ALL_CLASS_IDS)

# --- 중복 억제 파라미터 ---
MAX_USES_BASE = 1                     # 기본 재사용 상한(낮게)
MAX_USES_BOOST_PER_CLASS = {          # 부족 클래스일 때 가산치
    0: 2, 1: 2, 2: 3, 3: 0, 4: 3, 5: 3, 6: 3, 7: 3
}
MAX_PER_IMAGE_HARD = 1                # 한 원본에서 절대 최대 생성 수
RECENT_COOLDOWN = 5                   # 최근 사용 원본 쿨다운 길이

# 배경(라벨 없는 train 이미지) 증강 배수
BG_AUG_MULTIPLIER = 2                 # 원본 1 + 증강 3 = 총 4배

# ========== 색/감마 변환 정의(플립 제거, 색/감마만) ==========
def _build_color_ops():
    ops = {
        "gamma_bright": A.RandomGamma(gamma_limit=(55, 85), p=1.0),    # 더 밝게
        "gamma_dark":   A.RandomGamma(gamma_limit=(120, 180), p=1.0),  # 더 어둡게
        "warm": A.Compose([
            A.ColorJitter(brightness=0.10, contrast=0.15, saturation=0.08, hue=0.015, p=1.0),
            A.RGBShift(r_shift_limit=(8, 20), g_shift_limit=(-4, 6), b_shift_limit=(-20, -8), p=0.9),
            A.RandomGamma(gamma_limit=(95, 120), p=1.0),
        ], p=1.0),
        "cool": A.Compose([
            A.ColorJitter(brightness=0.08, contrast=0.12, saturation=0.06, hue=0.015, p=1.0),
            A.RGBShift(r_shift_limit=(-20, -8), g_shift_limit=(-6, 6), b_shift_limit=(8, 20), p=0.9),
            A.RandomGamma(gamma_limit=(95, 120), p=1.0),
        ], p=1.0),
        "desat": A.Compose([
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-45, -25), val_shift_limit=(-5, 10), p=1.0),
            A.RandomGamma(gamma_limit=(90, 115), p=1.0),
        ], p=1.0),
        "satboost": A.Compose([
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(20, 45), val_shift_limit=(-5, 10), p=1.0),
            A.RandomGamma(gamma_limit=(90, 115), p=1.0),
        ], p=1.0),
        "identity": A.Compose([], p=1.0),
    }
    return ops

COLOR_OPS = _build_color_ops()

# 선택 가중치(원본 유지 비활성화: identity 제외)
WEIGHTS = {
    "gamma_bright": 1.0,
    "gamma_dark":   1.0,
    "warm":         1.0,
    "cool":         1.0,
    "desat":        0.8,
    "satboost":     0.8,
    "identity":     0.3,

}

def build_aug(weights: dict) -> A.Compose:
    choices = []
    for name, t in COLOR_OPS.items():
        w = float(weights.get(name, 0.0))
        if w > 0:
            t.p = w
            choices.append(t)
    if not choices:
        choices = [A.NoOp(p=1.0)]

    return A.Compose([
        A.HorizontalFlip(p=0.25),
        A.OneOf(choices, p=1.0),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25
    ))

aug = build_aug(WEIGHTS)

# ========== 유틸 ==========
def read_yolo_label(lbl: Path):
    boxes, cls = [], []
    if not lbl.exists(): return boxes, cls
    with open(lbl, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5: continue
            try:
                c = int(float(p[0]))
            except:
                continue
            try:
                x, y, w, h = map(float, p[1:5])
            except:
                continue
            x = min(max(x, 0.0), 1.0); y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0); h = min(max(h, 0.0), 1.0)
            if w <= 0.0 or h <= 0.0: continue
            boxes.append([x, y, w, h]); cls.append(c)
    return boxes, cls

def write_yolo_label(lbl_path: Path, boxes, cls, *, allow_ids=ALLOW_IDS):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w") as f:
        for c, (x, y, w, h) in zip(cls, boxes):
            try:
                c_int = int(float(c))
            except:
                continue
            if allow_ids is not None and c_int not in allow_ids:
                continue
            x = float(max(0.0, min(1.0, x)))
            y = float(max(0.0, min(1.0, y)))
            w = float(max(0.0, min(1.0, w)))
            h = float(max(0.0, min(1.0, h)))
            if w <= 0.0 or h <= 0.0:
                continue
            f.write(f"{c_int} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def yolo_sanitize(boxes):
    eps = 1e-6; out = []
    for x, y, w, h in boxes:
        l = max(0.0, min(1.0, x - w/2.0))
        r = max(0.0, min(1.0, x + w/2.0))
        t = max(0.0, min(1.0, y - h/2.0))
        b = max(0.0, min(1.0, y + h/2.0))
        if r - l <= eps or b - t <= eps: continue
        nx = (l + r)/2.0; ny = (t + b)/2.0; nw = (r - l); nh = (b - t)
        nx = min(max(nx, eps), 1.0 - eps); ny = min(max(ny, eps), 1.0 - eps)
        nw = min(max(nw, eps), 1.0 - eps); nh = min(max(nh, eps), 1.0 - eps)
        out.append([nx, ny, nw, nh])
    return out

def collect_background_images(img_dir: Path, lbl_dir: Path):
    stems_lbl = {p.stem for p in lbl_dir.rglob("*.txt")} if lbl_dir.exists() else set()
    bg_images = []
    for ext in IMG_EXTS:
        for img in img_dir.rglob(f"*{ext}"):
            if img.stem not in stems_lbl:
                bg_images.append(img)
    return sorted(bg_images)

def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def next_unique_name(stem: str, used: set, tag: str):
    i = 1
    while True:
        s = f"{stem}_{tag}{i:04d}"
        if s not in used:
            used.add(s)
            return s
        i += 1

def effective_cap_for_file(file_cls_counter: Counter, deficit: Counter):
    cap = MAX_USES_BASE
    best_boost = 0
    for c, n in file_cls_counter.items():
        if n <= 0: continue
        if deficit.get(c, 0) > 0:
            best_boost = max(best_boost, MAX_USES_BOOST_PER_CLASS.get(c, 0))
    return cap + best_boost

def current_counts(label_dir: Path):
    cnt = Counter()
    for lbl in label_dir.rglob("*.txt"):
        _, cls = read_yolo_label(lbl)
        cnt.update(cls)
    return cnt

def list_train_stems(img_dir: Path):
    return {p.stem for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS}

# ========== 메인 루틴 ==========
def run_one_base(BASE: Path):
    IMG_DIR = BASE / "images" / "train"
    LBL_DIR = BASE / "labels" / "train"
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    # 유니크명 충돌 방지: 이미 존재하는 stem 모두 확보
    used_names = set(list_train_stems(IMG_DIR))

    # 1) 배경 증강
    bg_list = collect_background_images(IMG_DIR, LBL_DIR)
    print(f"\n[{BASE.name}] BG images in train: {len(bg_list)}")

    log_path = BASE / "aug_train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["src_img","src_lbl","out_img","out_lbl","add_per_class","totals_after"])

    for bg_img in bg_list:
        try:
            img = load_image(bg_img)
        except Exception:
            continue
        for _ in range(BG_AUG_MULTIPLIER):
            out_stem = next_unique_name(bg_img.stem, used_names, tag="bg")
            out_img = IMG_DIR / f"{out_stem}{bg_img.suffix.lower()}"
            out_lbl = LBL_DIR / f"{out_stem}.txt"
            transformed = aug(image=img, bboxes=[], class_labels=[])
            cv2.imwrite(str(out_img), transformed["image"])
            with open(out_lbl, "w") as f:
                pass
    print("  BG augmentation done.")

    # 2) 라벨 쌍 후보(오직 train)
    label_files = sorted(LBL_DIR.rglob("*.txt"))
    candidates = []  # (img_path, lbl_path, file_counts, stem)
    skipped = 0
    for lbl in label_files:
        boxes, cls_list = read_yolo_label(lbl)
        if not cls_list:
            continue
        img = None
        for ext in IMG_EXTS:
            cand = IMG_DIR / f"{lbl.stem}{ext}"
            if cand.exists():
                img = cand; break
        if img is None:
            skipped += 1
            continue

        # 🔥 추가: dup/bg로 생성된 증강본은 제외 (단, 원래 bg는 허용)
        if EXCLUDE_RE.search(lbl.stem):
            continue

        fcnt = Counter(cls_list)
        candidates.append((img, lbl, fcnt, lbl.stem))
    if not candidates:
        print("  No labeled train pairs. Skip.")
        return
    print(f"  Labeled train pairs: {len(candidates)}, skipped(no image match): {skipped}")

    # 3) 현재(train) 카운트 & 결손
    cur = current_counts(LBL_DIR)
    deficit = Counter({c: max(0, TARGET.get(c, 0) - cur.get(c, 0)) for c in ALL_CLASS_IDS})

    # 4) 반복 증강(중복 억제 포함)
    use_count = defaultdict(int)
    used_once = set()  # ✅ 추가
    all_once_done = False  # ✅ 추가
    recent_imgs = deque(maxlen=RECENT_COOLDOWN)
    MAX_ITERS = 200000
    iters = 0

    def base_score(deficit: Counter, fcnt: Counter):
        return sum(deficit[c] * fcnt.get(c, 0) for c in ALL_CLASS_IDS if deficit[c] > 0)

    while any(deficit[c] > 0 for c in ALL_CLASS_IDS) and iters < MAX_ITERS:
        iters += 1
        best = None; best_score = 0.0

        for img, lbl, fcnt, stem in candidates:
            if not all_once_done and use_count[img] >= 1:
                continue
            if img in recent_imgs:
                continue
            if use_count[img] >= MAX_PER_IMAGE_HARD and not all_once_done:
                continue
            cap = effective_cap_for_file(fcnt, deficit)
            if use_count[img] >= cap and not all_once_done:
                continue
            sc0 = base_score(deficit, fcnt)
            if sc0 <= 0:
                continue
            diversity_w = 1.0 / (1.0 + use_count[img])
            sc = sc0 * diversity_w
            if sc > best_score:
                best_score = sc
                best = (img, lbl, fcnt, stem)

        # ✅ 수정 시작
        if not best:
            # 남은 deficit이 있고, 후보 점수 모두 0이면 재사용 라운드로 전환
            useful_remaining = any(base_score(deficit, fcnt) > 0 for _, _, fcnt, _ in candidates)
            if useful_remaining and not all_once_done:
                print("  No more useful candidates (before all_once_done). Stop.")
                # ✅ 기존 break 대신 → 재사용 라운드로 강제 전환
                print("  🔁 Switching to reuse phase (dup002+).")
                all_once_done = True
                used_once.clear()
                use_count = defaultdict(int)
                recent_imgs.clear()
                continue
            elif not useful_remaining:
                print("  🔁 All originals exhausted, entering reuse round.")
                all_once_done = True
                used_once.clear()
                use_count = defaultdict(int)
                recent_imgs.clear()
                continue
            else:
                print("  🔁 All originals used or exhausted, deficit remains. Resetting for next reuse round.")
                all_once_done = True
                used_once.clear()
                use_count = defaultdict(int)
                recent_imgs.clear()
                continue
        # ✅ 수정 끝

        img_path, lbl_path, fcnt, stem = best
        out_stem = next_unique_name(stem, used_names, tag="dup")
        out_img = IMG_DIR / f"{out_stem}{img_path.suffix.lower()}"
        out_lbl = LBL_DIR / f"{out_stem}.txt"

        img = load_image(img_path)
        boxes0, cls0 = read_yolo_label(lbl_path)
        boxes0 = yolo_sanitize(boxes0)

        transformed = aug(image=img, bboxes=boxes0, class_labels=cls0)
        aug_img = transformed["image"]
        aug_boxes = transformed["bboxes"]
        aug_cls = transformed["class_labels"]

        keep_boxes, keep_cls = [], []
        for (x, y, w, h), c in zip(aug_boxes, aug_cls):
            if w > 0 and h > 0:
                keep_boxes.append([float(x), float(y), float(w), float(h)])
                keep_cls.append(int(float(c)))

        if not keep_cls or sum(Counter(keep_cls)[c] for c in ALL_CLASS_IDS if deficit[c] > 0) <= 0:
            use_count[img_path] += 1
            continue

        cv2.imwrite(str(out_img), aug_img)
        write_yolo_label(out_lbl, keep_boxes, keep_cls, allow_ids=ALLOW_IDS)

        add_cnt = Counter(keep_cls)
        cur.update(add_cnt)
        for c in ALL_CLASS_IDS:
            deficit[c] = max(0, TARGET.get(c, 0) - cur.get(c, 0))

        use_count[img_path] += 1
        used_once.add(img_path)
        if not all_once_done and len(used_once) >= len(candidates):
            all_once_done = True
            print("  ✅ All originals used once, reuse now allowed.")

        recent_imgs.append(img_path)

        add_str = "{" + ", ".join(f"{k}:{add_cnt.get(k,0)}" for k in ALL_CLASS_IDS) + "}"
        tot_str = "{" + ", ".join(f"{k}:{cur.get(k,0)}" for k in ALL_CLASS_IDS) + "}"
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([str(img_path), str(lbl_path), str(out_img), str(out_lbl), add_str, tot_str])

        if iters % 50 == 0 or all(deficit[c]==0 for c in ALL_CLASS_IDS):
            print(f"  [{iters}] cur={dict(cur)} deficit={dict(deficit)} last={out_stem}")

    print("  Final train counts:", dict(cur))
    print("  Remaining deficit:", dict(deficit))
    print(f"  Log: {log_path}")

# ===== 실행 =====
if __name__ == "__main__":
    for base in BASES:
        run_one_base(base)

