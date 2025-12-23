# aug_utils.py
from pathlib import Path
from collections import Counter, defaultdict, deque
import cv2, csv, random, re
import albumentations as A

# ============================================================
# 🔧 Augmentation 변환 생성
# ============================================================
def build_augmentor():
    color_ops = {
        "gamma_bright": A.RandomGamma(gamma_limit=(55, 85), p=1.0),
        "gamma_dark":   A.RandomGamma(gamma_limit=(120, 180), p=1.0),

        # ✅ Albumentations 최신 validation 대응:
        #    RGBShift는 (min,max) 튜플 + keyword 인자로 명시 (min<=max 보장)
        "warm": A.Compose([
            A.ColorJitter(brightness=0.10, contrast=0.15, saturation=0.08, hue=0.015),
            A.RGBShift(
                r_shift_limit=(0, 8),
                g_shift_limit=(-4, 0),
                b_shift_limit=(-20, 0),
                p=1.0,
            ),
        ]),

        "cool": A.Compose([
            A.ColorJitter(brightness=0.08, contrast=0.12, saturation=0.06, hue=0.015),
            A.RGBShift(
                r_shift_limit=(-20, 0),
                g_shift_limit=(-6, 0),
                b_shift_limit=(0, 10),
                p=1.0,
            ),
        ]),

        "identity": A.NoOp(),
    }

    return A.Compose([
        A.HorizontalFlip(p=0.25),
        A.OneOf(list(color_ops.values()), p=1.0)
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.25))


# ============================================================
# 🔧 유틸 함수
# ============================================================
IMG_EXTS = [".jpg",".jpeg",".png",".bmp",".webp",".JPG",".JPEG",".PNG",".BMP",".WEBP"]
EXCLUDE_RE = re.compile(r"_(dup|bg)\d+$", re.IGNORECASE)

def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img

def read_yolo_label(txt: Path):
    boxes, cls = [], []
    if not txt.exists():
        return boxes, cls

    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append([x, y, w, h])
        cls.append(cid)
    return boxes, cls

def write_yolo_label(out: Path, boxes, cls, allow_ids=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c, (x,y,w,h) in zip(cls, boxes):
            if allow_ids and c not in allow_ids:
                continue
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def sanitize_boxes(boxes):
    out = []
    for x,y,w,h in boxes:
        if w > 0 and h > 0:
            out.append([x,y,w,h])
    return out

def next_unique_name(stem: str, used: set, tag: str):
    i = 1
    while True:
        name = f"{stem}_{tag}{i:04d}"
        if name not in used:
            used.add(name)
            return name
        i += 1

def collect_background_images(img_dir: Path, lbl_dir: Path):
    lbl_stems = {p.stem for p in lbl_dir.rglob("*.txt")}
    out = []
    for ext in IMG_EXTS:
        for img in img_dir.rglob(f"*{ext}"):
            if img.stem not in lbl_stems:
                out.append(img)
    return out

def current_label_counts(lbl_dir: Path):
    """
    labels/train 아래 txt들을 읽어, 클래스별 박스(instance) 개수 카운트
    """
    cnt = Counter()
    for txt in lbl_dir.rglob("*.txt"):
        _, cls = read_yolo_label(txt)
        cnt.update(cls)
    return cnt


# ============================================================
# 🔧 메인 증강 로직 (설정값은 파라미터로 받음)
# ============================================================
def augment_dataset(
    base_dir: Path,
    TARGET: dict,
    SEED: int,
    BG_AUG_MULTIPLIER: int,
    MAX_USES_BASE: int,
    MAX_USES_BOOST: dict,
    MAX_PER_IMAGE_HARD: int,
    RECENT_COOLDOWN: int,
):
    random.seed(SEED)

    IMG_DIR = base_dir / "images" / "train"
    LBL_DIR = base_dir / "labels" / "train"
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    used_stems = {p.stem for p in IMG_DIR.rglob("*")}

    aug = build_augmentor()

    # ---------------- BG 증강 ----------------
    bg_list = collect_background_images(IMG_DIR, LBL_DIR)
    for bg in bg_list:
        img = load_image(bg)
        for _ in range(BG_AUG_MULTIPLIER):
            out_stem = next_unique_name(bg.stem, used_stems, "bg")
            out_img = IMG_DIR / f"{out_stem}{bg.suffix.lower()}"
            out_lbl = LBL_DIR / f"{out_stem}.txt"
            transformed = aug(image=img, bboxes=[], class_labels=[])
            cv2.imwrite(str(out_img), transformed["image"])
            out_lbl.touch()

    # ---------------- 라벨 이미지 후보 수집 ----------------
    candidates = []
    for lbl in LBL_DIR.rglob("*.txt"):
        if EXCLUDE_RE.search(lbl.stem):
            continue

        boxes, cls_list = read_yolo_label(lbl)
        if not cls_list:
            continue

        matched_img = None
        for ext in IMG_EXTS:
            cand = IMG_DIR / f"{lbl.stem}{ext}"
            if cand.exists():
                matched_img = cand
                break
        if not matched_img:
            continue

        candidates.append((matched_img, lbl, Counter(cls_list), lbl.stem))

    # ---------------- 현재 클래스 카운트 / deficit 계산 ----------------
    cur = current_label_counts(LBL_DIR)
    deficit = {c: max(0, TARGET[c] - cur.get(c, 0)) for c in TARGET}

    recent = deque(maxlen=RECENT_COOLDOWN)
    use_cnt = defaultdict(int)

    # ---------------- 메인 루프 ----------------
    for _ in range(200000):
        if all(d <= 0 for d in deficit.values()):
            break

        best = None
        best_score = 0
        for img, lbl, fcnt, stem in candidates:
            if img in recent:
                continue
            if use_cnt[img] >= MAX_PER_IMAGE_HARD:
                continue

            # 스코어 계산
            sc = sum(deficit[c] * fcnt.get(c, 0) for c in TARGET if deficit[c] > 0)
            if sc > best_score:
                best_score = sc
                best = (img, lbl, fcnt, stem)

        if not best:
            break

        img, lbl, fcnt, stem = best
        use_cnt[img] += 1
        recent.append(img)

        out_stem = next_unique_name(stem, used_stems, "dup")
        out_img = IMG_DIR / f"{out_stem}{img.suffix.lower()}"
        out_lbl = LBL_DIR / f"{out_stem}.txt"

        img0 = load_image(img)
        boxes0, cls0 = read_yolo_label(lbl)

        transformed = aug(image=img0, bboxes=boxes0, class_labels=cls0)
        boxes = sanitize_boxes(transformed["bboxes"])
        cls = transformed["class_labels"]

        cv2.imwrite(str(out_img), transformed["image"])
        write_yolo_label(out_lbl, boxes, cls)

        # deficit 업데이트
        added = Counter(cls)
        cur.update(added)
        for c in TARGET:
            deficit[c] = max(0, TARGET[c] - cur.get(c, 0))


# ============================================================
# ✅ Multiplier 기반: "클래스 상관없이 N배" 래퍼
#   - 현재 라벨 카운트를 먼저 센 뒤
#   - TARGET = 현재개수 * multiplier 로 자동 생성해서 augment_dataset 호출
# ============================================================
def build_target_by_multiplier(cur: Counter, num_classes: int, multiplier: float) -> dict:
    """
    TARGET은 augment_dataset이 기대하는 '최종 목표치(total)' dict.
    클래스 비율 유지하면서 전체 규모만 multiplier 배로 늘림.
    """
    target = {}
    for c in range(num_classes):
        base = int(cur.get(c, 0))
        target[c] = int(round(base * multiplier))
    return target


def augment_dataset_by_multiplier(
    base_dir: Path,
    multiplier: float,
    num_classes: int,
    *,
    SEED: int,
    BG_AUG_MULTIPLIER: int,
    MAX_USES_BASE: int,
    MAX_USES_BOOST: dict,
    MAX_PER_IMAGE_HARD: int,
    RECENT_COOLDOWN: int,
    verbose: bool = True,
):
    """
    base_dir 기준 labels/train을 스캔해 클래스별 instance 수를 센 뒤,
    TARGET을 multiplier 배로 자동 생성하여 augment_dataset을 실행한다.

    예)
      multiplier=3.0이면 class0: 100 -> 300, class1: 50 -> 150 ...
    """
    base_dir = Path(base_dir)

    IMG_DIR = base_dir / "images" / "train"
    LBL_DIR = base_dir / "labels" / "train"
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    cur = current_label_counts(LBL_DIR)
    target = build_target_by_multiplier(cur, num_classes=num_classes, multiplier=multiplier)

    if verbose:
        print(f"\n===== {base_dir.name} =====")
        print(f"[MULTIPLIER] x{multiplier}")
        print("[CURRENT -> TARGET]")
        for c in range(num_classes):
            print(f"  class {c}: {int(cur.get(c,0))} -> {int(target.get(c,0))}")

    augment_dataset(
        base_dir=base_dir,
        TARGET=target,
        SEED=SEED,
        BG_AUG_MULTIPLIER=BG_AUG_MULTIPLIER,
        MAX_USES_BASE=MAX_USES_BASE,
        MAX_USES_BOOST=MAX_USES_BOOST,
        MAX_PER_IMAGE_HARD=MAX_PER_IMAGE_HARD,
        RECENT_COOLDOWN=RECENT_COOLDOWN,
    )

    return target  # (원하면 실행파일에서 출력/로그용으로 사용)


