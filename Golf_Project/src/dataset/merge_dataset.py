# merge_utils.py
from pathlib import Path
import shutil, csv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}


# ================================================================
# 📂 디렉토리 준비
# ================================================================
def ensure_dirs(MERGE_ROOT: Path):
    for base in ["images", "labels"]:
        for split in ["train", "val"]:
            (MERGE_ROOT / base / split).mkdir(parents=True, exist_ok=True)
    (MERGE_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (MERGE_ROOT / "logs" / "classes_sources").mkdir(parents=True, exist_ok=True)


# ================================================================
# 🔧 파일 이동 관련
# ================================================================
def map_images(dir_path: Path):
    return {p.stem: p for p in dir_path.glob("*")
            if p.is_file() and p.suffix in IMG_EXTS}


def move_pair(MERGE_ROOT: Path, stem: str, img_path: Path | None,
              lbl_path: Path | None, split: str, prefix: str):

    new_stem = f"{prefix}__{stem}"

    # 이미지
    if img_path and img_path.exists():
        dst_img = MERGE_ROOT / "images" / split / f"{new_stem}{img_path.suffix.lower()}"
        shutil.move(str(img_path), dst_img)

    # 라벨
    if lbl_path and lbl_path.exists():
        dst_lbl = MERGE_ROOT / "labels" / split / f"{new_stem}.txt"
        shutil.move(str(lbl_path), dst_lbl)


def move_csvs_and_classes(MERGE_ROOT: Path, src_root: Path):
    prefix = src_root.name
    logs_dir = MERGE_ROOT / "logs"

    # CSV 이동
    for csv_file in src_root.glob("*.csv"):
        dst = logs_dir / f"{prefix}__{csv_file.name}"
        shutil.move(str(csv_file), dst)

    # classes 이동
    cls_file = src_root / "classes.txt"
    if cls_file.exists():
        dst = MERGE_ROOT / "logs" / "classes_sources" / f"{prefix}__classes.txt"
        shutil.move(str(cls_file), dst)


# ================================================================
# 🚀 병합 실행
# ================================================================
def merge_and_move(MERGE_ROOT: Path, src_root: Path):
    prefix = src_root.name

    for split in ["train", "val"]:
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split

        img_map = map_images(img_dir) if img_dir.exists() else {}
        lbl_map = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}

        stems = set(img_map) | set(lbl_map)

        for stem in stems:
            move_pair(
                MERGE_ROOT,
                stem,
                img_map.get(stem),
                lbl_map.get(stem),
                split,
                prefix
            )

    move_csvs_and_classes(MERGE_ROOT, src_root)

    shutil.rmtree(src_root, ignore_errors=True)


# ================================================================
# 🧠 클래스 처리
# ================================================================
def read_class_file(path: Path):
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def read_and_unify_class_names(MERGE_ROOT: Path):
    defaults = ["Divot", "Fixed_Divot", "Diseased_Grass", "Confused_Object",
                "Pole", "Sprinkler", "Drain", "Golf ball"]

    src_dir = MERGE_ROOT / "logs" / "classes_sources"
    files = sorted(src_dir.glob("*_classes.txt"))

    if not files:
        return defaults, "default"

    base_names = read_class_file(files[0])
    consistent = all(read_class_file(f) == base_names for f in files)

    if not consistent:
        warn = MERGE_ROOT / "logs" / "classes_mismatch.txt"
        with open(warn, "w", encoding="utf-8") as w:
            w.write("WARNING: classes.txt mismatch among sources\n")
            for f in files:
                w.write(f"- {f.name}\n")

    return base_names if base_names else defaults, ("ok" if consistent else "mismatch")


def write_classes_txt(MERGE_ROOT: Path, names):
    out = MERGE_ROOT / "classes.txt"
    out.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"✔ classes.txt → {out}")


def write_data_yaml(MERGE_ROOT: Path, names):
    yaml_path = MERGE_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names: [")
        f.write(", ".join(f'"{n}"' for n in names))
        f.write("]\n")
    print(f"✔ data.yaml → {yaml_path}")


# ================================================================
# 📊 요약
# ================================================================
def quick_summary(MERGE_ROOT: Path):
    for split in ["train", "val"]:
        n_img = len(list((MERGE_ROOT / "images" / split).glob("*")))
        n_lbl = len(list((MERGE_ROOT / "labels" / split).glob("*.txt")))
        print(f"[{split}] images={n_img}, labels={n_lbl}")


# ================================================================
# 🧾 로그 병합
# ================================================================
def merge_all_aug_logs(MERGE_ROOT: Path):
    logs_dir = MERGE_ROOT / "logs"
    out_path = logs_dir / "merged_aug_log.csv"

    candidate_logs = [p for p in logs_dir.glob("*.csv") if "aug" in p.name.lower()]
    if not candidate_logs:
        return

    header = None
    rows = []

    for csv_path in candidate_logs:
        try:
            with open(csv_path, encoding="utf-8") as f:
                r = csv.reader(f)
                h = next(r, None)
                if h is None:
                    continue

                if header is None:
                    header = h + ["source_csv"]
                elif h != header[:-1]:
                    continue

                for row in r:
                    rows.append(row + [csv_path.name])

        except:
            continue

    if header:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"✔ merged_aug_log.csv → {out_path}")
