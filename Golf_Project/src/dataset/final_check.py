# check_merge_utils.py
from pathlib import Path

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"
}


def check_split(base: Path, split: str):
    print(f"\n=== Checking {split} ===")

    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"❌ Missing {split} folder")
        return

    imgs = {p.stem for p in img_dir.rglob("*") if p.suffix in IMG_EXTS}
    lbls = {p.stem for p in lbl_dir.rglob("*.txt")}

    only_img = sorted(imgs - lbls)
    only_lbl = sorted(lbls - imgs)

    # bg는 OK
    only_img_nonbg = [s for s in only_img if "bg" not in s.lower()]

    # empty labels
    empty_lbl = []
    for lbl in lbl_dir.rglob("*.txt"):
        try:
            if lbl.stat().st_size == 0:
                empty_lbl.append(lbl.stem)
        except FileNotFoundError:
            continue

    print(f"Total images: {len(imgs)}, labels: {len(lbls)}")
    print(f"✅ Matched pairs: {len(imgs & lbls)}")

    if only_img_nonbg:
        print(f"⚠️ Images without label (non-bg) ({len(only_img_nonbg)}):")
        for s in only_img_nonbg[:10]:
            print(f"  - {s}")
        if len(only_img_nonbg) > 10:
            print(f"  ... +{len(only_img_nonbg)-10} more")

    if only_lbl:
        print(f"⚠️ Labels without image ({len(only_lbl)}):")
        for s in only_lbl[:10]:
            print(f"  - {s}")
        if len(only_lbl) > 10:
            print(f"  ... +{len(only_lbl)-10} more")

    if empty_lbl:
        print(f"⚠️ Empty label files ({len(empty_lbl)}):")
        for s in empty_lbl[:10]:
            print(f"  - {s}")
        if len(empty_lbl) > 10:
            print(f"  ... +{len(empty_lbl)-10} more")

    # Summary
    if not only_img_nonbg and not only_lbl and not empty_lbl:
        print("✅ All good — image/label pairs match perfectly (bg images allowed).")
