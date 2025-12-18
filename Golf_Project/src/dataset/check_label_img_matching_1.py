# src/dataset/check_label_img_matching_1.py

from pathlib import Path

def check_label_matching(
    base_dir: Path,
    image_ext: str = "*.jpg",
    label_ext: str = "*.txt",
    ignore_labels=("classes.txt",),
    verbose=True
):
    """
    이미지-라벨 매칭 여부 확인 유틸리티.
    하위 폴더별로 jpg ↔ txt 파일이 제대로 짝이 맞는지 검사한다.
    """

    jpg_only_total = []
    txt_only_total = []

    subdirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])

    for subdir in subdirs:
        jpg_stems = {f.stem for f in subdir.glob(image_ext)}
        txt_stems = {f.stem for f in subdir.glob(label_ext)
                     if f.name not in ignore_labels}

        jpg_only = sorted(jpg_stems - txt_stems)
        txt_only = sorted(txt_stems - jpg_stems)

        if verbose and (jpg_only or txt_only):
            print(f"\n📂 폴더: {subdir.name}")

            if jpg_only:
                print("  📸 JPG만 있고 대응되는 TXT 없는 파일:")
                for stem in jpg_only:
                    print(f"    {stem}.jpg")

            if txt_only:
                print("  📝 TXT만 있고 대응되는 JPG 없는 파일:")
                for stem in txt_only:
                    print(f"    {stem}.txt")

        # 절대 경로 수집
        jpg_only_total.extend(subdir / f"{stem}.jpg" for stem in jpg_only)
        txt_only_total.extend(subdir / f"{stem}.txt" for stem in txt_only)

    # 요약 출력
    print("\n==============================")
    print(f"📸 JPG만 있는 파일: {len(jpg_only_total)}개")
    print(f"📝 TXT만 있는 파일: {len(txt_only_total)}개")
    print("==============================")

    return jpg_only_total, txt_only_total