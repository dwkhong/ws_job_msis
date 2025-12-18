# src/dataset/copy_dataset.py

from pathlib import Path
import shutil

def copy_folders(
    src_list,
    dst_root,
    overwrite=True,
    verbose=True,
):
    """
    여러 폴더(src_list)를 목적지(dst_root) 아래로 복사하는 범용 함수.

    Parameters
    ----------
    src_list : list[Path or str]
        복사할 폴더 목록
    dst_root : Path or str
        복사될 루트 디렉토리
    overwrite : bool
        목적지 폴더가 이미 있어도 덮어쓸지 여부
    verbose : bool
        복사 과정 출력 여부
    """
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    dirs_exist_ok = overwrite

    for src in src_list:
        src = Path(src)

        if not src.exists():
            print(f"[WARN] 원본 폴더 없음: {src}")
            continue

        target = dst_root / src.name

        if verbose:
            print(f"📁 복사: {src} → {target}")

        shutil.copytree(src, target, dirs_exist_ok=dirs_exist_ok)

    print(f"\n[완료] 총 {len(src_list)}개 중 존재하는 폴더만 {dst_root} 아래로 복사했습니다.")