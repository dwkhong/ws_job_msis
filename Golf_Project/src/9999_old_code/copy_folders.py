"""
Small utility to copy multiple source folders into a single destination.

Example:
    python copy_folders.py --dst /data/merged \
        --src /data/day1/good --src /data/day2/good \
        --overwrite --skip-missing
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Iterable, Sequence


def copy_folders(
    sources: Sequence[Path],
    destination: Path,
    *,
    overwrite: bool = False,
    skip_missing: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Copy each folder in `sources` into `destination` while keeping the original
    folder name (destination/<source.name>).
    """
    destination.mkdir(parents=True, exist_ok=True)

    for src in sources:
        if not src.exists():
            if skip_missing:
                print(f"[skip] missing: {src}")
                continue
            raise FileNotFoundError(f"Source folder not found: {src}")
        if not src.is_dir():
            raise NotADirectoryError(f"Source is not a directory: {src}")

        target = destination / src.name
        if target.exists() and not overwrite:
            print(f"[skip] exists: {target}")
            continue

        prefix = "[dry-run] " if dry_run else ""
        print(f"{prefix}copying {src} -> {target}")
        if dry_run:
            continue

        shutil.copytree(src, target, dirs_exist_ok=overwrite)

    print(f"\nDone. Copied {len(sources)} folder(s) into {destination}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy multiple folders into a destination folder."
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination folder to create/use.",
    )
    parser.add_argument(
        "--src",
        action="append",
        type=Path,
        help="Source folder to copy (can be given multiple times).",
    )
    parser.add_argument(
        "--src-file",
        type=Path,
        help="Text file with one source folder per line (blank lines ignored).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination subfolders.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip sources that do not exist instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing anything.",
    )
    return parser.parse_args(argv)


def load_sources(args: argparse.Namespace) -> list[Path]:
    sources: list[Path] = []
    if args.src:
        sources.extend(args.src)
    if args.src_file:
        lines = args.src_file.read_text(encoding="utf-8").splitlines()
        sources.extend(Path(line.strip()) for line in lines if line.strip())

    if not sources:
        raise SystemExit("No sources provided. Use --src and/or --src-file.")
    return sources


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    sources = load_sources(args)
    copy_folders(
        sources,
        args.dst,
        overwrite=args.overwrite,
        skip_missing=args.skip_missing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
