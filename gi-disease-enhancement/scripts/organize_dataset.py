#!/usr/bin/env python3
"""
Organize Kvasir Dataset v2 into train/val/test splits.

Splits: 70% train, 10% val, 20% test
Maintains class folder structure in each split.
"""

import shutil
import sys
import yaml
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Project root
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "kvasir-dataset-v2"
SPLITS_DIR = ROOT / "data" / "splits"

CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def get_image_files(class_dir: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_class(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Split file list into train/val/test using stratified proportions."""
    # First split: train vs (val + test)
    train_files, remaining = train_test_split(
        files,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
    )
    # Second split: val vs test from remaining
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_files, test_files = train_test_split(
        remaining,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_SEED,
    )
    return train_files, val_files, test_files


def copy_files(files: list[Path], dest_dir: Path, desc: str) -> None:
    """Copy files to destination with progress bar."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(files, desc=desc, leave=False):
        shutil.copy2(f, dest_dir / f.name)


def main() -> None:
    # Validate raw dataset exists
    if not RAW_DIR.exists():
        print(f"Error: Raw dataset not found at {RAW_DIR}")
        sys.exit(1)

    # Validate all classes exist
    missing = [c for c in CLASSES if not (RAW_DIR / c).is_dir()]
    if missing:
        print(f"Error: Missing class directories: {missing}")
        sys.exit(1)

    print(f"Source:  {RAW_DIR}")
    print(f"Output:  {SPLITS_DIR}")
    print(f"Split:   train={TRAIN_RATIO:.0%} / val={VAL_RATIO:.0%} / test={TEST_RATIO:.0%}")
    print(f"Seed:    {RANDOM_SEED}")
    print()

    stats = {
        "seed": RANDOM_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "classes": {},
        "totals": {"train": 0, "val": 0, "test": 0, "total": 0},
    }

    for cls in CLASSES:
        class_dir = RAW_DIR / cls
        files = get_image_files(class_dir)

        if not files:
            print(f"Warning: No images found in {class_dir}")
            continue

        train_files, val_files, test_files = split_class(files)

        # Copy to split directories
        copy_files(train_files, SPLITS_DIR / "train" / cls, f"{cls}/train")
        copy_files(val_files, SPLITS_DIR / "val" / cls, f"{cls}/val")
        copy_files(test_files, SPLITS_DIR / "test" / cls, f"{cls}/test")

        # Record stats
        counts = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
        stats["classes"][cls] = counts
        stats["totals"]["train"] += counts["train"]
        stats["totals"]["val"] += counts["val"]
        stats["totals"]["test"] += counts["test"]
        stats["totals"]["total"] += counts["total"]

        print(f"  {cls:<28s}  total={counts['total']:4d}  "
              f"train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")

    # Summary
    t = stats["totals"]
    print()
    print(f"  {'TOTAL':<28s}  total={t['total']:4d}  "
          f"train={t['train']:4d}  val={t['val']:4d}  test={t['test']:4d}")

    # Save stats
    stats_path = SPLITS_DIR / "dataset_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()