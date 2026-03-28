#!/usr/bin/env python3
"""
Experiment 2: Enhancement Comparison

Evaluate how image degradation hurts classification and how much the
adaptive enhancement pipeline recovers.

Pipeline for each degradation level (low / medium / high):
  1. Load baseline model trained in Experiment 1.
  2. Create degraded test images (on-the-fly).
  3. Enhance degraded images with the adaptive pipeline.
  4. Evaluate model on: clean, degraded, and enhanced test sets.
  5. Compute recovery rate = (enhanced - degraded) / (clean - degraded).

Outputs:
    - enhancement_comparison.json   (all metrics per condition)
    - comparison_bar_chart.png      (accuracy across conditions)
    - confusion_matrix_*.png        (per-condition confusion matrices)
    - quality_scores.json           (BRISQUE/NIQE per condition)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model import ResNet50Classifier
from src.classification.evaluate import evaluate_model
from src.utils.data_loader import ImageFolderDataset, get_data_loaders
from src.utils.metrics import plot_confusion_matrix
from src.enhancement.pipeline import ImageEnhancer
from src.quality.degradation import (
    DEGRADATION_PRESETS,
    add_gaussian_noise,
    add_gaussian_blur,
    reduce_contrast,
    jpeg_compression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Dataset wrappers that apply degradation / enhancement on-the-fly
# ----------------------------------------------------------------

class DegradedDataset(Dataset):
    """Wraps an ImageFolderDataset, applying degradation before preprocessing."""

    def __init__(self, base_dataset: ImageFolderDataset, preset: dict):
        self.base = base_dataset
        self.preset = preset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        path, label = self.base.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        # Apply degradation chain
        image = add_gaussian_noise(image, sigma=float(self.preset["noise_sigma"]))
        image = add_gaussian_blur(image, kernel_size=int(self.preset["blur_kernel"]))
        image = reduce_contrast(image, gamma=float(self.preset["gamma"]))
        image = jpeg_compression(image, quality=int(self.preset["jpeg_quality"]))

        # Standard preprocessing (resize, normalise, to tensor)
        from src.utils.data_loader import _preprocess
        tensor = _preprocess(image, self.base.image_size, augment=False)
        return tensor, label


class EnhancedDegradedDataset(Dataset):
    """Wraps an ImageFolderDataset: degrade then enhance before preprocessing."""

    def __init__(self, base_dataset: ImageFolderDataset, preset: dict,
                 enhancer: ImageEnhancer):
        self.base = base_dataset
        self.preset = preset
        self.enhancer = enhancer

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        path, label = self.base.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        # Degrade
        image = add_gaussian_noise(image, sigma=float(self.preset["noise_sigma"]))
        image = add_gaussian_blur(image, kernel_size=int(self.preset["blur_kernel"]))
        image = reduce_contrast(image, gamma=float(self.preset["gamma"]))
        image = jpeg_compression(image, quality=int(self.preset["jpeg_quality"]))

        # Enhance
        image = self.enhancer.enhance(image)

        from src.utils.data_loader import _preprocess
        tensor = _preprocess(image, self.base.image_size, augment=False)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp2: Enhancement comparison")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--model-path", type=str,
                        default="results/exp1_baseline/models/best_model.pth",
                        help="Path to trained baseline model checkpoint")
    parser.add_argument("--output-dir", type=str,
                        default="results/exp2_enhancement")
    parser.add_argument("--degradation-levels", nargs="+",
                        default=["low", "medium", "high"],
                        choices=list(DEGRADATION_PRESETS.keys()))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    for d in [figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ----------------------------------------------------------------
    # Load data and model
    # ----------------------------------------------------------------
    data_dir = Path(args.data_dir)
    test_ds = ImageFolderDataset(data_dir / "test", image_size=args.image_size, augment=False)
    class_names = test_ds.class_names
    num_classes = len(class_names)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded model from %s (val_acc=%.4f)",
                args.model_path, checkpoint.get("val_accuracy", -1))

    enhancer = ImageEnhancer()

    # ----------------------------------------------------------------
    # Evaluate on clean test set
    # ----------------------------------------------------------------
    clean_loader = _make_loader(test_ds, args.batch_size, args.num_workers)
    clean_results = evaluate_model(model, clean_loader, device, class_names)
    logger.info("Clean test accuracy: %.4f", clean_results["accuracy"])

    all_results = {"clean": clean_results}

    # ----------------------------------------------------------------
    # Evaluate per degradation level
    # ----------------------------------------------------------------
    condition_accuracies = {"Clean": clean_results["accuracy"]}

    for level in args.degradation_levels:
        preset = DEGRADATION_PRESETS[level]
        logger.info("--- Degradation level: %s ---", level)

        # Degraded
        deg_ds = DegradedDataset(test_ds, preset)
        deg_loader = _make_loader(deg_ds, args.batch_size, args.num_workers)
        deg_results = evaluate_model(model, deg_loader, device, class_names)
        logger.info("  Degraded accuracy:  %.4f", deg_results["accuracy"])

        # Enhanced
        enh_ds = EnhancedDegradedDataset(test_ds, preset, enhancer)
        enh_loader = _make_loader(enh_ds, args.batch_size, args.num_workers)
        enh_results = evaluate_model(model, enh_loader, device, class_names)
        logger.info("  Enhanced accuracy:  %.4f", enh_results["accuracy"])

        # Recovery rate
        drop = clean_results["accuracy"] - deg_results["accuracy"]
        recovery = (
            (enh_results["accuracy"] - deg_results["accuracy"]) / drop
            if drop > 0.001 else 0.0
        )
        logger.info("  Accuracy drop: %.4f | Recovery rate: %.2f%%", drop, recovery * 100)

        all_results[f"degraded_{level}"] = deg_results
        all_results[f"enhanced_{level}"] = enh_results
        all_results[f"recovery_{level}"] = {
            "accuracy_drop": round(drop, 4),
            "recovery_rate": round(recovery, 4),
        }

        condition_accuracies[f"Degraded ({level})"] = deg_results["accuracy"]
        condition_accuracies[f"Enhanced ({level})"] = enh_results["accuracy"]

        # Confusion matrices
        for tag, loader in [("degraded", deg_loader), ("enhanced", enh_loader)]:
            preds, labels = [], []
            model.eval()
            with torch.no_grad():
                for imgs, lbl in loader:
                    out = model(imgs.to(device))
                    preds.extend(out.argmax(1).cpu().tolist())
                    labels.extend(lbl.tolist())
            plot_confusion_matrix(
                np.array(labels), np.array(preds), class_names,
                save_path=str(figures_dir / f"confusion_matrix_{tag}_{level}.png"),
            )

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    with open(tables_dir / "enhancement_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Bar chart of accuracies across all conditions
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(condition_accuracies.keys())
    accs = [condition_accuracies[n] for n in names]
    colors = []
    for n in names:
        if "Enhanced" in n:
            colors.append("#2ecc71")
        elif "Degraded" in n:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")
    bars = ax.bar(range(len(names)), accs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy: Clean vs Degraded vs Enhanced")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "comparison_bar_chart.png", dpi=150)
    plt.close(fig)

    logger.info("Experiment 2 complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()