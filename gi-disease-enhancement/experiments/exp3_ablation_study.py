#!/usr/bin/env python3
"""
Experiment 3: Ablation Study

Isolate the contribution of each enhancement component by testing
every combination on medium-degraded images.

Components tested:
    - CLAHE only
    - Denoise only
    - Sharpen only
    - CLAHE + Denoise
    - CLAHE + Sharpen
    - Denoise + Sharpen
    - Full pipeline (CLAHE + Denoise + Sharpen)

Reference conditions:
    - Clean (no degradation)
    - Degraded (no enhancement)

Outputs:
    - ablation_results.json      (per-condition metrics)
    - ablation_bar_chart.png     (accuracy comparison)
    - ablation_table.png         (publication-ready table figure)
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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model import ResNet50Classifier
from src.classification.evaluate import evaluate_model
from src.utils.data_loader import ImageFolderDataset, _preprocess
from src.enhancement.clahe import adaptive_clahe
from src.enhancement.denoise import adaptive_denoise
from src.enhancement.sharpen import adaptive_sharpen
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
# Ablation configurations
# ----------------------------------------------------------------

# Each entry maps a condition name to the set of enhancement stages to apply.
# Stages run in order: denoise -> clahe -> sharpen (matching the full pipeline).
ABLATION_CONDITIONS = {
    "No enhancement":           {"denoise": False, "clahe": False, "sharpen": False},
    "CLAHE only":               {"denoise": False, "clahe": True,  "sharpen": False},
    "Denoise only":             {"denoise": True,  "clahe": False, "sharpen": False},
    "Sharpen only":             {"denoise": False, "clahe": False, "sharpen": True},
    "CLAHE + Denoise":          {"denoise": True,  "clahe": True,  "sharpen": False},
    "CLAHE + Sharpen":          {"denoise": False, "clahe": True,  "sharpen": True},
    "Denoise + Sharpen":        {"denoise": True,  "clahe": False, "sharpen": True},
    "Full pipeline":            {"denoise": True,  "clahe": True,  "sharpen": True},
}


class AblationDataset(Dataset):
    """Apply degradation then selective enhancement stages."""

    def __init__(
        self,
        base_dataset: ImageFolderDataset,
        preset: dict,
        stages: dict[str, bool],
    ):
        self.base = base_dataset
        self.preset = preset
        self.stages = stages
        self.enhancer = ImageEnhancer()

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

        # Selectively enhance
        quality = self.enhancer.assess_quality(image)

        if self.stages.get("denoise") and quality["noise_level"] >= self.enhancer.denoise_threshold:
            image = adaptive_denoise(image, quality["noise_level"])

        if self.stages.get("clahe") and quality["contrast_score"] < self.enhancer.contrast_threshold:
            image = adaptive_clahe(image, quality["contrast_score"])

        if self.stages.get("sharpen") and quality["blur_score"] >= self.enhancer.blur_threshold:
            image = adaptive_sharpen(image, quality["blur_score"])

        tensor = _preprocess(image, self.base.image_size, augment=False)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp3: Ablation study")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--model-path", type=str,
                        default="results/exp1_baseline/models/best_model.pth")
    parser.add_argument("--output-dir", type=str,
                        default="results/exp3_ablation")
    parser.add_argument("--degradation-level", type=str, default="medium",
                        choices=list(DEGRADATION_PRESETS.keys()))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


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
    # Load model and data
    # ----------------------------------------------------------------
    data_dir = Path(args.data_dir)
    test_ds = ImageFolderDataset(data_dir / "test", image_size=args.image_size, augment=False)
    class_names = test_ds.class_names
    num_classes = len(class_names)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded model from %s", args.model_path)

    preset = DEGRADATION_PRESETS[args.degradation_level]
    logger.info("Degradation level: %s — %s", args.degradation_level, preset)

    # ----------------------------------------------------------------
    # Evaluate clean baseline
    # ----------------------------------------------------------------
    clean_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    clean_results = evaluate_model(model, clean_loader, device, class_names)
    clean_acc = clean_results["accuracy"]
    logger.info("Clean accuracy: %.4f", clean_acc)

    # ----------------------------------------------------------------
    # Run ablation conditions
    # ----------------------------------------------------------------
    all_results = {"clean": {"accuracy": clean_acc, "per_class": clean_results["per_class"]}}
    condition_names = []
    condition_accs = []

    for cond_name, stages in ABLATION_CONDITIONS.items():
        logger.info("--- %s ---", cond_name)

        ds = AblationDataset(test_ds, preset, stages)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        results = evaluate_model(model, loader, device, class_names)

        acc = results["accuracy"]
        drop = clean_acc - acc
        recovery = (
            (acc - all_results.get("No enhancement", {}).get("accuracy", acc))
            / (clean_acc - all_results.get("No enhancement", {}).get("accuracy", acc))
            if "No enhancement" in all_results
            and (clean_acc - all_results["No enhancement"]["accuracy"]) > 0.001
            else 0.0
        )

        logger.info("  Accuracy: %.4f | Drop: %.4f | Recovery: %.2f%%",
                     acc, drop, recovery * 100)

        all_results[cond_name] = {
            "accuracy": acc,
            "accuracy_drop": round(drop, 4),
            "recovery_rate": round(recovery, 4),
            "stages": stages,
            "per_class": results["per_class"],
        }
        condition_names.append(cond_name)
        condition_accs.append(acc)

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    with open(tables_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved ablation results to %s", tables_dir / "ablation_results.json")

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(condition_names))
    colors = ["#e74c3c" if "No enhancement" in n else "#2ecc71" if "Full" in n else "#3498db"
              for n in condition_names]

    bars = ax.bar(x, condition_accs, color=colors)
    ax.axhline(y=clean_acc, color="gray", linestyle="--", linewidth=1, label=f"Clean ({clean_acc:.3f})")

    ax.set_xticks(list(x))
    ax.set_xticklabels(condition_names, rotation=40, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Ablation Study — {args.degradation_level.title()} Degradation")
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bar, acc in zip(bars, condition_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(figures_dir / "ablation_bar_chart.png", dpi=150)
    plt.close(fig)

    # Summary table as figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table_data = []
    headers = ["Condition", "Accuracy", "Drop", "Recovery %"]
    for name in condition_names:
        r = all_results[name]
        table_data.append([
            name,
            f"{r['accuracy']:.4f}",
            f"{r['accuracy_drop']:.4f}",
            f"{r['recovery_rate'] * 100:.1f}%",
        ])
    table = ax.table(cellText=table_data, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    fig.tight_layout()
    fig.savefig(figures_dir / "ablation_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Experiment 3 complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()