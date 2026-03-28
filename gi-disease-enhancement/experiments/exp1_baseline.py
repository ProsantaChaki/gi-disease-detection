#!/usr/bin/env python3
"""
Experiment 1: Baseline Classification

Train ResNet-50 on clean GI endoscopy images and evaluate on the clean
test set. This establishes the upper-bound accuracy that degradation
experiments will be measured against.

Outputs:
    - best_model.pth / final_model.pth  (model checkpoints)
    - training_curves.png               (loss & accuracy over epochs)
    - confusion_matrix.png              (test-set confusion matrix)
    - baseline_results.json             (per-class metrics)
    - TensorBoard logs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model import ResNet50Classifier, freeze_backbone
from src.classification.train import train
from src.classification.evaluate import evaluate_model
from src.utils.data_loader import get_data_loaders
from src.utils.metrics import plot_confusion_matrix
from src.utils.visualization import plot_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp1: Baseline classification")
    parser.add_argument("--data-dir", type=str, default="data/splits",
                        help="Path to split dataset (train/val/test subdirs)")
    parser.add_argument("--output-dir", type=str, default="results/exp1_baseline",
                        help="Directory for checkpoints, figures, and metrics")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs to train with frozen backbone before unfreezing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    for d in [models_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    logger.info("Loading data from %s", data_dir)
    loaders = get_data_loaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    class_names = loaders["class_names"]
    num_classes = len(class_names)
    logger.info("Classes (%d): %s", num_classes, class_names)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = ResNet50Classifier(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train head only (backbone frozen)
    if args.freeze_epochs > 0:
        logger.info("Phase 1: Training head only for %d epochs", args.freeze_epochs)
        freeze_backbone(model, freeze=True)
        optimizer_head = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 10,
        )
        history_head = train(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            criterion=criterion,
            optimizer=optimizer_head,
            device=device,
            num_epochs=args.freeze_epochs,
            output_dir=models_dir,
            patience=args.freeze_epochs,  # no early stop in phase 1
        )

    # Phase 2: fine-tune full model
    logger.info("Phase 2: Fine-tuning full model for up to %d epochs", args.epochs)
    freeze_backbone(model, freeze=False)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    history = train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=models_dir,
        scheduler=scheduler,
        patience=args.patience,
    )

    # ----------------------------------------------------------------
    # Evaluation on test set
    # ----------------------------------------------------------------
    logger.info("Evaluating best model on test set")
    checkpoint = torch.load(models_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_model(model, loaders["test"], device, class_names=class_names)

    logger.info("Test accuracy: %.4f", results["accuracy"])
    for cls, m in results["per_class"].items():
        logger.info("  %s: P=%.3f  R=%.3f  F1=%.3f  (n=%d)",
                     cls, m["precision"], m["recall"], m["f1"], m["support"])

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    # Metrics JSON
    results_path = tables_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics to %s", results_path)

    # Training curves
    plot_training_curves(history, save_path=str(figures_dir / "training_curves.png"))

    # Confusion matrix
    import numpy as np
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loaders["test"]:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    plot_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        class_names=class_names,
        save_path=str(figures_dir / "confusion_matrix.png"),
    )

    logger.info("Experiment 1 complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()