"""Evaluation and inference utilities for the GI disease classifier.

Computes per-class metrics (precision, recall, F1), confusion matrices,
and supports batch inference on a list of image paths.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """Compute comprehensive classification metrics on a dataset.

    Collects all predictions in a single pass then derives accuracy,
    per-class precision / recall / F1, and a confusion matrix.

    Args:
        model: Trained classification model.
        dataloader: Evaluation dataloader yielding (images, labels).
        device: Torch device for computation.
        class_names: Optional list of human-readable class names.
            If provided, per-class metrics use these as keys.

    Returns:
        Dict with:
        - ``accuracy``: Overall accuracy.
        - ``per_class``: Dict mapping class name/index to
          ``{precision, recall, f1, support}``.
        - ``confusion_matrix``: 2-D list (row=true, col=pred).

    Example:
        >>> results = evaluate_model(model, test_loader, device, class_names=["polyp", "ulcer"])
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
        >>> for cls, m in results['per_class'].items():
        ...     print(f"  {cls}: F1={m['f1']:.3f}")
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    num_classes = max(all_labels_arr.max(), all_preds_arr.max()) + 1

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels_arr, all_preds_arr):
        cm[t, p] += 1

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        name = class_names[i] if class_names and i < len(class_names) else str(i)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    accuracy = float((all_preds_arr == all_labels_arr).mean())

    logger.info("Evaluation — accuracy: %.4f, samples: %d", accuracy, len(all_labels))
    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def generate_predictions(
    model: nn.Module,
    image_paths: list[str | Path],
    device: torch.device,
    image_size: int = 224,
    class_names: list[str] | None = None,
) -> list[dict]:
    """Run inference on a list of image file paths.

    Each image is loaded, resized, normalised with ImageNet stats, and
    classified. Returns per-image predictions with confidence scores.

    Args:
        model: Trained classification model.
        image_paths: List of paths to input images.
        device: Torch device.
        image_size: Spatial size to resize images to.
        class_names: Optional class name list for labelling predictions.

    Returns:
        List of dicts, one per image, with keys ``path``,
        ``predicted_class``, ``confidence``, and ``probabilities``.

    Example:
        >>> from pathlib import Path
        >>> paths = list(Path("data/test").glob("*.png"))
        >>> preds = generate_predictions(model, paths, device)
        >>> for p in preds[:3]:
        ...     print(f"{p['path']}: {p['predicted_class']} ({p['confidence']:.2%})")
    """
    model.eval()
    # ImageNet normalisation constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    predictions: list[dict] = []

    for img_path in image_paths:
        img_path = Path(img_path)
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read %s — skipping", img_path)
            continue

        # Preprocess: BGR->RGB, resize, normalise, to tensor
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size))
        normalised = (resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        pred_label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)

        predictions.append({
            "path": str(img_path),
            "predicted_class": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                (class_names[i] if class_names and i < len(class_names) else str(i)): round(float(probs[i]), 4)
                for i in range(len(probs))
            },
        })

    logger.info("Generated predictions for %d / %d images", len(predictions), len(image_paths))
    return predictions


def save_results(
    predictions: list[dict],
    output_path: str | Path,
) -> None:
    """Save prediction results to a JSON file.

    Args:
        predictions: List of prediction dicts from ``generate_predictions``.
        output_path: Destination file path (``.json``).

    Example:
        >>> save_results(preds, "results/tables/test_predictions.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info("Saved %d predictions to %s", len(predictions), output_path)