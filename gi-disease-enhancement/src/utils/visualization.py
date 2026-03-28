"""Visualization utilities for the GI disease enhancement project.

Provides publication-ready plots for image comparisons, training
progress, and quality-score distributions using matplotlib and seaborn.
"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent style across all figures
sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_image_comparison(
    original: np.ndarray,
    degraded: np.ndarray,
    enhanced: np.ndarray,
    titles: tuple[str, str, str] = ("Original", "Degraded", "Enhanced"),
    save_path: str | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> plt.Figure:
    """Show original, degraded, and enhanced images side by side.

    All images are expected as BGR uint8 arrays (OpenCV convention) and
    are converted to RGB for display.

    Args:
        original: Original BGR image.
        degraded: Degraded BGR image.
        enhanced: Enhanced BGR image.
        titles: Subplot titles.
        save_path: If provided, save the figure to this path.
        figsize: Figure size in inches.

    Returns:
        The matplotlib ``Figure`` object.

    Example:
        >>> import cv2
        >>> orig = cv2.imread("data/raw/img001.png")
        >>> deg  = cv2.imread("data/degraded/medium/img001.png")
        >>> enh  = cv2.imread("data/enhanced/medium/img001.png")
        >>> fig = plot_image_comparison(orig, deg, enh)
    """
    images = [original, degraded, enhanced]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Image comparison saved to %s", save_path)

    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot training and validation loss / accuracy curves.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc`` — each a list of per-epoch floats.
        save_path: If provided, save the figure to this path.
        figsize: Figure size.

    Returns:
        The matplotlib ``Figure`` object.

    Example:
        >>> history = {
        ...     "train_loss": [0.9, 0.5, 0.3],
        ...     "val_loss":   [1.0, 0.6, 0.4],
        ...     "train_acc":  [0.6, 0.8, 0.9],
        ...     "val_acc":    [0.5, 0.7, 0.85],
        ... }
        >>> fig = plot_training_curves(history, save_path="results/figures/curves.png")
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)

    # Loss
    ax_loss.plot(epochs, history["train_loss"], "o-", label="Train", markersize=4)
    ax_loss.plot(epochs, history["val_loss"], "s-", label="Validation", markersize=4)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend()

    # Accuracy
    ax_acc.plot(epochs, history["train_acc"], "o-", label="Train", markersize=4)
    ax_acc.plot(epochs, history["val_acc"], "s-", label="Validation", markersize=4)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy Curves")
    ax_acc.legend()

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to %s", save_path)

    return fig


def plot_quality_distribution(
    quality_scores: dict[str, list[float]],
    labels: list[str] | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot distributions of quality scores across image groups.

    Useful for comparing quality metrics (e.g. BRISQUE) between
    original, degraded, and enhanced image sets.

    Args:
        quality_scores: Dict mapping group name to a list of scalar
            quality scores.  Example::

                {"Original": [12.3, 14.1, ...],
                 "Degraded": [45.2, 51.0, ...],
                 "Enhanced": [18.7, 20.3, ...]}

        labels: Optional x-axis group labels (defaults to dict keys).
        save_path: If provided, save the figure to this path.
        figsize: Figure size.

    Returns:
        The matplotlib ``Figure`` object.

    Example:
        >>> scores = {
        ...     "Original": [12, 14, 11, 15],
        ...     "Degraded": [45, 51, 48, 55],
        ...     "Enhanced": [18, 20, 17, 22],
        ... }
        >>> fig = plot_quality_distribution(scores)
    """
    group_names = labels if labels else list(quality_scores.keys())

    fig, (ax_box, ax_kde) = plt.subplots(1, 2, figsize=figsize)

    # Box plot
    box_data = [quality_scores[g] for g in group_names]
    bp = ax_box.boxplot(box_data, labels=group_names, patch_artist=True)
    palette = sns.color_palette("Set2", len(group_names))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    ax_box.set_ylabel("Quality Score")
    ax_box.set_title("Quality Score Distribution")

    # KDE plot
    for name, color in zip(group_names, palette):
        values = quality_scores[name]
        if len(values) > 1:
            sns.kdeplot(values, ax=ax_kde, label=name, color=color, fill=True, alpha=0.3)
        else:
            ax_kde.axvline(values[0], label=name, color=color)
    ax_kde.set_xlabel("Quality Score")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title("Quality Score Density")
    ax_kde.legend()

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Quality distribution saved to %s", save_path)

    return fig