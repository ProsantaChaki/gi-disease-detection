"""Classification metrics and confusion-matrix plotting.

All metric functions operate on plain Python / numpy arrays so they
work independently of the training framework.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall classification accuracy.

    Args:
        y_true: Ground-truth integer labels, shape ``(N,)``.
        y_pred: Predicted integer labels, shape ``(N,)``.

    Returns:
        Accuracy in [0, 1].

    Example:
        >>> calculate_accuracy(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1]))
        0.75
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Compute precision (macro or micro averaged).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"macro"`` (per-class mean) or ``"micro"`` (global TP/FP).

    Returns:
        Precision in [0, 1].

    Example:
        >>> precision(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), average="macro")
        0.75
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp = (y_true == y_pred).sum()
        return float(tp / len(y_pred)) if len(y_pred) > 0 else 0.0

    precisions = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(precisions))


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Compute recall (macro or micro averaged).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"macro"`` (per-class mean) or ``"micro"`` (global TP/FN).

    Returns:
        Recall in [0, 1].

    Example:
        >>> recall(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), average="macro")
        0.75
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp = (y_true == y_pred).sum()
        return float(tp / len(y_true)) if len(y_true) > 0 else 0.0

    recalls = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"macro"`` or ``"micro"``.

    Returns:
        F1 score in [0, 1].

    Example:
        >>> f1_score(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), average="macro")
        0.6666666666666666
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1, and support for each class.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Human-readable names for each class index.

    Returns:
        Dict mapping class name to ``{precision, recall, f1, support}``.

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 2, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 2, 0])
        >>> m = calculate_per_class_metrics(y_true, y_pred, ["A", "B", "C"])
        >>> m["A"]["recall"]
        0.5
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    results: dict[str, dict[str, float]] = {}

    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        name = class_names[c] if class_names and c < len(class_names) else str(c)
        results[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": support,
        }

    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot a confusion matrix heatmap with annotations.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class names for axis labels.
        save_path: If provided, save the figure to this path.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib ``Figure`` object.

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 2, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 2, 0])
        >>> fig = plot_confusion_matrix(y_true, y_pred, ["A", "B", "C"])
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig