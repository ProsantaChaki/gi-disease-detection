"""Training loop for GI disease classification.

Provides per-epoch train/validate functions and a full training loop
with checkpointing, early stopping, and TensorBoard logging of
loss/accuracy curves.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch.

    Args:
        model: The classification model.
        dataloader: Training dataloader yielding (images, labels).
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Optimiser instance.
        device: Torch device for computation.

    Returns:
        Dict with ``loss`` (mean) and ``accuracy`` for the epoch.

    Example:
        >>> metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        >>> print(f"Train loss: {metrics['loss']:.4f}, acc: {metrics['accuracy']:.2%}")
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation pass (no gradient computation).

    Args:
        model: The classification model.
        dataloader: Validation dataloader yielding (images, labels).
        criterion: Loss function.
        device: Torch device for computation.

    Returns:
        Dict with ``loss`` and ``accuracy``.

    Example:
        >>> metrics = validate(model, val_loader, criterion, device)
        >>> print(f"Val loss: {metrics['loss']:.4f}, acc: {metrics['accuracy']:.2%}")
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def _save_training_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    """Plot and save loss and accuracy curves to *output_dir*."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="Train")
    ax_acc.plot(epochs, history["val_acc"], label="Val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training & Validation Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved training curves to %s", output_dir / "training_curves.png")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    output_dir: str | Path = "results/models",
    scheduler: LRScheduler | None = None,
    patience: int = 10,
) -> dict[str, list[float]]:
    """Full training loop with checkpointing, early stopping, and logging.

    Saves the best model (by validation accuracy), the final model,
    training curves as a PNG, and per-epoch metrics to TensorBoard.

    Args:
        model: The classification model (already on *device*).
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        criterion: Loss function.
        optimizer: Optimiser.
        device: Torch device.
        num_epochs: Maximum number of epochs.
        output_dir: Directory for checkpoints and figures.
        scheduler: Optional learning-rate scheduler (stepped per epoch).
        patience: Early-stopping patience in epochs.

    Returns:
        History dict with keys ``train_loss``, ``train_acc``,
        ``val_loss``, ``val_acc`` — each a list of per-epoch floats.

    Example:
        >>> history = train(
        ...     model, train_loader, val_loader, criterion, optimizer,
        ...     device=torch.device("cuda"), num_epochs=30,
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        # TensorBoard
        writer.add_scalars("loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, epoch)
        writer.add_scalars("accuracy", {
            "train": train_metrics["accuracy"],
            "val": val_metrics["accuracy"],
        }, epoch)

        logger.info(
            "Epoch %d/%d — train_loss: %.4f  train_acc: %.4f  "
            "val_loss: %.4f  val_acc: %.4f",
            epoch, num_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
        )

        # Checkpointing — save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
            }, output_dir / "best_model.pth")
            logger.info("New best model saved (val_acc=%.4f)", best_val_acc)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # Save final checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, output_dir / "final_model.pth")

    writer.close()
    _save_training_curves(history, output_dir)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    return history