"""Data loading utilities for GI disease classification.

Provides two PyTorch Dataset classes:

- ``GIDataset``: loads images and labels from a CSV or a flat list of
  (path, label) pairs — useful when splits are defined externally.
- ``ImageFolderDataset``: mirrors torchvision's ImageFolder convention
  where each subdirectory name is a class label.

The helper ``get_data_loaders`` builds train / val / test DataLoaders
from either layout with standard ImageNet preprocessing.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

# ImageNet normalisation constants used by torchvision pretrained models
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _preprocess(image: np.ndarray, image_size: int, augment: bool) -> torch.Tensor:
    """Resize, optionally augment, normalise, and convert to tensor."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))

    if augment:
        if np.random.random() > 0.5:
            resized = np.fliplr(resized).copy()
        if np.random.random() > 0.5:
            resized = np.flipud(resized).copy()

    normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalised).permute(2, 0, 1)  # HWC -> CHW


class GIDataset(Dataset):
    """Dataset from an explicit list of (image_path, label) pairs.

    Args:
        samples: List of ``(path, label_int)`` tuples.
        image_size: Spatial size to resize images to.
        augment: Apply random horizontal/vertical flips.

    Example:
        >>> samples = [("data/raw/img001.png", 0), ("data/raw/img002.png", 1)]
        >>> ds = GIDataset(samples, image_size=224)
        >>> img, label = ds[0]
        >>> img.shape
        torch.Size([3, 224, 224])
    """

    def __init__(
        self,
        samples: list[tuple[str | Path, int]],
        image_size: int = 224,
        augment: bool = False,
    ):
        self.samples = [(Path(p), l) for p, l in samples]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        tensor = _preprocess(image, self.image_size, self.augment)
        return tensor, label


class ImageFolderDataset(Dataset):
    """Dataset that reads from a directory-per-class layout.

    Expected structure::

        data_dir/
            class_a/
                img001.png
                img002.png
            class_b/
                img003.png
                ...

    Args:
        data_dir: Root directory containing class subdirectories.
        image_size: Spatial size to resize images to.
        augment: Apply random horizontal/vertical flips.

    Attributes:
        class_names: Sorted list of discovered class directory names.
        class_to_idx: Mapping from class name to integer label.

    Example:
        >>> ds = ImageFolderDataset("data/splits/train", image_size=224, augment=True)
        >>> print(ds.class_names)
        ['esophagitis', 'normal', 'polyps', 'ulcerative-colitis']
        >>> img, label = ds[0]
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        self.class_names = sorted(
            d.name for d in self.data_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.data_dir / cls_name
            label = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))

        logger.info(
            "ImageFolderDataset: %d images, %d classes from %s",
            len(self.samples), len(self.class_names), self.data_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        tensor = _preprocess(image, self.image_size, self.augment)
        return tensor, label


def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders from a directory-per-class layout.

    If ``data_dir`` contains subdirectories named ``train``, ``val`` (or
    ``validation``), and ``test``, those are used directly.  Otherwise the
    dataset is loaded as a single ``ImageFolderDataset`` and split
    randomly according to *val_split* and *test_split*.

    Args:
        data_dir: Root data directory.
        batch_size: Batch size for all loaders.
        num_workers: Dataloader worker processes.
        image_size: Spatial resize target.
        val_split: Fraction of data for validation when splitting.
        test_split: Fraction of data for testing when splitting.
        seed: Random seed for reproducible splits.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        ``DataLoader`` instances.  Also includes ``"class_names"``
        (list of str).

    Example:
        >>> loaders = get_data_loaders("data/splits", batch_size=16)
        >>> for images, labels in loaders["train"]:
        ...     print(images.shape, labels.shape)
        ...     break
        torch.Size([16, 3, 224, 224]) torch.Size([16])
    """
    data_dir = Path(data_dir)

    # Check for pre-split directory structure
    subdirs = {d.name for d in data_dir.iterdir() if d.is_dir()}
    val_name = "val" if "val" in subdirs else "validation" if "validation" in subdirs else None
    pre_split = "train" in subdirs and val_name is not None

    if pre_split:
        train_ds = ImageFolderDataset(data_dir / "train", image_size, augment=True)
        val_ds = ImageFolderDataset(data_dir / val_name, image_size, augment=False)
        test_ds = (
            ImageFolderDataset(data_dir / "test", image_size, augment=False)
            if "test" in subdirs
            else val_ds
        )
        class_names = train_ds.class_names
        logger.info("Using pre-split directories: train=%d, val=%d, test=%d",
                     len(train_ds), len(val_ds), len(test_ds))
    else:
        full_ds = ImageFolderDataset(data_dir, image_size, augment=False)
        class_names = full_ds.class_names

        n = len(full_ds)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=generator).tolist()

        n_test = int(n * test_split)
        n_val = int(n * val_split)

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        # Training subset gets augmentation via a wrapper
        train_ds = _AugmentedSubset(full_ds, train_idx, image_size)
        val_ds = Subset(full_ds, val_idx)
        test_ds = Subset(full_ds, test_idx)

        logger.info("Random split: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
        "class_names": class_names,
    }
    return loaders


class _AugmentedSubset(Dataset):
    """Subset wrapper that re-reads images with augmentation enabled."""

    def __init__(self, parent: ImageFolderDataset, indices: list[int], image_size: int):
        self.samples = [parent.samples[i] for i in indices]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        tensor = _preprocess(image, self.image_size, augment=True)
        return tensor, label