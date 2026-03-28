"""No-reference image quality assessment for GI endoscopy images.

Provides both learned metrics (BRISQUE, NIQE via pyiqa) and classical
signal-level estimators (blur, noise, contrast) that can drive the
adaptive enhancement pipeline.
"""

from functools import lru_cache

import cv2
import numpy as np
import pyiqa
import torch


@lru_cache(maxsize=4)
def _get_iqa_metric(metric_name: str, device: str = "cpu") -> pyiqa.InferenceModel:
    """Return a cached pyiqa inference model.

    Caching avoids re-loading weights on every call, which matters when
    scoring many images in a loop.
    """
    return pyiqa.create_metric(metric_name, device=torch.device(device))


def _bgr_to_tensor(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert a BGR uint8 image to the [0,1] RGB float tensor pyiqa expects."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)


def calculate_brisque(image: np.ndarray, device: str = "cpu") -> float:
    """Compute the BRISQUE no-reference quality score.

    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) fits a
    multivariate Gaussian to normalised luminance coefficients and
    measures their deviation from a natural-scene model.

    Args:
        image: BGR image as uint8 numpy array.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        BRISQUE score (lower is better quality, typical range 0-100).

    Example:
        >>> import cv2
        >>> img = cv2.imread("endoscopy.png")
        >>> score = calculate_brisque(img)
        >>> print(f"BRISQUE: {score:.2f}")
    """
    metric = _get_iqa_metric("brisque", device)
    tensor = _bgr_to_tensor(image, device)
    with torch.no_grad():
        score = metric(tensor)
    return float(score.item())


def calculate_niqe(image: np.ndarray, device: str = "cpu") -> float:
    """Compute the NIQE no-reference quality score.

    NIQE (Natural Image Quality Evaluator) compares local normalised
    luminance statistics against a pre-trained pristine-image model.
    Unlike BRISQUE it requires no distortion-specific training.

    Args:
        image: BGR image as uint8 numpy array.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        NIQE score (lower is better quality, typical range 0-20).

    Example:
        >>> img = cv2.imread("endoscopy.png")
        >>> score = calculate_niqe(img)
        >>> print(f"NIQE: {score:.2f}")
    """
    metric = _get_iqa_metric("niqe", device)
    tensor = _bgr_to_tensor(image, device)
    with torch.no_grad():
        score = metric(tensor)
    return float(score.item())


def detect_blur(image: np.ndarray) -> float:
    """Estimate image sharpness using Laplacian variance.

    The variance of the Laplacian response correlates strongly with
    perceived sharpness — sharp images produce high variance, blurry
    images produce low variance.

    Args:
        image: BGR image as uint8 numpy array.

    Returns:
        Laplacian variance (higher = sharper). Values below ~100
        typically indicate noticeable blur in endoscopy images.

    Example:
        >>> img = cv2.imread("blurry.png")
        >>> sharpness = detect_blur(img)
        >>> print(f"Laplacian variance: {sharpness:.1f}")
        >>> if sharpness < 100:
        ...     print("Image is blurry")
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_noise(image: np.ndarray) -> float:
    """Estimate noise standard deviation using local variance in smooth regions.

    Applies a robust estimator based on the median absolute deviation
    (MAD) of the Laplacian, which responds to high-frequency noise
    while being insensitive to edges and texture.

    Args:
        image: BGR image as uint8 numpy array.

    Returns:
        Estimated noise standard deviation (0-100 typical range).

    Example:
        >>> img = cv2.imread("noisy.png")
        >>> sigma = estimate_noise(img)
        >>> print(f"Estimated noise sigma: {sigma:.1f}")
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # MAD estimator: sigma = median(|L|) / 0.6745
    sigma = float(np.median(np.abs(laplacian)) / 0.6745)
    return np.clip(sigma, 0, 100)


def measure_contrast(image: np.ndarray) -> dict[str, float]:
    """Measure image contrast using multiple complementary metrics.

    Returns RMS contrast (L-channel standard deviation), Michelson
    contrast ((max-min)/(max+min)), and Weber contrast of the mean
    relative to the midpoint.

    Args:
        image: BGR image as uint8 numpy array.

    Returns:
        Dict with keys:
        - ``rms``: Root-mean-square contrast in [0, 128].
        - ``michelson``: Michelson contrast in [0, 1].
        - ``dynamic_range``: Fraction of the 0-255 range actually used.

    Example:
        >>> img = cv2.imread("low_contrast.png")
        >>> c = measure_contrast(img)
        >>> print(f"RMS contrast: {c['rms']:.1f}")
        >>> print(f"Michelson:    {c['michelson']:.3f}")
        >>> print(f"Dynamic range used: {c['dynamic_range']:.1%}")
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)

    # RMS contrast — standard deviation of luminance
    rms = float(l_channel.std())

    # Michelson contrast
    l_min, l_max = float(l_channel.min()), float(l_channel.max())
    denom = l_max + l_min
    michelson = (l_max - l_min) / denom if denom > 0 else 0.0

    # Dynamic range utilisation
    dynamic_range = (l_max - l_min) / 255.0

    return {
        "rms": round(rms, 2),
        "michelson": round(michelson, 4),
        "dynamic_range": round(dynamic_range, 4),
    }