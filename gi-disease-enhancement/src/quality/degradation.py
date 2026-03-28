"""Synthetic image degradation for controlled enhancement experiments.

Generates degraded versions of clean endoscopy images at configurable
severity levels so that enhancement methods can be evaluated with
known ground truth.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Degradation parameter presets keyed by severity level.
# Each preset maps to (noise_sigma, blur_kernel, gamma, jpeg_quality).
DEGRADATION_PRESETS: dict[str, dict[str, float | int]] = {
    "low": {"noise_sigma": 10, "blur_kernel": 3, "gamma": 0.8, "jpeg_quality": 70},
    "medium": {"noise_sigma": 25, "blur_kernel": 5, "gamma": 0.6, "jpeg_quality": 40},
    "high": {"noise_sigma": 50, "blur_kernel": 9, "gamma": 0.4, "jpeg_quality": 15},
}


# ------------------------------------------------------------------
# Individual degradation functions
# ------------------------------------------------------------------


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Add additive white Gaussian noise.

    Args:
        image: BGR image as uint8 numpy array.
        sigma: Standard deviation of the Gaussian noise in [0, 255].

    Returns:
        Noisy BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("clean.png")
        >>> noisy = add_gaussian_noise(img, sigma=30)
    """
    noise = np.random.default_rng().normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to simulate defocus or motion blur.

    Args:
        image: BGR image as uint8 numpy array.
        kernel_size: Size of the Gaussian kernel (must be odd and >= 1).

    Returns:
        Blurred BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("sharp.png")
        >>> blurry = add_gaussian_blur(img, kernel_size=7)
    """
    # Ensure odd kernel
    kernel_size = max(1, kernel_size | 1)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def reduce_contrast(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Reduce contrast via gamma compression.

    Gamma < 1 compresses the dynamic range (lower contrast, washed out).
    Gamma > 1 expands it. This simulates poor illumination or sensor
    dynamic range limitations common in endoscopy.

    Args:
        image: BGR image as uint8 numpy array.
        gamma: Gamma value. Values in (0, 1) reduce contrast.

    Returns:
        Contrast-reduced BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("good_contrast.png")
        >>> low_contrast = reduce_contrast(img, gamma=0.5)
    """
    lut = np.array(
        [np.clip(((i / 255.0) ** gamma) * 255, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def jpeg_compression(image: np.ndarray, quality: int = 30) -> np.ndarray:
    """Apply JPEG compression artifacts.

    Encodes the image to JPEG at the given quality level and decodes
    it back, introducing blocking and ringing artifacts.

    Args:
        image: BGR image as uint8 numpy array.
        quality: JPEG quality factor in [1, 100]. Lower values produce
            heavier compression artifacts.

    Returns:
        Compressed BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("pristine.png")
        >>> compressed = jpeg_compression(img, quality=20)
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))]
    _, buf = cv2.imencode(".jpg", image, encode_params)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------
# Dataset-level degradation
# ------------------------------------------------------------------


def _degrade_single(image: np.ndarray, preset: dict[str, float | int]) -> np.ndarray:
    """Apply all four degradations in sequence using preset parameters."""
    result = add_gaussian_noise(image, sigma=float(preset["noise_sigma"]))
    result = add_gaussian_blur(result, kernel_size=int(preset["blur_kernel"]))
    result = reduce_contrast(result, gamma=float(preset["gamma"]))
    result = jpeg_compression(result, quality=int(preset["jpeg_quality"]))
    return result


def create_degraded_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    levels: list[str] | None = None,
) -> dict[str, int]:
    """Generate degraded copies of every image in *input_dir*.

    For each severity level a subdirectory is created under *output_dir*,
    and every ``.png`` / ``.jpg`` / ``.jpeg`` / ``.bmp`` / ``.tif``
    image in *input_dir* is degraded and saved there.

    Args:
        input_dir: Path to a flat directory of clean source images.
        output_dir: Root path for output. Level subdirectories are
            created automatically (e.g. ``output_dir/medium/``).
        levels: List of severity level names. Must be keys of
            ``DEGRADATION_PRESETS``. Defaults to
            ``["low", "medium", "high"]``.

    Returns:
        Dict mapping each level name to the number of images written.

    Example:
        >>> counts = create_degraded_dataset(
        ...     "data/raw", "data/degraded", levels=["low", "high"]
        ... )
        >>> print(counts)
        {'low': 500, 'high': 500}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if levels is None:
        levels = ["low", "medium", "high"]

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_paths:
        logger.warning("No images found in %s", input_dir)
        return {level: 0 for level in levels}

    counts: dict[str, int] = {}

    for level in levels:
        if level not in DEGRADATION_PRESETS:
            raise ValueError(
                f"Unknown degradation level '{level}'. "
                f"Choose from: {list(DEGRADATION_PRESETS)}"
            )

        preset = DEGRADATION_PRESETS[level]
        level_dir = output_dir / level
        level_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Could not read %s — skipping", img_path)
                continue

            degraded = _degrade_single(image, preset)
            out_path = level_dir / img_path.name
            cv2.imwrite(str(out_path), degraded)
            written += 1

        counts[level] = written
        logger.info("Level '%s': wrote %d images to %s", level, written, level_dir)

    return counts