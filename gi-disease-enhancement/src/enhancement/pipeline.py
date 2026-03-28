"""Adaptive image enhancement pipeline for GI endoscopy images.

Combines quality assessment with targeted enhancement stages:
denoising, contrast correction (CLAHE), and sharpening. Each stage
adapts its parameters based on detected quality issues so that
already-good aspects of the image are left untouched.
"""

import cv2
import numpy as np

from .clahe import adaptive_clahe
from .denoise import adaptive_denoise
from .sharpen import adaptive_sharpen


class ImageEnhancer:
    """Adaptive enhancement pipeline driven by quality assessment.

    Assesses noise, contrast, and blur, then applies the corresponding
    enhancement stages only as strongly as needed.

    Args:
        denoise_threshold: Noise level below which denoising is skipped.
        contrast_threshold: Quality score above which CLAHE is skipped.
        blur_threshold: Blur score below which sharpening is skipped.

    Example:
        >>> import cv2
        >>> enhancer = ImageEnhancer()
        >>> img = cv2.imread("endoscopy.png")
        >>> quality = enhancer.assess_quality(img)
        >>> print(quality)
        {'noise_level': 18.2, 'contrast_score': 0.45, 'blur_score': 0.62}
        >>> enhanced = enhancer.enhance(img)
    """

    def __init__(
        self,
        denoise_threshold: float = 10.0,
        contrast_threshold: float = 0.8,
        blur_threshold: float = 0.2,
    ):
        self.denoise_threshold = denoise_threshold
        self.contrast_threshold = contrast_threshold
        self.blur_threshold = blur_threshold

    def assess_quality(self, image: np.ndarray) -> dict[str, float]:
        """Estimate noise level, contrast quality, and blur severity.

        Uses lightweight heuristics suitable for real-time pipelines:
        - Noise: median absolute deviation on Laplacian (robust to edges).
        - Contrast: normalised standard deviation of the L channel.
        - Blur: inverse normalised Laplacian variance (higher = blurrier).

        Args:
            image: BGR image as uint8 numpy array.

        Returns:
            Dict with keys ``noise_level`` (0-100), ``contrast_score``
            (0-1, higher is better), and ``blur_score`` (0-1, higher is
            blurrier).

        Example:
            >>> enhancer = ImageEnhancer()
            >>> q = enhancer.assess_quality(cv2.imread("img.png"))
            >>> q["noise_level"]
            22.5
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        noise_level = self._estimate_noise(gray)
        contrast_score = self._estimate_contrast(image)
        blur_score = self._estimate_blur(gray)

        return {
            "noise_level": round(noise_level, 2),
            "contrast_score": round(contrast_score, 4),
            "blur_score": round(blur_score, 4),
        }

    def enhance(self, image: np.ndarray, quality: dict[str, float] | None = None) -> np.ndarray:
        """Run the adaptive enhancement pipeline.

        Stages are applied in order: denoise -> CLAHE -> sharpen.
        Each stage is skipped or attenuated when the corresponding
        quality metric is already acceptable.

        Args:
            image: BGR image as uint8 numpy array.
            quality: Pre-computed quality dict from ``assess_quality``.
                If None, quality is assessed automatically.

        Returns:
            Enhanced BGR image as uint8 numpy array.

        Example:
            >>> enhancer = ImageEnhancer()
            >>> enhanced = enhancer.enhance(cv2.imread("endoscopy.png"))
        """
        if quality is None:
            quality = self.assess_quality(image)

        result = image.copy()

        # Stage 1: Denoise
        if quality["noise_level"] >= self.denoise_threshold:
            result = adaptive_denoise(result, quality["noise_level"])

        # Stage 2: Contrast enhancement
        if quality["contrast_score"] < self.contrast_threshold:
            result = adaptive_clahe(result, quality["contrast_score"])

        # Stage 3: Sharpening
        if quality["blur_score"] >= self.blur_threshold:
            result = adaptive_sharpen(result, quality["blur_score"])

        return result

    # ------------------------------------------------------------------
    # Internal quality estimators
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """Estimate noise std via the MAD of high-frequency Laplacian response."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Median absolute deviation — robust to edges
        sigma = np.median(np.abs(laplacian)) / 0.6745
        return float(np.clip(sigma, 0, 100))

    @staticmethod
    def _estimate_contrast(image: np.ndarray) -> float:
        """Score contrast as normalised L-channel standard deviation."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float64)
        score = l_channel.std() / 128.0  # normalise to ~[0, 1]
        return float(np.clip(score, 0, 1))

    @staticmethod
    def _estimate_blur(gray: np.ndarray) -> float:
        """Score blur via inverse normalised Laplacian variance.

        A sharp image has high variance; a blurry image has low variance.
        The score is mapped so that 0 = sharp, 1 = very blurry.
        """
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Empirical mapping: variance ~500+ is sharp, ~50 or below is blurry
        score = 1.0 - np.clip(laplacian_var / 500.0, 0, 1)
        return float(score)