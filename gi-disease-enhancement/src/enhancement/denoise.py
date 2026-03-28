"""Denoising filters for GI endoscopy images.

Provides bilateral filtering and adaptive denoising that preserves
edge structure important for lesion boundary detection.
"""

import cv2
import numpy as np


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filter for edge-preserving noise reduction.

    The bilateral filter smooths while keeping edges sharp by combining
    a spatial Gaussian with a range (intensity) Gaussian.

    Args:
        image: BGR image as uint8 numpy array.
        d: Diameter of each pixel neighborhood. Use -1 to compute
            automatically from sigma_space.
        sigma_color: Filter sigma in the color space. Larger values
            mean more distant colors are mixed together.
        sigma_space: Filter sigma in the coordinate space. Larger values
            mean more distant pixels influence each other.

    Returns:
        Denoised BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("noisy_endoscopy.png")
        >>> clean = bilateral_filter(img, d=9, sigma_color=75, sigma_space=75)
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def adaptive_denoise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Apply denoising with strength adapted to the estimated noise level.

    For low noise, uses a light bilateral filter. For moderate noise,
    applies a stronger bilateral pass. For high noise, uses
    Non-Local Means which is slower but more effective.

    Args:
        image: BGR image as uint8 numpy array.
        noise_level: Estimated noise standard deviation in [0, 100].
            Can be computed from a flat region or a blind estimator.

    Returns:
        Denoised BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("noisy.png")
        >>> clean = adaptive_denoise(img, noise_level=25.0)
    """
    if noise_level < 15:
        # Light bilateral for low noise
        return bilateral_filter(image, d=5, sigma_color=40, sigma_space=40)

    if noise_level < 40:
        # Stronger bilateral for moderate noise
        return bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)

    # Non-Local Means for heavy noise — slower but preserves texture
    h = np.clip(noise_level * 0.4, 10, 30)
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=h,
        hForColorComponents=h,
        templateWindowSize=7,
        searchWindowSize=21,
    )