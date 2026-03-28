"""Sharpening filters for GI endoscopy images.

Provides unsharp masking and adaptive sharpening to recover edge
detail lost to motion blur or optical limitations during endoscopy.
"""

import cv2
import numpy as np


def unsharp_mask(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Apply unsharp masking to enhance edges and fine detail.

    Creates a blurred copy, computes the difference (the "mask"),
    and adds a scaled version back to the original.

    Args:
        image: BGR image as uint8 numpy array.
        kernel_size: Size of the Gaussian blur kernel. Must be odd.
        sigma: Standard deviation of the Gaussian blur.
        amount: Strength of the sharpening effect. Values > 1.0 give
            stronger sharpening; values < 1.0 give subtle enhancement.

    Returns:
        Sharpened BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("blurry_endoscopy.png")
        >>> sharp = unsharp_mask(img, kernel_size=5, sigma=1.0, amount=1.5)
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    # float32 to avoid uint8 overflow during arithmetic
    sharpened = cv2.addWeighted(
        image.astype(np.float32),
        1.0 + amount,
        blurred.astype(np.float32),
        -amount,
        0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def adaptive_sharpen(image: np.ndarray, blur_score: float) -> np.ndarray:
    """Apply sharpening with strength adapted to the detected blur level.

    Higher blur scores (more blur) receive stronger sharpening with
    larger kernels. Near-sharp images get minimal processing to
    avoid artifact introduction.

    Args:
        image: BGR image as uint8 numpy array.
        blur_score: Blur severity in [0, 1] where 0 is perfectly sharp
            and 1 is heavily blurred. Typically computed via Laplacian
            variance or a learned blur estimator.

    Returns:
        Sharpened BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("slightly_blurry.png")
        >>> sharp = adaptive_sharpen(img, blur_score=0.6)
    """
    if blur_score < 0.2:
        # Nearly sharp — skip to avoid ringing artifacts
        return image

    # Scale amount and kernel with blur severity
    amount = np.interp(blur_score, [0.2, 1.0], [0.5, 2.0])
    kernel_size = 3 if blur_score < 0.5 else 5 if blur_score < 0.8 else 7
    sigma = np.interp(blur_score, [0.2, 1.0], [0.5, 2.0])

    return unsharp_mask(image, kernel_size=kernel_size, sigma=sigma, amount=amount)