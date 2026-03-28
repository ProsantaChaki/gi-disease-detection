"""CLAHE-based contrast enhancement for GI endoscopy images.

Applies Contrast Limited Adaptive Histogram Equalization to improve
local contrast while preventing noise amplification common in
endoscopic imagery.
"""

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to an image in LAB color space.

    Converts to LAB, applies CLAHE to the L channel only, and converts
    back to BGR. This preserves color information while enhancing contrast.

    Args:
        image: BGR image as uint8 numpy array.
        clip_limit: Contrast limiting threshold. Higher values allow more
            contrast but risk amplifying noise.
        tile_grid_size: Number of tiles in row and column directions.
            Smaller tiles give more local adaptation.

    Returns:
        Enhanced BGR image as uint8 numpy array.

    Example:
        >>> import cv2
        >>> img = cv2.imread("endoscopy.png")
        >>> enhanced = apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8))
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def adaptive_clahe(image: np.ndarray, quality_score: float) -> np.ndarray:
    """Apply CLAHE with parameters adapted to the image quality score.

    Lower quality scores (darker, lower contrast images) receive stronger
    enhancement via higher clip limits and finer tile grids.

    Args:
        image: BGR image as uint8 numpy array.
        quality_score: Image quality score in [0, 1] where 0 is worst
            and 1 is best. Typically from a no-reference IQA metric.

    Returns:
        Enhanced BGR image as uint8 numpy array.

    Example:
        >>> img = cv2.imread("low_contrast.png")
        >>> enhanced = adaptive_clahe(img, quality_score=0.3)
    """
    # Low quality -> higher clip limit (more enhancement)
    clip_limit = np.interp(quality_score, [0.0, 1.0], [4.0, 1.0])

    # Low quality -> finer grid (more local adaptation)
    grid_size = 16 if quality_score < 0.3 else 12 if quality_score < 0.6 else 8

    return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=(grid_size, grid_size))