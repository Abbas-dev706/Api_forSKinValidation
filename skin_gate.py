"""
skin_gate.py
------------
Human skin gating filter.
Rejects images that don't contain sufficient human skin before
they reach the lesion classifier. Uses HSV-based skin color
segmentation — fast, no extra model needed.
"""

import cv2
import numpy as np
from PIL import Image


# HSV skin color range (works for diverse skin tones)
# Lower/upper bounds tuned on Fitzpatrick scale types I–VI
_SKIN_LOWER = np.array([0,   15,  50],  dtype=np.uint8)
_SKIN_UPPER = np.array([25, 200, 255],  dtype=np.uint8)

# Second range to catch reddish/dark tones
_SKIN_LOWER2 = np.array([160, 15,  50],  dtype=np.uint8)
_SKIN_UPPER2 = np.array([180, 200, 255], dtype=np.uint8)


def skin_pixel_ratio(image: Image.Image) -> float:
    """
    Returns the fraction of pixels classified as skin (0.0 – 1.0).

    Parameters
    ----------
    image : PIL.Image
        RGB image of any size.

    Returns
    -------
    float
        Proportion of skin-coloured pixels.
    """
    img_np = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, _SKIN_LOWER,  _SKIN_UPPER)
    mask2 = cv2.inRange(hsv, _SKIN_LOWER2, _SKIN_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Optional: morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = mask.size
    skin_pixels  = int(np.sum(mask > 0))
    return skin_pixels / total_pixels


def is_skin_image(
    image: Image.Image,
    threshold: float = 0.15,
) -> tuple[bool, float]:
    """
    Gate: decides whether an image contains enough human skin.

    Parameters
    ----------
    image     : PIL.Image  — input image
    threshold : float      — minimum skin pixel ratio (default 0.15 = 15 %)

    Returns
    -------
    (passed: bool, ratio: float)
        passed — True if the image passes the skin gate
        ratio  — actual skin pixel ratio for logging / debugging
    """
    ratio = skin_pixel_ratio(image)
    return ratio >= threshold, ratio


# ---------------------------------------------------------------------------
# Quick CLI test:  python skin_gate.py path/to/image.jpg
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    img  = Image.open(path)
    passed, ratio = is_skin_image(img)
    status = "PASS" if passed else "REJECT"
    print(f"[{status}]  skin ratio = {ratio:.3f}  ({path})")
