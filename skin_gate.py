import cv2
import numpy as np
from PIL import Image

# Refined HSV ranges to reduce beige/sand overlaps
# Tightened Saturation (15 -> 30) to ignore very pale beige
_SKIN_LOWER = np.array([0,   30,  60],  dtype=np.uint8)
_SKIN_UPPER = np.array([25, 160, 255],  dtype=np.uint8)

_SKIN_LOWER2 = np.array([160, 30,  60],  dtype=np.uint8)
_SKIN_UPPER2 = np.array([180, 160, 255], dtype=np.uint8)

def get_texture_score(img_np):
    """Computes the variance of the Laplacian to detect graininess."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def skin_pixel_ratio(image: Image.Image) -> float:
    img_np = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, _SKIN_LOWER,  _SKIN_UPPER)
    mask2 = cv2.inRange(hsv, _SKIN_LOWER2, _SKIN_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Morphological opening to remove small 'noise' (like sand grains)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Closing to solidify the skin area
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (7, 7))

    total_pixels = mask.size
    skin_pixels  = int(np.sum(mask > 0))
    return skin_pixels / total_pixels

def is_skin_image(image: Image.Image, threshold: float = 0.25) -> tuple[bool, float]:
    img_np = np.array(image.convert("RGB"))
    ratio = skin_pixel_ratio(image)
    
    # Texture Check: Desert/Sand usually has higher variance than skin
    # Note: 500 is a baseline; you may need to tune this based on your dataset
    texture_score = get_texture_score(img_np)
    is_not_too_grainy = texture_score < 1200 

    # Combined logic
    passed = (ratio >= threshold) and is_not_too_grainy
    return passed, ratio
