# skin_gate.py
import cv2
import numpy as np
from PIL import Image

def skin_pixel_ratio(image: Image.Image) -> float:
    img_np = np.array(image.convert("RGB"))
    
    # 1. YCrCb Strict Bounds
    ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    lower_ycrcb = np.array([0, 138, 67], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 133], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # 2. HSV Strict Bounds (Limits the "Sand/Wood" yellowish hues)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_hsv = np.array([0, 15, 0], dtype=np.uint8)
    upper_hsv = np.array([17, 170, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 3. RGB Heuristics (Peer et al. algorithm for skin)
    R, G, B = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    mask_rgb = (
        (R > 95) & (G > 40) & (B > 20) & 
        ((np.maximum(R, np.maximum(G, B)) - np.minimum(R, np.minimum(G, B))) > 15) & 
        (np.abs(R.astype(int) - G.astype(int)) > 15) & 
        (R > G) & (R > B)
    ).astype(np.uint8) * 255

    # Intersect all three masks (Pixels MUST pass all three tests)
    combined_color_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    combined_color_mask = cv2.bitwise_and(combined_color_mask, mask_rgb)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Edge Subtraction (Canny)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=80, threshold2=160)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Final Subtraction
    refined_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(edges))

    total_pixels = refined_mask.size
    skin_pixels = int(np.sum(refined_mask > 0))
    
    return skin_pixels / total_pixels

def is_skin_image(image: Image.Image, threshold: float = 0.15) -> tuple[bool, float]:
    ratio = skin_pixel_ratio(image)
    return ratio >= threshold, ratio