import cv2
import numpy as np


def estimate_cloud_mask(image_np, sensitivity=0.5, min_area_ratio=0.0005):
    """
    Estimate cloud mask M in [0, 1] for an RGB image.
    Accurately isolates cloud bodies while ignoring small bright urban structures (roofs, concrete).

    Args:
        image_np: np.ndarray of shape (H, W, 3), float32 in [0, 1] or uint8 in [0, 255].
        sensitivity: float in [0, 1], higher values detect thinner clouds/haze.
        min_area_ratio: float, minimum connected area ratio to keep (filters out isolated roofs).

    Returns:
        mask: np.ndarray of shape (H, W), float32 in [0, 1].
    """
    if image_np.dtype == np.uint8:
        img = image_np.astype(np.float32) / 255.0
    else:
        img = image_np.copy().astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

    h, w = img.shape[:2]
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # 1. Whiteness Metric (High brightness + low color saturation)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    sat = np.zeros_like(max_c)
    valid = max_c > 1e-4
    sat[valid] = delta[valid] / (max_c[valid] + 1e-6)
    whiteness = (1.0 - sat) * max_c

    # 2. Atmospheric Dark Channel (minimum across local neighborhood)
    k_dark = max(7, int(round(min(h, w) * 0.015)) | 1)
    dark_min = cv2.erode(min_c, cv2.getStructuringElement(cv2.MORPH_RECT, (k_dark, k_dark)))
    dark_smooth = cv2.boxFilter(dark_min, -1, (k_dark, k_dark))

    # Real clouds elevate both the dark channel and whiteness
    cloud_score = (1.0 - sat) * dark_smooth

    # Threshold based on sensitivity parameter
    thresh = 0.40 - (sensitivity - 0.5) * 0.25
    core_cloud = (cloud_score > thresh) & (min_c > (0.35 - (sensitivity - 0.5) * 0.15))
    core_u8 = (core_cloud * 255).astype(np.uint8)

    # 3. Morphological cleanup: close internal holes
    k_morph = max(9, int(round(min(h, w) * 0.015)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
    mask_clean = cv2.morphologyEx(core_u8, cv2.MORPH_CLOSE, kernel)

    # 4. Filter out small isolated urban roofs/houses by connected component size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
    min_area = max(50, int((h * w) * min_area_ratio))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_clean[labels == i] = 0

    # 5. Smooth dilation to cover cloud margins and soft edges
    mask_dilated = cv2.dilate(mask_clean, kernel, iterations=2)
    mask_float = mask_dilated.astype(np.float32) / 255.0

    return mask_float

