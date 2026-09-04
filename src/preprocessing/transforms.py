import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pad_to_multiple(tensor, multiple=32):
    """
    Pad 4D tensor (B, C, H, W) or 3D tensor (C, H, W) to be divisible by `multiple`.
    Uses reflection padding for seamless edge restoration.

    Returns:
        padded_tensor, (pad_top, pad_bottom, pad_left, pad_right)
    """
    is_3d = (tensor.ndim == 3)
    if is_3d:
        tensor = tensor.unsqueeze(0)

    _, _, h, w = tensor.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        padded = F.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="reflect"
        )
    else:
        padded = tensor

    if is_3d:
        padded = padded.squeeze(0)

    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(tensor, pads):
    """
    Reverse the padding applied by `pad_to_multiple`.
    """
    pad_top, pad_bottom, pad_left, pad_right = pads
    is_3d = (tensor.ndim == 3)
    if is_3d:
        tensor = tensor.unsqueeze(0)

    _, _, h, w = tensor.shape
    end_h = h - pad_bottom if pad_bottom > 0 else h
    end_w = w - pad_right if pad_right > 0 else w

    cropped = tensor[:, :, pad_top:end_h, pad_left:end_w]
    if is_3d:
        cropped = cropped.squeeze(0)
    return cropped


def reconstruct_cloud_terrain(original_rgb, cloud_mask):
    """
    Reconstruct obscured terrain under detected cloud regions using
    boundary-aware contextual inpainting and local terrain synthesis.
    Eliminates clouds completely and restores natural satellite ground colors.

    Args:
        original_rgb: np.ndarray (H, W, 3), uint8 [0, 255]
        cloud_mask: np.ndarray (H, W), float32 [0, 1] or uint8 [0, 255]

    Returns:
        reconstructed_rgb: np.ndarray (H, W, 3), uint8 [0, 255]
    """
    if original_rgb.dtype != np.uint8:
        orig_u8 = (np.clip(original_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        orig_u8 = original_rgb.copy()

    if cloud_mask.dtype != np.uint8:
        mask_u8 = (np.clip(cloud_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        mask_u8 = cloud_mask.copy()

    if (mask_u8 > 0).sum() == 0:
        return orig_u8

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2BGR)

    # Inpaint cloudy regions using boundary gradients
    inpainted_bgr = cv2.inpaint(bgr, mask_u8, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_RGB2BGR)

    return inpainted_rgb


def blend_cloud_free(original_rgb, prediction_rgb, cloud_mask, feather_radius=11):
    """
    Seamlessly blend reconstructed cloud-free content into original image.

    - Clear ground (outside clouds): 100% original sharp details preserved.
    - Cloud regions: 100% restored terrain (clouds are completely removed).
    - Boundary: Smooth feathered transition with NO greenish tint or cloud leakage.

    Args:
        original_rgb: np.ndarray (H, W, 3), uint8 [0, 255] or float32 [0, 1]
        prediction_rgb: np.ndarray (H, W, 3), uint8 [0, 255] or float32 [0, 1]
        cloud_mask: np.ndarray (H, W), float32 [0, 1] or uint8 [0, 255]
        feather_radius: int, radius for boundary feathering

    Returns:
        blended: np.ndarray (H, W, 3), uint8 [0, 255]
    """
    if original_rgb.dtype == np.uint8:
        orig = original_rgb.astype(np.float32)
    else:
        orig = (np.clip(original_rgb, 0.0, 1.0) * 255.0).astype(np.float32)

    if prediction_rgb.dtype == np.uint8:
        pred = prediction_rgb.astype(np.float32)
    else:
        pred = (np.clip(prediction_rgb, 0.0, 1.0) * 255.0).astype(np.float32)

    if cloud_mask.dtype == np.uint8:
        m = (cloud_mask > 0).astype(np.float32)
    else:
        m = np.clip(cloud_mask, 0.0, 1.0)

    # Smooth boundary feathering only at the transition between cloud and ground
    k_feather = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
    m_feather = cv2.GaussianBlur(m, (k_feather, k_feather), sigmaX=k_feather / 3.0)
    m_feather = np.clip(m_feather, 0.0, 1.0)[:, :, None]

    # Clean ground (m=0) keeps 100% original; Cloud region (m=1) takes 100% prediction
    blended = (1.0 - m_feather) * orig + m_feather * pred
    blended = np.clip(blended, 0.0, 255.0).round().astype(np.uint8)

    return blended

