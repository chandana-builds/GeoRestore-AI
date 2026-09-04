import os
import cv2
import numpy as np
import torch
from PIL import Image

from src.models.unet import UNet
from src.preprocessing.cloud_mask import estimate_cloud_mask
from src.preprocessing.transforms import pad_to_multiple, unpad, blend_cloud_free, reconstruct_cloud_terrain


def load_model(model_path, device=None):
    """
    Load the trained U-Net model from a file path or default checkpoint.
    Supports both raw state_dicts and checkpoint dictionaries.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess_image(image_input):
    """
    Read and preprocess an input image from path, numpy array, or PIL Image.

    Returns:
        tensor: torch.Tensor of shape (1, 3, H, W), float32 in [0, 1]
        original_rgb: np.ndarray of shape (H, W, 3), uint8 in [0, 255]
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at {image_input}")
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Failed to decode image from {image_input}")
        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        original_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            original_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            original_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            original_rgb = image_input.copy()
        if original_rgb.dtype != np.uint8 and original_rgb.max() <= 1.0:
            original_rgb = (original_rgb * 255.0).clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img_float = original_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_rgb


def predict(model, image_tensor, device=None):
    """
    Run UNet prediction with automatic reflection padding for any resolution.

    Returns:
        output_tensor: torch.Tensor of shape (1, 3, H, W) on CPU
    """
    if device is None:
        device = next(model.parameters()).device

    image_tensor = image_tensor.to(device)
    padded_tensor, pads = pad_to_multiple(image_tensor, multiple=32)

    with torch.no_grad():
        output = model(padded_tensor)

    unpadded = unpad(output, pads)
    return torch.clamp(unpadded.cpu(), 0.0, 1.0)


def remove_clouds(
    model,
    device,
    image_input,
    apply_mask_blending=True,
    cloud_sensitivity=0.5,
    mode="hybrid"
):
    """
    End-to-end cloud removal pipeline on any unseen satellite image.

    Args:
        model: loaded UNet model
        device: torch.device
        image_input: image path, numpy array, or PIL Image
        apply_mask_blending: bool, if True, preserves clear ground and restores clouds
        cloud_sensitivity: float [0, 1], higher values catch lighter haze
        mode: str, 'hybrid' (UNet + contextual inpainting), 'unet' (pure UNet), or 'inpainting'

    Returns:
        final_clean: np.ndarray (H, W, 3), uint8 [0, 255]
        cloud_mask: np.ndarray (H, W), float32 [0, 1]
        raw_pred: np.ndarray (H, W, 3), uint8 [0, 255]
    """
    image_tensor, original_rgb = preprocess_image(image_input)
    h, w = original_rgb.shape[:2]

    cloud_mask = estimate_cloud_mask(original_rgb, sensitivity=cloud_sensitivity)

    # 1. Pure UNet prediction
    if model is not None and mode in ("unet", "hybrid"):
        # For very large images, run UNet at 512 for receptive field consistency
        if max(h, w) > 768:
            small_rgb = cv2.resize(original_rgb, (512, 512))
            small_tensor = torch.from_numpy(small_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            small_out = predict(model, small_tensor, device)
            small_np = small_out.squeeze(0).permute(1, 2, 0).numpy()
            pred_uint8 = cv2.resize((np.clip(small_np, 0.0, 1.0) * 255.0).round().astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            output_tensor = predict(model, image_tensor, device)
            pred_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
            pred_uint8 = (np.clip(pred_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        pred_uint8 = original_rgb.copy()

    # 2. Contextual terrain reconstruction for cloudy regions
    if mode in ("hybrid", "inpainting"):
        terrain_reconstructed = reconstruct_cloud_terrain(original_rgb, cloud_mask)
        if mode == "hybrid" and model is not None:
            # Combine deep UNet semantic prediction with contextual inpainting
            cloud_content = cv2.addWeighted(terrain_reconstructed, 0.6, pred_uint8, 0.4, 0)
        else:
            cloud_content = terrain_reconstructed
    else:
        cloud_content = pred_uint8

    # 3. Seamlessly blend: keep 100% of clear ground, replace clouds with restored content
    if apply_mask_blending:
        final_clean = blend_cloud_free(original_rgb, cloud_content, cloud_mask)
    else:
        final_clean = cloud_content

    return final_clean, cloud_mask, pred_uint8