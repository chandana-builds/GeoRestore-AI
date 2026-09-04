import os
import sys
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.predict import load_model as src_load_model, remove_clouds


def load_model():
    """
    Load the best available trained model onto CUDA (if available) or CPU.
    Automatically fetches weights from Hugging Face Hub if not present locally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    best_path = os.path.join(checkpoint_dir, "best_unet.pth")
    final_path = os.path.join(checkpoint_dir, "unet_final.pth")

    if not os.path.exists(best_path) and not os.path.exists(final_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading model checkpoint from Hugging Face Hub (chandana987/georestore-model)...")
            best_path = hf_hub_download(
                repo_id="chandana987/georestore-model",
                filename="best_unet.pth",
                local_dir=checkpoint_dir
            )
        except Exception as err:
            print(f"Failed to auto-download weights from Hugging Face: {err}")

    model_path = best_path if os.path.exists(best_path) else final_path
    model = src_load_model(model_path, device)
    return model, device



def predict_image(model, device, image, apply_mask_blending=True, cloud_sensitivity=0.5, mode="hybrid"):
    """
    Predict and restore a cloud-free satellite image from arbitrary uploaded image.

    Returns:
        clean_img: np.ndarray (H, W, 3), uint8
        cloud_mask: np.ndarray (H, W), float32 [0, 1]
    """
    clean_img, cloud_mask, _ = remove_clouds(
        model=model,
        device=device,
        image_input=image,
        apply_mask_blending=apply_mask_blending,
        cloud_sensitivity=cloud_sensitivity,
        mode=mode
    )
    return clean_img, cloud_mask
