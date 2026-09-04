import os
import sys
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.inference.predict import load_model, remove_clouds


def main():
    parser = argparse.ArgumentParser(description="Compare Cloudy vs Cloud-Free Satellite Image")
    parser.add_argument("--cloud", type=str, default=None, help="Path to cloudy image")
    parser.add_argument("--label", type=str, default=None, help="Path to ground truth clean image (optional)")
    parser.add_argument("--output", type=str, default=None, help="Path to save comparison figure")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Model resolution
    if args.model is not None:
        model_path = args.model
    else:
        best_path = os.path.join(project_dir, "outputs", "checkpoints", "best_unet.pth")
        final_path = os.path.join(project_dir, "outputs", "checkpoints", "unet_final.pth")
        model_path = best_path if os.path.exists(best_path) else final_path

    model = load_model(model_path, device)

    # Image paths
    cloud_path = args.cloud
    label_path = args.label

    if cloud_path is None:
        # Default to an unseen sample from RICE1 or RICE2
        test_rice2 = os.path.join(project_dir, "data", "raw", "RICE2", "cloud", "100.png")
        test_rice1 = os.path.join(project_dir, "data", "raw", "RICE1", "cloud", "450.png")
        if os.path.exists(test_rice2):
            cloud_path = test_rice2
            label_path = os.path.join(project_dir, "data", "raw", "RICE2", "label", "100.png")
        elif os.path.exists(test_rice1):
            cloud_path = test_rice1
            label_path = os.path.join(project_dir, "data", "raw", "RICE1", "label", "450.png")
        else:
            cloud_path = os.path.join(project_dir, "data", "raw", "RICE1", "cloud", "0.png")
            label_path = os.path.join(project_dir, "data", "raw", "RICE1", "label", "0.png")

    print(f"Cloud Image: {cloud_path}")
    cloud_img = cv2.cvtColor(cv2.imread(cloud_path), cv2.COLOR_BGR2RGB)

    final_clean, cloud_mask, raw_pred = remove_clouds(
        model=model,
        device=device,
        image_input=cloud_path,
        apply_mask_blending=True
    )

    has_label = label_path is not None and os.path.exists(label_path)
    if has_label:
        label_img = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        n_cols = 4
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        n_cols = 3
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cloud_img)
    axes[0].set_title("Input (Cloudy)", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cloud_mask, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Estimated Cloud Mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(final_clean)
    axes[2].set_title("Restored (Cloud-Free)", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    if has_label:
        axes[3].imshow(label_img)
        axes[3].set_title("Ground Truth", fontsize=13, fontweight="bold")
        axes[3].axis("off")

    plt.tight_layout()

    save_dir = os.path.join(project_dir, "outputs", "comparisons")
    os.makedirs(save_dir, exist_ok=True)
    save_path = args.output if args.output is not None else os.path.join(save_dir, "comparison.png")

    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"\nComparison Figure Saved at: {save_path}")


if __name__ == "__main__":
    main()