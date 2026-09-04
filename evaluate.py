import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.dataset.data_loader import RiceDataset
from src.inference.predict import load_model, remove_clouds
from src.evaluation.metrics import calculate_psnr, calculate_ssim


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoRestore-AI on Test Data")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--blend", action="store_true", default=True, help="Apply cloud mask blending")
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

    print(f"Loading Model: {model_path}")
    model = load_model(model_path, device)

    # Load validation split from both RICE1 and RICE2
    rice1_path = os.path.join(project_dir, "data", "raw", "RICE1")
    rice2_path = os.path.join(project_dir, "data", "raw", "RICE2")
    dataset_paths = [p for p in [rice1_path, rice2_path] if os.path.exists(p)]

    val_dataset = RiceDataset(
        dataset_paths=dataset_paths,
        split="val",
        val_ratio=0.15,
        seed=42,
        is_train=False
    )

    num_eval = min(args.samples, len(val_dataset))
    print(f"Evaluating on {num_eval} validation / unseen test samples...\n")

    psnr_scores = []
    ssim_scores = []

    for i in range(num_eval):
        cloud_t, label_t = val_dataset[i]

        cloud_np = (cloud_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        clean_pred, _, _ = remove_clouds(
            model=model,
            device=device,
            image_input=cloud_np,
            apply_mask_blending=args.blend
        )

        pred_t = torch.from_numpy(clean_pred.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        lbl_t = label_t.unsqueeze(0)

        psnr = calculate_psnr(pred_t, lbl_t)
        ssim_val = calculate_ssim(pred_t, lbl_t)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim_val)

        if (i + 1) % 10 == 0 or (i + 1) == num_eval:
            print(f"[{i + 1}/{num_eval}] Sample {i + 1} - PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}")

    print("\n" + "=" * 50)
    print("           EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Evaluated Samples : {num_eval}")
    print(f"Mean PSNR               : {np.mean(psnr_scores):.2f} +/- {np.std(psnr_scores):.2f} dB")
    print(f"Max PSNR                : {np.max(psnr_scores):.2f} dB")
    print(f"Min PSNR                : {np.min(psnr_scores):.2f} dB")
    print(f"Mean SSIM               : {np.mean(ssim_scores):.4f} +/- {np.std(ssim_scores):.4f}")
    print(f"Max SSIM                : {np.max(ssim_scores):.4f}")
    print(f"Min SSIM                : {np.min(ssim_scores):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()