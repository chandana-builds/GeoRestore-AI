import os
import sys
import argparse
import cv2
import torch
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.inference.predict import load_model, remove_clouds


def main():
    parser = argparse.ArgumentParser(description="GeoRestore-AI Cloud Removal Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to input cloudy image")
    parser.add_argument("--dir", type=str, default=None, help="Directory of images to process in batch")
    parser.add_argument("--output", type=str, default=None, help="Output image file or directory")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--no_blend", action="store_true", help="Disable cloud mask soft-blending")
    parser.add_argument("--sensitivity", type=float, default=0.5, help="Cloud detection sensitivity (0.0 - 1.0)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using Device: {device}")

    # Model path resolution
    if args.model is not None:
        model_path = args.model
    else:
        best_path = os.path.join(project_dir, "outputs", "checkpoints", "best_unet.pth")
        final_path = os.path.join(project_dir, "outputs", "checkpoints", "unet_final.pth")
        model_path = best_path if os.path.exists(best_path) else final_path

    print(f"Loading Model: {model_path}")
    model = load_model(model_path, device)
    print("Model Loaded Successfully!")

    # Default output directory
    save_dir = os.path.join(project_dir, "outputs", "predictions")
    os.makedirs(save_dir, exist_ok=True)

    # Batch directory processing
    if args.dir is not None:
        if not os.path.isdir(args.dir):
            raise NotADirectoryError(f"Directory not found: {args.dir}")

        out_dir = args.output if args.output is not None else save_dir
        os.makedirs(out_dir, exist_ok=True)

        files = [
            f for f in sorted(os.listdir(args.dir))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
        ]
        print(f"\nProcessing {len(files)} images from: {args.dir}")

        for i, fname in enumerate(files):
            img_path = os.path.join(args.dir, fname)
            clean, mask, _ = remove_clouds(
                model=model,
                device=device,
                image_input=img_path,
                apply_mask_blending=(not args.no_blend),
                cloud_sensitivity=args.sensitivity
            )
            out_path = os.path.join(out_dir, f"clean_{fname}")
            cv2.imwrite(out_path, cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
            print(f"[{i + 1}/{len(files)}] Restored: {fname} -> {out_path}")

        print(f"\nBatch processing complete! All results saved in {out_dir}")
        return

    # Single image processing
    if args.image is not None:
        image_path = args.image
    else:
        # Fallback to sample
        image_path = os.path.join(project_dir, "data", "raw", "RICE1", "cloud", "450.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(project_dir, "data", "raw", "RICE1", "cloud", "0.png")

    print(f"Input Image: {image_path}")

    final_clean, cloud_mask, raw_pred = remove_clouds(
        model=model,
        device=device,
        image_input=image_path,
        apply_mask_blending=(not args.no_blend),
        cloud_sensitivity=args.sensitivity
    )

    if args.output is not None:
        save_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    else:
        save_path = os.path.join(save_dir, "prediction.png")

    cv2.imwrite(save_path, cv2.cvtColor(final_clean, cv2.COLOR_RGB2BGR))

    # Also save the cloud mask for inspection
    mask_save_path = os.path.join(save_dir, "cloud_mask.png")
    cv2.imwrite(mask_save_path, (cloud_mask * 255.0).astype(np.uint8))

    print("\nCloud Removal Completed Successfully!")
    print(f"  Restored Image : {save_path}")
    print(f"  Cloud Mask     : {mask_save_path}")


if __name__ == "__main__":
    main()