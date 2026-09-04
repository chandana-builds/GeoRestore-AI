import os
import sys
import json
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.dataset.data_loader import RiceDataset
from src.models.unet import UNet
from src.training.loss import ReconstructionLoss
from src.training.optimizer import get_optimizer
from src.training.train import train_one_epoch
from src.training.validate import validate_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Train GeoRestore-AI Cloud Removal UNet")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    args = parser.parse_args()

    # ==========================================================
    # Device
    # ==========================================================
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)

    # ==========================================================
    # Datasets Path (RICE1 + RICE2)
    # ==========================================================
    rice1_path = os.path.join(project_dir, "data", "raw", "RICE1")
    rice2_path = os.path.join(project_dir, "data", "raw", "RICE2")

    dataset_paths = []
    if os.path.exists(rice1_path):
        dataset_paths.append(rice1_path)
    if os.path.exists(rice2_path):
        dataset_paths.append(rice2_path)

    print(f"Dataset Paths Found: {dataset_paths}", flush=True)

    # Create Train & Validation Datasets with Augmentations (256x256 for rapid convergence)
    train_dataset = RiceDataset(
        dataset_paths=dataset_paths,
        split="train",
        val_ratio=0.15,
        seed=42,
        is_train=True,
        img_size=(256, 256)
    )

    val_dataset = RiceDataset(
        dataset_paths=dataset_paths,
        split="val",
        val_ratio=0.15,
        seed=42,
        is_train=False,
        img_size=(256, 256)
    )

    print(f"Total Train Samples: {len(train_dataset)}", flush=True)
    print(f"Total Validation Samples: {len(val_dataset)}", flush=True)


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # ==========================================================
    # Model Architecture
    # ==========================================================
    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)

    # ==========================================================
    # Loss Function & Optimizer
    # ==========================================================
    loss_fn = ReconstructionLoss(l1_weight=1.0, ssim_weight=0.5, edge_weight=0.2).to(device)
    optimizer = get_optimizer(model, learning_rate=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ==========================================================
    # Checkpoints Directory
    # ==========================================================
    checkpoint_dir = os.path.join(project_dir, "outputs", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_unet.pth")
    final_checkpoint_path = os.path.join(checkpoint_dir, "unet_final.pth")

    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": args.epochs,
        "best_epoch": 1,
        "best_val_loss": float("inf")
    }

    print("\nStarting Training Pipeline...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}] (LR: {optimizer.param_groups[0]['lr']:.6f})")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch Summary -> Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # Save best model
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  >>> Best model saved! (Val Loss: {val_loss:.5f})")

    # ==========================================================
    # Save Final Checkpoint & Metrics History
    # ==========================================================
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"\nFinal Checkpoint Saved: {final_checkpoint_path}")

    loss_file = os.path.join(project_dir, "outputs", "train_losses.json")
    with open(loss_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training History Saved: {loss_file}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), history["train_loss"], label="Train Loss", marker="o")
    plt.plot(range(1, args.epochs + 1), history["val_loss"], label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GeoRestore AI - Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    loss_plot_path = os.path.join(project_dir, "outputs", "training_loss.png")
    plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Loss Plot Saved: {loss_plot_path}")

    print("\nTraining Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()