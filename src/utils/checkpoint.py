import torch
import os


def save_checkpoint(model, optimizer, epoch, loss, save_path="checkpoints/model.pth"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    torch.save(checkpoint, save_path)

    print(f"Checkpoint saved to: {save_path}")