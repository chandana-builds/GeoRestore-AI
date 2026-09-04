import os
import json
import matplotlib.pyplot as plt


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    loss_file = os.path.join(project_dir, "outputs", "train_losses.json")

    if not os.path.exists(loss_file):
        print(f"Loss history file not found: {loss_file}")
        return

    with open(loss_file, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(9, 5))

    if isinstance(data, dict):
        train_losses = data.get("train_loss", [])
        val_losses = data.get("val_loss", [])
        epochs = list(range(1, len(train_losses) + 1))

        plt.plot(epochs, train_losses, marker="o", linewidth=2, label="Training Loss")
        if val_losses:
            plt.plot(epochs, val_losses, marker="s", linewidth=2, label="Validation Loss")
    else:
        epochs = list(range(1, len(data) + 1))
        plt.plot(epochs, data, marker="o", linewidth=2, label="Training Loss")

    plt.title("GeoRestore AI - Loss vs Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(project_dir, "outputs", "training_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nLoss curve saved successfully to:\n{save_path}")


if __name__ == "__main__":
    main()