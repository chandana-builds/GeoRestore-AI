import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.unet import UNet

print("Creating U-Net...")
model = UNet()
print("Model created successfully!")

# Test standard resolution
x1 = torch.randn(1, 3, 512, 512)
y1 = model(x1)
print(f"Standard Input: {x1.shape} -> Output: {y1.shape} (value range: [{y1.min():.2f}, {y1.max():.2f}])")

# Test arbitrary odd resolution (unseen size)
x2 = torch.randn(1, 3, 373, 519)
y2 = model(x2)
print(f"Arbitrary Input: {x2.shape} -> Output: {y2.shape} (value range: [{y2.min():.2f}, {y2.max():.2f}])")

print("\nAll model architecture tests passed!")
