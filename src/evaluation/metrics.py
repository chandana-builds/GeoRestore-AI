import torch
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(prediction, target):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    """

    mse = F.mse_loss(prediction, target)

    if mse == 0:
        return 100

    return 20 * log10(1.0 / torch.sqrt(mse).item())


def calculate_ssim(prediction, target):
    """
    Calculate Structural Similarity Index (SSIM)
    """

    prediction = prediction.squeeze().permute(1, 2, 0).cpu().numpy()
    target = target.squeeze().permute(1, 2, 0).cpu().numpy()

    score = ssim(
        prediction,
        target,
        channel_axis=2,
        data_range=1.0
    )

    return score