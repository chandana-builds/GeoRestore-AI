import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size, sigma):
    gauss = torch.tensor([
        -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)
        for x in range(window_size)
    ]).exp()
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filters for edge preservation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        # Convert RGB to grayscale approximation for edge detection
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return loss_x + loss_y


class ReconstructionLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, edge_weight=0.2, window_size=11, channels=3):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.window_size = window_size
        self.channels = channels

        self.l1 = nn.L1Loss()
        self.edge = EdgeLoss()

        # Register SSIM window once as a buffer so it stays on GPU with zero overhead
        window = create_window(window_size, channels)
        self.register_buffer("window", window)

    def ssim_loss(self, img1, img2):
        channel = self.channels
        window = self.window

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()

    def forward(self, prediction, target):
        loss_l1 = self.l1(prediction, target)
        loss_ssim = self.ssim_loss(prediction, target)
        loss_edge = self.edge(prediction, target)

        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.edge_weight * loss_edge
        )
        return total_loss