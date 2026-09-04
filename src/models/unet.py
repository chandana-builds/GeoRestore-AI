import torch.nn as nn

from .blocks import DoubleConv, Down, Decoder


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        # ---------------- Encoder ----------------
        b = base_channels
        self.down1 = Down(in_channels, b)
        self.down2 = Down(b, b * 2)
        self.down3 = Down(b * 2, b * 4)
        self.down4 = Down(b * 4, b * 8)

        # ---------------- Bottleneck ----------------

        self.bottleneck = DoubleConv(b * 8, b * 16)

        # ---------------- Decoder ----------------

        self.up4 = Decoder(b * 16, b * 8)
        self.up3 = Decoder(b * 8, b * 4)
        self.up2 = Decoder(b * 4, b * 2)
        self.up1 = Decoder(b * 2, b)

        # ---------------- Output ----------------

        self.output = nn.Sequential(
            nn.Conv2d(b, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.output(x)