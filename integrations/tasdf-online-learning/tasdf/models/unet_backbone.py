import torch
import torch.nn as nn
from .unet_blocks import DoubleConv, Up, Down


class UNetBackbone(nn.Module):
    """
    Unet Backbone 4 layer
    Args:
        Input: Image.
        Output: Feature map F with 64 channels

        F will be share into Pixel Head and Topo Head
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super(UNetBackbone,self).__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        self.inc = DoubleConv(in_channels, base_channels)

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)


        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

    def forward(self, x: torch. Tensor) -> torch.Tensor:
        """
        DataFlow
        x: [B,1,H,W]
        Return F:[B,64,H,W]
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x