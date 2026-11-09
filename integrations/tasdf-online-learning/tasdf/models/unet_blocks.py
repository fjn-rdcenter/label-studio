import torch 
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """ Double Convolution (Conv3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels : int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size = 3, padding = 1, bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size = 3, padding = 1, bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)



class Down(nn.Module):
    """Erosion Block (Downsampling): MaxPoo2d 2x2 """

    def __init__(self, in_channels:int, out_channels:int):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling: Upscale 2x -> Skip connection -> Double Conv"""

    def __init__(self, in_channels:int, out_channels:int):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """x1: Down Tensor of Decoder, x2: Skip Connection Tensor from Encoder"""

        x1 = self.up(x1)
        #Handle odd size difference 

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y > 0 or diff_x > 0:
            x1 = F.pad(
                x1,
                [
                    diff_x //2,
                    diff_x - diff_x //2,
                    diff_y //2,
                    diff_y - diff_y //2,
                ],
            )
        x = torch.cat([x2,x1], dim = 1)
        return self.conv(x)