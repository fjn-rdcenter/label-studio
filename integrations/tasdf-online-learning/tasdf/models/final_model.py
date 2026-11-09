import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_backbone import UNetBackbone
from .dual_heads import DualHeads


class TASDFSubNet(nn.Module):
    """Một nhánh mạng con độc lập N_i trong khung kiến trúc NSW của TASDF."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 5,
        k_iters: int = 10,
        erosion: str = "square",
    ):
        super(TASDFSubNet, self).__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels, base_channels=base_channels
        )
        self.heads = DualHeads(
            in_channels=base_channels,
            num_classes=num_classes,
            k_iters=k_iters,
            erosion=erosion,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        p, t = self.heads(f)
        return p, t


class TASDFNet(nn.Module):
    """
    Mô hình TASDF gồm 2 mạng con không chia sẻ trọng số (Non-Shared Weights: net1, net2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 5,
        k_iters: int = 10,
        erosion: str = "square",
    ):
        super(TASDFNet, self).__init__()
        self.num_classes = num_classes
        self.net1 = TASDFSubNet(in_channels, base_channels, num_classes, k_iters, erosion)
        self.net2 = TASDFSubNet(in_channels, base_channels, num_classes, k_iters, erosion)

    def forward(self, x: torch.Tensor, branch: str = "both"):
        if branch == "net1":
            return self.net1(x)
        elif branch == "net2":
            return self.net2(x)
        elif branch == "both":
            p1, t1 = self.net1(x)
            p2, t2 = self.net2(x)
            return (p1, t1), (p2, t2)
        elif branch == "ensemble":
            p1, t1 = self.net1(x)
            p2, t2 = self.net2(x)
            # Trung bình trên phân phối xác suất softmax để đồng nhất với Validator và Predictor
            prob_mean = (F.softmax(p1, dim=1) + F.softmax(p2, dim=1)) / 2.0
            p_ens = torch.log(torch.clamp(prob_mean, min=1e-8))
            t_ens = (t1 + t2) / 2.0
            return p_ens, t_ens
        else:
            raise ValueError(
                f"Giá trị branch không hợp lệ: {branch}. Chọn 'both', 'net1', 'net2' hoặc 'ensemble'."
            )