import torch
import torch.nn as nn
from .itgm import ITGM


class PixelHead(nn.Module):
    """Nhánh dự đoán phân đoạn mức điểm ảnh (Pixel-level segmentation head)."""

    def __init__(self, in_channels: int = 64, num_classes: int = 5):
        super(PixelHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.conv(f)


class TopoHead(nn.Module):
    """
    Nhánh dự đoán cấu trúc tô-pô với mô-đun ITGM:
    - Ánh xạ bản đồ đặc trưng F sang C_topo kênh cấu trúc giải phẫu qua Conv + Sigmoid.
    - Trích xuất khung xương mềm qua k vòng lặp hình thái học ITGM.
    - Thực hiện kết nối tắt nhân từng phần tử (element-wise multiplication) giữa
      bản đồ cấu trúc và khung xương mềm để bảo toàn tính liên thông giải phẫu.
    """

    def __init__(
        self,
        in_channels: int = 64,
        topo_channels: int = 4,
        k_iters: int = 10,
        erosion: str = "square",
    ):
        super(TopoHead, self).__init__()
        self.topo_channels = max(1, topo_channels)
        self.proj = nn.Conv2d(in_channels, self.topo_channels, kernel_size=1)
        self.itgm = ITGM(iterations=k_iters, erosion=erosion)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: [B, in_channels, H, W]
        Trả về T: [B, topo_channels, H, W] thuộc miền [0, 1]
        """
        struct_prob = torch.sigmoid(self.proj(f))
        soft_skeleton = self.itgm(struct_prob)
        # Phép nhân từng phần tử (skip connection multiplication) theo Hình 2(b)
        topo_map = soft_skeleton * struct_prob
        return topo_map


class DualHeads(nn.Module):
    """Kết hợp song song PixelHead và TopoHead."""

    def __init__(self, in_channels: int = 64, num_classes: int = 5, k_iters: int = 10, erosion: str = "square"):
        super(DualHeads, self).__init__()
        topo_channels = max(1, num_classes - 1)
        self.pixel_head = PixelHead(in_channels=in_channels, num_classes=num_classes)
        self.topo_head = TopoHead(
            in_channels=in_channels,
            topo_channels=topo_channels,
            k_iters=k_iters,
            erosion=erosion,
        )

    def forward(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_logits = self.pixel_head(f)
        t_skeleton = self.topo_head(f)
        return p_logits, t_skeleton