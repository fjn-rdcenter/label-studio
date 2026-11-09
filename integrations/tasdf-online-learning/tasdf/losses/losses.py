import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasdf.models.itgm import ITGM


CPS_TAU = 0.80   # fix M2-B: ngưỡng tin cậy của nhãn giả


def sigmoid_rampup(current_epoch: int, max_epochs: int, alpha: float = 1.0) -> float:
    """Hàm điều phối trọng số động Sigmoid Ramp-up w(t) theo Laine & Aila (2016)."""
    if max_epochs <= 0:
        return alpha
    current = np.clip(current_epoch, 0, max_epochs)
    phase = 1.0 - (current / max_epochs)
    return float(alpha * np.exp(-5.0 * phase * phase))


class MainLoss(nn.Module):
    """
    Hàm mất mát toàn thể TASDF (Mục 3.3):
    - L_sup = omega_0 * CE(P1, Y) + omega_1 * CE(P2, Y) [+ neo khung xương tô-pô chống sụp đổ nghiệm]
    - L_pc  = MSE(softmax(P1_u), softmax(P2_u))
    - L_tc  = MSE(T1_u, T2_u)
    - L_distillation = lambda * w(t) * L_tc + w(t) * L_pc
    - L_total = gamma * L_sup + (1 - gamma) * L_distillation
    """

    def __init__(
        self,
        lambda_topo: float = 0.10,
        gamma: float = 0.5,
        alpha: float = 1.0,
        max_epochs: int = 2000,
        omega_0: float = 1.0,
        omega_1: float = 1.0,
        topo_sup_weight: float = 0.10,
        k_iters: int = 10,
        ignore_index: int = 255,
        erosion: str = "square",
    ):
        super(MainLoss, self).__init__()
        self.ignore_index = ignore_index
        self.lambda_topo = lambda_topo
        self.gamma = gamma
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.omega_0 = omega_0
        self.omega_1 = omega_1
        self.topo_sup_weight = topo_sup_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.target_itgm = ITGM(iterations=k_iters, erosion=erosion)

    @torch.no_grad()
    def _extract_target_skeleton(
        self, targets: torch.Tensor, num_classes: int, target_channels: int
    ) -> torch.Tensor:
        """
        Trích xuất khung xương chuẩn từ mặt nạ nhãn Y cho các lớp tiền cảnh (1..C-1)
        để neo nhánh TopoHead không bị suy biến về 0.
        """
        valid = targets != self.ignore_index
        one_hot = F.one_hot(torch.where(valid, targets, torch.zeros_like(targets)).clamp(0, num_classes - 1), num_classes=num_classes)
        # Chuyển về [B, C, H, W] và lấy các lớp tiền cảnh 1..C-1
        one_hot_chw = one_hot.permute(0, 3, 1, 2).float()
        fg_masks = one_hot_chw[:, 1:, :, :] if num_classes > 1 else one_hot_chw

        if fg_masks.shape[1] != target_channels:
            # Tương thích ngược nếu bài test truyền t_labeled có 1 kênh gộp
            fg_union = ((targets > 0) & valid).unsqueeze(1).float()
            fg_masks = fg_union.expand(-1, target_channels, -1, -1)

        fg_masks = fg_masks * valid.unsqueeze(1).float()
        target_skel = self.target_itgm(fg_masks) * fg_masks
        return target_skel, valid.unsqueeze(1).float()

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        sup_term, unsup_term, metrics = self.terms(*args, **kwargs)
        return sup_term + unsup_term, metrics

    def terms(
        self,
        p1_labeled: torch.Tensor,
        p2_labeled: torch.Tensor,
        targets_labeled: torch.Tensor,
        p1_unlabeled: torch.Tensor,
        p2_unlabeled: torch.Tensor,
        t1_unlabeled: torch.Tensor,
        t2_unlabeled: torch.Tensor,
        current_epoch: int,
        t1_labeled: torch.Tensor | None = None,
        t2_labeled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """(gamma * L_sup, (1 - gamma) * L_distillation, metrics); forward = tổng hai phần (Eq. 12)."""
        # 1. Tổn thất có giám sát mức điểm ảnh (Phương trình 8)
        loss_ce1 = self.ce_loss(p1_labeled, targets_labeled)
        loss_ce2 = self.ce_loss(p2_labeled, targets_labeled)
        loss_sup = self.omega_0 * loss_ce1 + self.omega_1 * loss_ce2

        # Neo giám sát khung xương tô-pô trên tập có nhãn để chống Mode Collapse cho TopoHead
        loss_topo_sup_val = 0.0
        if self.topo_sup_weight > 0.0 and t1_labeled is not None and t2_labeled is not None:
            num_classes = p1_labeled.shape[1]
            target_skel, valid_w = self._extract_target_skeleton(
                targets_labeled, num_classes=num_classes, target_channels=t1_labeled.shape[1]
            )
            denom = (valid_w.sum() * t1_labeled.shape[1]).clamp_min(1.0)
            loss_topo_sup = (
                (((t1_labeled - target_skel) ** 2) * valid_w).sum() + (((t2_labeled - target_skel) ** 2) * valid_w).sum()
            ) / denom
            loss_sup = loss_sup + self.topo_sup_weight * loss_topo_sup
            loss_topo_sup_val = loss_topo_sup.item()

        # 2. Tổn thất nhất quán mức điểm ảnh trên dữ liệu không nhãn (Phương trình 9)
        # fix M2-B: CPS (Chen et al., CVPR 2021) - CE với nhãn giả cứng của mạng kia, chỉ pixel conf >= CPS_TAU
        with torch.no_grad():
            conf1, pl1 = F.softmax(p1_unlabeled.float(), dim=1).max(dim=1)
            conf2, pl2 = F.softmax(p2_unlabeled.float(), dim=1).max(dim=1)
            pl1[conf1 < CPS_TAU] = self.ignore_index
            pl2[conf2 < CPS_TAU] = self.ignore_index

        def masked_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            n = (target != self.ignore_index).sum().clamp_min(1)   # mọi pixel bị loại -> 0, không NaN
            return F.cross_entropy(logits.float(), target, ignore_index=self.ignore_index, reduction="sum") / n

        loss_pc = masked_ce(p1_unlabeled, pl2) + masked_ce(p2_unlabeled, pl1)

        # 3. Tổn thất nhất quán mức tô-pô trên dữ liệu không nhãn (Phương trình 10)
        loss_tc = F.mse_loss(t1_unlabeled, t2_unlabeled)

        # 4. Tổn thất tự chưng cất và Tổng tổn thất (Phương trình 11 & 12)
        w_t = sigmoid_rampup(current_epoch, self.max_epochs, self.alpha)
        loss_distill = self.lambda_topo * w_t * loss_tc + w_t * loss_pc

        sup_term, unsup_term = self.gamma * loss_sup, (1.0 - self.gamma) * loss_distill
        total_loss = sup_term + unsup_term

        metrics = {
            "total_loss": total_loss.item(),
            "loss_sup": loss_sup.item(),
            "loss_topo_sup": loss_topo_sup_val,
            "loss_pc": loss_pc.item(),
            "loss_tc": loss_tc.item(),
            "w_t": w_t,
            "loss_distill": loss_distill.item(),
        }
        return sup_term, unsup_term, metrics