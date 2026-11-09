"""Online update of a TASDF model: trainable subset, correction-aware episode, outer EMA,
anti-forgetting gate and weight-shift telemetry. Pure torch, no web code."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tasdf.losses.losses import MainLoss
from tasdf.models.final_model import TASDFNet
from tasdf.utils.metrics_v2 import SegMetricsV2

NUM_CLASSES = 5
IGNORE = 255
# norm_and_heads: BN affine everywhere + last decoder stage + both heads (311,570 params at base 64)
EXPECTED_TRAINABLE_BASE64 = 311_570


def is_bn(name: str) -> bool:
    return ".double_conv.1." in name or ".double_conv.4." in name


def is_trainable(name: str) -> bool:
    return is_bn(name) or ".up4." in name or ".heads." in name


def build_model(base_channels: int = 64, erosion: str = "cross") -> TASDFNet:
    return TASDFNet(in_channels=1, base_channels=base_channels, num_classes=NUM_CLASSES, k_iters=10, erosion=erosion)


def load_state_dict(path: str) -> dict:
    # operator-provided checkpoint (trusted): full trainer checkpoint or bare state_dict
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj


def select_trainable(model: nn.Module) -> int:
    """Freeze everything except norm_and_heads; keep BN running stats frozen (eval mode)."""
    count = 0
    for name, p in model.named_parameters():
        p.requires_grad_(is_trainable(name))
        count += p.numel() if p.requires_grad else 0
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return count


def trainable_state(model: nn.Module) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_trainable(n)}


@torch.no_grad()
def predict_probs(model: nn.Module, images: torch.Tensor, batch: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """images: (N,1,H,W) float -> ensemble probs (N,C,H,W) and topo (N,4,H,W) on CPU."""
    model.eval()
    device = next(model.parameters()).device
    probs, topos = [], []
    for i in range(0, len(images), batch):
        log_p, t = model(images[i:i + batch].to(device), branch="ensemble")
        probs.append(log_p.exp().float().cpu())
        topos.append(t.float().cpu())
    return torch.cat(probs), torch.cat(topos)


def mean_dice(pred: np.ndarray, target: np.ndarray) -> tuple[float, dict]:
    """Mean Dice over foreground classes present in pred or target (single image)."""
    per = {}
    for c in range(1, NUM_CLASSES):
        p, t = pred == c, target == c
        denom = p.sum() + t.sum()
        if denom:
            per[c] = float(2 * (p & t).sum() / denom)
    return (float(np.mean(list(per.values()))) if per else 1.0), per


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = (target != IGNORE).unsqueeze(1).float()
    probs = logits.float().softmax(1)[:, 1:] * valid
    safe = torch.where(target == IGNORE, torch.zeros_like(target), target)
    onehot = F.one_hot(safe, NUM_CLASSES).permute(0, 3, 1, 2)[:, 1:].float() * valid
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (denom + 1)).mean()


def episode_loss(model, loss_fn: MainLoss, x, y, w, topo_sup_weight: float) -> tuple[torch.Tensor, dict]:
    """gamma * (W-weighted CE + Dice + topo-sup) + M2-B unlabeled term (CPS + topo consistency)."""
    (p1, t1), (p2, t2) = model(x, branch="both")
    sup = 0.0
    for p in (p1, p2):
        ce = F.cross_entropy(p.float(), y, ignore_index=IGNORE, reduction="none")
        sup = sup + (ce * w).sum() / w.sum() + _soft_dice_loss(p, y)
    topo = torch.zeros((), device=x.device)
    if topo_sup_weight > 0:
        skel, valid = loss_fn._extract_target_skeleton(y, num_classes=NUM_CLASSES, target_channels=t1.shape[1])
        denom = (valid.sum() * t1.shape[1]).clamp_min(1.0)
        topo = ((((t1.float() - skel) ** 2) * valid).sum() + (((t2.float() - skel) ** 2) * valid).sum()) / denom
        sup = sup + topo_sup_weight * topo
    # reuse M2-B unlabeled term on the same images; its own sup term is ignored
    _, unsup, m = loss_fn.terms(p1, p2, y, p1, p2, t1, t2, current_epoch=loss_fn.max_epochs)
    total = loss_fn.gamma * sup + unsup
    return total, {"total": total.item(), "sup": float(sup.detach()), "topo_sup": topo.item(), "pc": m["loss_pc"], "tc": m["loss_tc"]}


def run_episode(teacher: nn.Module, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, *, epochs: int, lr: float,
                ema_alpha: float, topo_sup_weight: float, erosion: str, amp: bool) -> tuple[nn.Module, list[dict]]:
    """Train a student copy on (new sample + replay), return EMA candidate and per-epoch losses.
    x: (B,1,H,W), y: (B,H,W) long, w: (B,H,W) pixel weights. Teacher is left untouched."""
    device = next(teacher.parameters()).device
    student = copy.deepcopy(teacher)
    select_trainable(student)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = MainLoss(gamma=0.5, topo_sup_weight=0.0, max_epochs=1, erosion=erosion).to(device)
    x, y, w = x.to(device), y.to(device), w.to(device)
    history = []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss, metrics = episode_loss(student, loss_fn, x, y, w, topo_sup_weight)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        history.append(metrics)

    candidate = copy.deepcopy(teacher)
    s_params = dict(student.named_parameters())
    with torch.no_grad():
        # per-parameter EMA keeps net1/net2 separate (names are net1.* / net2.*); frozen params stay equal
        for name, p in candidate.named_parameters():
            if is_trainable(name):
                p.mul_(ema_alpha).add_(s_params[name].detach(), alpha=1 - ema_alpha)
    candidate.eval()
    return candidate, history


def evaluate(model: nn.Module, images: torch.Tensor, masks: np.ndarray) -> dict:
    """Ensemble prediction on holdout -> pooled mDice and per-class Dice."""
    probs, _ = predict_probs(model, images)
    preds = probs.argmax(1).numpy().astype(np.uint8)
    metric = SegMetricsV2(num_classes=NUM_CLASSES, ignore_index=IGNORE)
    for p, t in zip(preds, masks):
        metric.update(p, t)
    out = metric.compute()
    dice = {int(c): v for c, v in out["pooled"]["dice"].items()}
    return {"pooled_mDice": out["pooled_mDice"], "dice": dice}


def gate(cand: dict, prev: dict, base: dict, prev_min=-0.005, prev_drop=0.05, base_min=-0.05, base_drop=0.08):
    """Four anti-forgetting inequalities (spec section 3.3, component 5)."""
    reasons = []
    d_prev = cand["pooled_mDice"] - prev["pooled_mDice"]
    d_base = cand["pooled_mDice"] - base["pooled_mDice"]
    if d_prev < prev_min:
        reasons.append(f"mDice vs prev {d_prev:+.4f} < {prev_min}")
    if d_base < base_min:
        reasons.append(f"mDice vs v0 {d_base:+.4f} < {base_min}")
    for c, v in cand["dice"].items():
        if prev["dice"].get(c, v) - v > prev_drop:
            reasons.append(f"class {c} drop vs prev {prev['dice'][c] - v:.4f} > {prev_drop}")
        if base["dice"].get(c, v) - v > base_drop:
            reasons.append(f"class {c} drop vs v0 {base['dice'][c] - v:.4f} > {base_drop}")
    return not reasons, reasons


GROUPS = ("encoder_frozen", "decoder_frozen", "bn_affine", "decoder_up4", "pixel_head", "topo_head")


def group_of(name: str) -> str:
    if is_bn(name):
        return "bn_affine"
    if ".heads.pixel_head." in name:
        return "pixel_head"
    if ".heads.topo_head." in name:
        return "topo_head"
    if ".up4." in name:
        return "decoder_up4"
    if any(f".{b}." in name for b in ("up1", "up2", "up3")):  # includes ConvTranspose (.up.) weights
        return "decoder_frozen"
    return "encoder_frozen"


def layer_shifts(before: nn.Module, after: nn.Module, base: dict) -> dict:
    """Per-group L2 shift vs previous version, relative %, cumulative drift vs v0, pixel-head cosine drift."""
    acc = {g: {"d2": 0.0, "n2": 0.0, "b2": 0.0} for g in GROUPS}
    prev = dict(before.named_parameters())
    pix_now, pix_base = [], []
    for name, p in after.named_parameters():
        g = group_of(name)
        now, old, ref = p.detach().float().cpu(), prev[name].detach().float().cpu(), base[name].float()
        acc[g]["d2"] += float(((now - old) ** 2).sum())
        acc[g]["n2"] += float((old ** 2).sum())
        acc[g]["b2"] += float(((now - ref) ** 2).sum())
        if g == "pixel_head":
            pix_now.append(now.flatten()); pix_base.append(ref.flatten())
    out = {g: {"l2_shift": math.sqrt(v["d2"]), "rel_shift_pct": 100 * math.sqrt(v["d2"]) / (math.sqrt(v["n2"]) + 1e-12),
               "drift_from_v0": math.sqrt(v["b2"])} for g, v in acc.items()}
    a, b = torch.cat(pix_now), torch.cat(pix_base)
    out["pixel_head_cosine_drift"] = float(1 - F.cosine_similarity(a, b, dim=0))
    return out


def local_probe(before_probs: torch.Tensor, after_probs: torch.Tensor, y: np.ndarray) -> dict:
    """Fit on the just-trained sample (NOT generalisation): Dice before/after, absorption of corrected pixels."""
    pb, pa = before_probs.argmax(0).numpy(), after_probs.argmax(0).numpy()
    wrong = pb != y
    absorbed = int((pa[wrong] == y[wrong]).sum())
    return {
        "dice_before": mean_dice(pb, y)[0],
        "dice_after": mean_dice(pa, y)[0],
        "corrected_pixels": int(wrong.sum()),
        "absorbed_pixels": absorbed,
        "absorption_rate_pct": 100.0 * absorbed / max(1, int(wrong.sum())),
        "mean_prob_shift": float(0.5 * (after_probs - before_probs).abs().sum(0).mean()),
    }
