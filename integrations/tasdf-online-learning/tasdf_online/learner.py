"""TASDF model helpers for retraining: trainable subset, evaluation, anti-forgetting gate and
weight-shift telemetry. Pure torch, no web code."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # metrics reloaded from registry.json have str keys ("1"), fresh ones have int keys
    prev_d, base_d = ({int(k): x for k, x in m["dice"].items()} for m in (prev, base))
    for c, v in cand["dice"].items():
        c = int(c)
        if prev_d.get(c, v) - v > prev_drop:
            reasons.append(f"class {c} drop vs prev {prev_d[c] - v:.4f} > {prev_drop}")
        if base_d.get(c, v) - v > base_drop:
            reasons.append(f"class {c} drop vs v0 {base_d[c] - v:.4f} > {base_drop}")
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
