"""Long-term retraining on all reviewed annotations: sample selection, content-hash split, epoch trainer,
BN recalibration and the promotion decision. Pure functions; the web service decides when to run a job."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import learner
from .ls_format import IGNORE, annotation_target

VAL_MODULO = 5  # ~20% of tasks are project validation, never trained on


def image_split(img: np.ndarray) -> str:
    """Fixed per image content: the same picture in two projects always lands on the same side."""
    digest_ = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()
    return "val" if int(digest_, 16) % VAL_MODULO == 0 else "train"


def digest(result: list[dict]) -> str:
    geom = [{k: v for k, v in r.items() if k != "id"} for r in result if isinstance(r, dict)]
    return hashlib.sha1(json.dumps(geom, sort_keys=True).encode()).hexdigest()


def select_samples(tasks: list[dict], project_id: int, from_name: str | None, min_quality: int) -> list[dict]:
    """One sample per task: its latest usable annotation -> {task_id, project, digest, y, task}.
    The train/val side is decided later from the image content (image_split)."""
    out = []
    for t in tasks:
        usable = [a for a in t.get("annotations") or []
                  if not a.get("was_cancelled") and a.get("result") and int(a.get("quality_level") or 1) >= min_quality]
        if not usable:
            continue
        ann = max(usable, key=lambda a: (a.get("updated_at") or "", a.get("id") or 0))
        y = annotation_target(ann["result"], from_name)
        if not ((y != 0) & (y != IGNORE)).any():
            continue  # nothing drawn that we understand
        out.append({"task_id": t["id"], "project": project_id, "digest": digest(ann["result"]),
                    "annotation_id": ann.get("id"), "y": y, "task": t})
    return out


def mean_dice_ignore(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean Dice over foreground classes present in pred or target, ignoring IGNORE pixels."""
    valid = y != IGNORE
    per = []
    for c in range(1, learner.NUM_CLASSES):
        p, t = (pred == c) & valid, (y == c)
        if p.sum() + t.sum():
            per.append(2 * (p & t).sum() / (p.sum() + t.sum()))
    return float(np.mean(per)) if per else 1.0


def mean_dice_per_image(model: nn.Module, x: torch.Tensor, ys: list[np.ndarray]) -> list[float]:
    if not len(ys):
        return []
    probs, _ = learner.predict_probs(model, x)
    return [mean_dice_ignore(p.argmax(0).numpy(), y) for p, y in zip(probs, ys)]


def project_dice(model: nn.Module, x: torch.Tensor, ys: list[np.ndarray]) -> float | None:
    per = mean_dice_per_image(model, x, ys)
    return float(np.mean(per)) if per else None


def augment(x: torch.Tensor, y: torch.Tensor, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Flips, 90-degree rotations and +-15% brightness/contrast; labels follow the geometry."""
    if rng.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if rng.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = rng.randrange(4)
    if k:
        x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    gain, bias = 1 + rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15) * 0.5
    return ((x - 0.5) * gain + 0.5 + bias).clamp(0, 1), y


def train_epochs(base: nn.Module, x: torch.Tensor, y: torch.Tensor, anchors: tuple | None, *, mode: str, epochs: int,
                 lr: float, amp: bool, score_fn=None, patience: int = 5, anchor_frac: float = 0.25, batch: int = 8,
                 micro_batch: int = 2, max_seconds: float | None = None, seed: int = 0) -> tuple[nn.Module, list[dict]]:
    """Epoch-based supervised training (CE + soft-Dice, IGNORE-aware) of the `mode` subset of a copy of `base`.

    Each optimizer step uses `batch` images: project samples plus at most `anchor_frac` anchors.
    With `score_fn`, keeps the best-scoring epoch and stops after `patience` epochs without gain;
    without it, returns the last epoch (fixed budget: no selection on held-out data)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = next(base.parameters()).device
    model = copy.deepcopy(base)
    learner.select_trainable(model, mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n_anchor = int(batch * anchor_frac) if anchors is not None and len(anchors[0]) else 0
    per_step = batch - n_anchor
    steps_per_epoch = max(1, math.ceil(len(x) / per_step))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score = score_fn(model) if score_fn else None
    best_state, since_best, history, t0 = learner.trainable_state(model, mode), 0, [], time.time()
    learner.select_trainable(model, mode)
    for epoch in range(1, epochs + 1):
        order = list(range(len(x)))
        rng.shuffle(order)
        total = 0.0
        for s in range(steps_per_epoch):
            pick = order[s * per_step:(s + 1) * per_step] or rng.sample(range(len(x)), min(per_step, len(x)))
            bx, by = [x[pick]], [y[pick]]
            if n_anchor:
                idx = rng.sample(range(len(anchors[0])), min(n_anchor, len(anchors[0])))
                bx.append(anchors[0][idx])
                by.append(torch.as_tensor(anchors[1][idx]))
            bx, by = torch.cat(bx), torch.cat(by)
            opt.zero_grad(set_to_none=True)
            for k in range(0, len(bx), micro_batch):
                xb, yb = augment(bx[k:k + micro_batch], by[k:k + micro_batch], rng)
                xb, yb = xb.to(device), yb.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    (p1, _), (p2, _) = model(xb, branch="both")
                loss = sum(F.cross_entropy(p.float(), yb, ignore_index=IGNORE) + learner._soft_dice_loss(p, yb) for p in (p1, p2))
                loss = loss * len(xb) / len(bx)
                scaler.scale(loss).backward()
                total += float(loss.detach())
            scaler.step(opt)
            scaler.update()
            sched.step()
        entry = {"epoch": epoch, "loss": round(total / steps_per_epoch, 4), "seconds": round(time.time() - t0, 1)}
        if score_fn:
            score = score_fn(model)
            learner.select_trainable(model, mode)
            entry["score"] = round(score, 4)
            if score > best_score:
                best_score, best_state, since_best = score, learner.trainable_state(model, mode), 0
            else:
                since_best += 1
        history.append(entry)
        if (score_fn and since_best >= patience) or (max_seconds and time.time() - t0 > max_seconds):
            break
    if score_fn:
        model.load_state_dict(best_state, strict=False)
    return model.eval(), history


@torch.no_grad()
def recalibrate_bn(model: nn.Module, x: torch.Tensor, batch: int = 8) -> None:
    """Re-estimate BatchNorm running statistics on this stage's images (cumulative average, no gradient)."""
    device = next(model.parameters()).device
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for i in range(0, len(x), batch):
        model(x[i:i + batch].to(device), branch="both")
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


def bootstrap_std(deltas: list[float], n: int = 1000, seed: int = 0) -> float:
    """Std of the mean of per-image Dice differences: how much of a gain is just which images we drew."""
    if len(deltas) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=np.float64)
    return float(rng.choice(d, size=(n, len(d)), replace=True).mean(1).std())


def promotion_decision(*, cand_val: list[float], prod_val: list[float], cand_train: float | None,
                       d10_ok: bool, d10_reasons: list[str], encoder_moved: bool, val_min: int, min_gain: float,
                       max_gap: float, small_val_tolerance: float = 0.01) -> tuple[bool, list[str], str | None]:
    """Promote only on a real, held-out gain: beyond bootstrap noise, without memorization, without forgetting.
    cand_val/prod_val are per-image Dice of candidate and production on the same never-trained images."""
    reasons, note = list(d10_reasons) if not d10_ok else [], None
    if encoder_moved:
        reasons.append("encoder weights changed")
    if cand_val:
        deltas = [c - p for c, p in zip(cand_val, prod_val)]
        gain = float(np.mean(deltas))
        if len(cand_val) >= val_min:
            margin = max(min_gain, 2 * bootstrap_std(deltas))
            if gain < margin:
                reasons.append(f"val gain {gain:+.4f} < required {margin:.4f} (noise-aware)")
        else:
            note = f"insufficient_val ({len(cand_val)} < {val_min})"
            if gain < -small_val_tolerance:
                reasons.append(f"val dropped {gain:+.4f}")
        if cand_train is not None and cand_train - float(np.mean(cand_val)) > max_gap:
            reasons.append(f"train-val gap {cand_train - float(np.mean(cand_val)):.4f} > {max_gap} (memorization)")
    else:
        note = "no_val"
    return not reasons, reasons, note
