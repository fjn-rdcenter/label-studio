"""Batch retraining on all reviewed annotations of a project (replaces per-annotation episodes).

Pure functions + one training loop; the web service decides when to run it and whether to promote."""
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


def split_of(project_id: int, task_id: int) -> str:
    """Fixed per task: relabelling a task never moves it between train and val."""
    return "val" if int(hashlib.sha1(f"{project_id}:{task_id}".encode()).hexdigest(), 16) % VAL_MODULO == 0 else "train"


def digest(result: list[dict]) -> str:
    geom = [{k: v for k, v in r.items() if k != "id"} for r in result if isinstance(r, dict)]
    return hashlib.sha1(json.dumps(geom, sort_keys=True).encode()).hexdigest()


def select_samples(tasks: list[dict], project_id: int, from_name: str | None, min_quality: int) -> list[dict]:
    """One sample per task: its latest usable annotation -> {task_id, split, digest, y, task}."""
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
        out.append({"task_id": t["id"], "split": split_of(project_id, t["id"]), "digest": digest(ann["result"]),
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


def project_dice(model: nn.Module, x: torch.Tensor, ys: list[np.ndarray]) -> float | None:
    if not len(ys):
        return None
    probs, _ = learner.predict_probs(model, x)
    return float(np.mean([mean_dice_ignore(p.argmax(0).numpy(), y) for p, y in zip(probs, ys)]))


def train(base: nn.Module, x: torch.Tensor, y: torch.Tensor, anchors: tuple | None, *, steps: int, lr: float,
          eval_every: int, score_fn, amp: bool, batch: int = 4, micro_batch: int = 2,
          seed: int = 0) -> tuple[nn.Module, list[dict]]:
    """Supervised CE + soft-Dice (IGNORE-aware) on the norm_and_heads subset of a copy of `base`.
    Each step: `batch` project samples + at most as many anchors; keeps the checkpoint with the best score_fn."""
    rng = random.Random(seed)
    device = next(base.parameters()).device
    model = copy.deepcopy(base)
    learner.select_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score, best_state, history = score_fn(model), learner.trainable_state(model), []
    learner.select_trainable(model)  # score_fn put it in eval mode
    for step in range(1, steps + 1):
        pick = rng.sample(range(len(x)), min(batch, len(x)))
        bx, by = [x[pick]], [y[pick]]
        if anchors is not None and len(anchors[0]):
            idx = rng.sample(range(len(anchors[0])), min(len(pick), len(anchors[0])))
            bx.append(anchors[0][idx])
            by.append(torch.from_numpy(anchors[1][idx]))
        bx, by = torch.cat(bx), torch.cat(by)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for k in range(0, len(bx), micro_batch):  # gradient accumulation: fits a 6 GB GPU
            xb, yb = bx[k:k + micro_batch].to(device), by[k:k + micro_batch].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                (p1, _), (p2, _) = model(xb, branch="both")
            loss = sum(F.cross_entropy(p.float(), yb, ignore_index=IGNORE) + learner._soft_dice_loss(p, yb) for p in (p1, p2))
            loss = loss * len(xb) / len(bx)
            scaler.scale(loss).backward()
            total += float(loss.detach())
        scaler.step(opt)
        scaler.update()
        entry = {"step": step, "loss": round(total, 4)}
        if step % eval_every == 0 or step == steps:
            score = score_fn(model)
            learner.select_trainable(model)
            entry["score"] = round(score, 4)
            if score > best_score:
                best_score, best_state = score, learner.trainable_state(model)
        history.append(entry)
    model.load_state_dict(best_state, strict=False)
    return model.eval(), history


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
