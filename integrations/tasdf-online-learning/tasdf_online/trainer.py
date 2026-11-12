"""Batch retraining on all reviewed annotations of a project (replaces per-annotation episodes).

Pure functions + one training loop; the web service decides when to run it and whether to promote."""
from __future__ import annotations

import copy
import hashlib
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import learner
from .ls_format import IGNORE, annotation_target

VAL_MODULO = 5  # ~20% of tasks are project validation, never trained on


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
