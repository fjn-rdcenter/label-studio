"""Short-term tier: a per-well adapter (BN affine + heads) learned in a few steps from the images a physician confirmed.
It only shapes proposals inside that well and is dropped with it; the long-term tier never sees it."""
from __future__ import annotations

import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import learner, trainer
from .bundle_registry import adapter_state, is_adapter_key
from .ls_format import IGNORE

SOURCES = ("production", "adapter", "copy")  # tie order: prefer the shared model, then the well's own


@dataclass
class Sample:
    x: torch.Tensor  # (1, 256, 256) float
    y: torch.Tensor  # (256, 256) int64, IGNORE where the physician drew nothing
    changed: torch.Tensor  # (256, 256) bool: pixels the physician changed from the proposal
    prior: torch.Tensor  # (256, 256) int64: production argmax


@dataclass
class WellState:
    adapter: dict[str, torch.Tensor]
    samples: list[Sample]
    proposals: dict[str, dict[str, np.ndarray]]  # image url -> {source: 256 class map}
    scores: list[dict[str, float]]
    last_used: float


class WellAdapters:
    def __init__(self, max_wells: int = 16, ttl_s: float = 7200.0):
        self.max_wells, self.ttl_s = max_wells, ttl_s
        self._wells: OrderedDict[tuple[int, int], WellState] = OrderedDict()

    def get(self, key: tuple[int, int], init: Callable[[], dict[str, torch.Tensor]]) -> WellState:
        now = time.time()
        for k in [k for k, s in self._wells.items() if k != key and now - s.last_used > self.ttl_s]:
            del self._wells[k]
        state = self._wells.get(key)
        if state is None:
            state = self._wells[key] = WellState(init(), [], {}, [], now)
        self._wells.move_to_end(key)
        state.last_used = now
        while len(self._wells) > self.max_wells:
            self._wells.popitem(last=False)
        return state

    def clear(self) -> None:
        self._wells.clear()

    def __len__(self) -> int:
        return len(self._wells)


def adapter_loss(model: nn.Module, prod_adapter: dict, batch: list[Sample], w_changed: float = 2.0,
                 w_keep: float = 0.1, w_l2: float = 1e-3) -> torch.Tensor:
    device = next(model.parameters()).device
    x = torch.stack([s.x for s in batch]).to(device)
    y = torch.stack([s.y for s in batch]).to(device)
    weight = 1 + w_changed * torch.stack([s.changed for s in batch]).to(device).float()
    prior = torch.stack([s.prior for s in batch]).to(device)
    labeled = y != IGNORE
    (p1, _), (p2, _) = model(x, branch="both")
    total = torch.zeros((), device=device)
    for p in (p1, p2):
        ce = F.cross_entropy(p.float(), y, ignore_index=IGNORE, reduction="none")
        total = total + (ce * weight)[labeled].sum() / weight[labeled].sum().clamp(min=1) + learner._soft_dice_loss(p, y)
        if (~labeled).any():  # keep what production already said where the physician drew nothing
            total = total + w_keep * F.cross_entropy(p.float(), prior, reduction="none")[~labeled].mean()
    l2 = sum(((p - prod_adapter[n].to(p)) ** 2).sum() for n, p in model.named_parameters()
             if is_adapter_key(n) and n in prod_adapter)
    return total + w_l2 * l2


@torch.no_grad()
def _mean_dice(model: nn.Module, samples: list[Sample]) -> float:
    probs, _ = learner.predict_probs(model, torch.stack([s.x for s in samples]))
    pred = probs.argmax(1).numpy()
    return float(np.mean([trainer.mean_dice_ignore(p, s.y.numpy()) for p, s in zip(pred, samples)]))


def adapt(model: nn.Module, state: WellState, prod_adapter: dict, steps: int = 3, lr: float = 1e-4,
          max_samples: int = 32, rollback_margin: float = 0.01, seed: int = 0) -> dict:
    t0 = time.time()
    batch = state.samples[-max_samples:]
    if not batch:
        return {"loss": 0.0, "seconds": 0.0, "reverted": False, "samples": 0}
    old = {k: v.detach().clone() for k, v in state.adapter.items()}
    model.load_state_dict(old, strict=False)
    before = _mean_dice(model, batch)
    params = []
    for name, p in model.named_parameters():
        p.requires_grad_(is_adapter_key(name))
        if is_adapter_key(name):
            params.append(p)
    model.eval()  # BN running statistics stay frozen: only gamma/beta and the heads learn
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    loss = torch.zeros(())
    for _ in range(steps):
        aug = []
        for s in batch:
            xs, ys = trainer.augment(s.x[None], torch.stack([s.y, s.changed.long(), s.prior])[None], rng)
            aug.append(Sample(xs[0], ys[0, 0], ys[0, 1].bool(), ys[0, 2]))
        loss = adapter_loss(model, prod_adapter, aug)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for p in params:
        p.requires_grad_(False)
    after = _mean_dice(model, batch)
    reverted = after < before - rollback_margin
    if reverted:
        model.load_state_dict(old, strict=False)
        state.adapter = old
    else:
        state.adapter = {k: v.detach().clone() for k, v in adapter_state(model.state_dict()).items()}
    return {"loss": float(loss.detach()), "seconds": round(time.time() - t0, 3), "reverted": reverted, "samples": len(batch)}


def choose_source(scores: list[dict[str, float]], min_scored: int = 3, window: int = 5) -> str:
    if len(scores) < min_scored:
        return "production"
    recent = scores[-window:]
    return max(SOURCES, key=lambda k: float(np.mean([s.get(k, 0.0) for s in recent])))
