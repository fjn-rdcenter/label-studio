"""
Metric v2 (M3-1, E-01): reports several aggregations side by side, because the paper does
not specify how Dice/Jaccard/Recall are averaged.

- pooled:    TP/FP/FN summed over all images, then per-class score (one number per class).
- per_image: per-class score per image, averaged over images where the class is present.
- pixel_acc: multi-class pixel accuracy over non-ignored pixels.
- hd95:      95th percentile of symmetric boundary distances (medpy convention: percentile of
             both directed distance sets concatenated), in pixels; NaN when pred or GT is empty.
- topology:  mean |beta0 pred - beta0 GT| (connected components, 8-conn) per class and
             mean |beta1 pred - beta1 GT| (holes) for TE.
Callers choose the resolution (256 px or original) by resizing pred/target beforehand.
"""
from __future__ import annotations

import cv2
import numpy as np

TE = 2


def _boundary(m: np.ndarray) -> np.ndarray:
    return m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    if not pred.any() or not gt.any():
        return float("nan")
    bp, bg = _boundary(pred), _boundary(gt)
    dist_to = lambda b: cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d = np.concatenate([dist_to(bg)[bp], dist_to(bp)[bg]])
    return float(np.percentile(d, 95))


def betti0(m: np.ndarray) -> int:
    return int(cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1)


def betti1(m: np.ndarray) -> int:
    # Holes = background components (4-conn, dual of 8-conn foreground) not connected to the border.
    padded = np.pad(~m, 1, constant_values=True).astype(np.uint8)
    return int(cv2.connectedComponents(padded, connectivity=4)[0] - 2)


class SegMetricsV2:
    def __init__(self, num_classes: int = 5, ignore_index: int | None = 255):
        self.c = num_classes
        self.ignore = ignore_index
        self.tp, self.fp, self.fn = (np.zeros(num_classes) for _ in range(3))
        self.correct = self.total = 0
        self.per_image: dict[str, list[list[float]]] = {k: [[] for _ in range(num_classes)] for k in ("dice", "iou", "recall", "hd95")}
        self.b0_err: list[list[int]] = [[] for _ in range(num_classes)]
        self.b1_err_te: list[int] = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """pred, target: integer label maps of the same 2-D shape."""
        valid = target != self.ignore if self.ignore is not None else np.ones_like(target, bool)
        self.correct += int(((pred == target) & valid).sum())
        self.total += int(valid.sum())
        for c in range(1, self.c):
            p, t = (pred == c) & valid, (target == c) & valid
            tp, fp, fn = float((p & t).sum()), float((p & ~t).sum()), float((~p & t).sum())
            self.tp[c] += tp; self.fp[c] += fp; self.fn[c] += fn
            if t.any():
                self.per_image["dice"][c].append(2 * tp / (2 * tp + fp + fn))
                self.per_image["iou"][c].append(tp / (tp + fp + fn))
                self.per_image["recall"][c].append(tp / (tp + fn))
                self.per_image["hd95"][c].append(hd95(p, t))
                self.b0_err[c].append(abs(betti0(p) - betti0(t)))
                if c == TE:
                    self.b1_err_te.append(abs(betti1(p) - betti1(t)))

    def compute(self) -> dict:
        cls = range(1, self.c)
        tp, fp, fn = self.tp, self.fp, self.fn
        safe = lambda num, den: float(num / den) if den > 0 else float("nan")
        pooled = {
            "dice": {c: safe(2 * tp[c], 2 * tp[c] + fp[c] + fn[c]) for c in cls},
            "iou": {c: safe(tp[c], tp[c] + fp[c] + fn[c]) for c in cls},
            "recall": {c: safe(tp[c], tp[c] + fn[c]) for c in cls},
        }
        mean = lambda xs: float(np.nanmean(xs)) if len(xs) and not np.all(np.isnan(xs)) else float("nan")
        per_image = {k: {c: mean(v[c]) for c in cls} for k, v in self.per_image.items()}
        out = {
            "pooled": pooled,
            "per_image": per_image,
            "pixel_acc": safe(self.correct, self.total),
            "beta0_abs_err": {c: mean(self.b0_err[c]) for c in cls},
            "beta1_abs_err_te": mean(self.b1_err_te),
        }
        out["pooled_mDice"] = mean([v for v in pooled["dice"].values()])
        out["per_image_mDice"] = mean([v for v in per_image["dice"].values()])
        return out
