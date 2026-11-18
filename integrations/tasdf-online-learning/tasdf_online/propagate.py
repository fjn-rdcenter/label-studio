"""The four propagation rules on 256x256 masks: only carried objects, refined inside a band, stop on doubt."""
from __future__ import annotations

import cv2
import numpy as np

from .ls_format import NAME_TO_CLASS, TE, ZP, mask_to_results, results_to_mask

RING_CLASSES = {ZP, TE}


def _labels(result: dict) -> list:
    return (result.get("value") or {}).get("polygonlabels") or []


def object_class(result: dict) -> int | None:
    labels = _labels(result)
    if result.get("type") != "polygonlabels" or not labels:
        return None
    return NAME_TO_CLASS.get(labels[0])


def object_mask(result: dict, size: int = 256) -> np.ndarray:
    return results_to_mask([result], None, size) > 0


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled > 0


def _largest(mask: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """The connected component with the largest area (or the largest overlap with `weight`)."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return np.zeros(mask.shape, bool)
    scores = np.bincount(lab[weight] if weight is not None else lab.ravel(), minlength=n)
    scores[0] = 0
    if scores.max() == 0:
        return np.zeros(mask.shape, bool)
    return lab == int(scores.argmax())


def _has_hole(mask: np.ndarray) -> bool:
    return bool((_fill(mask) & ~mask).any())


def refine_object(carried: np.ndarray, model: np.ndarray, band_px: int, ring: bool) -> np.ndarray:
    k = _disk(band_px)
    core = cv2.erode(carried.astype(np.uint8), k) > 0
    zone = cv2.dilate(carried.astype(np.uint8), k) > 0
    out = _largest(core | (model & zone), carried)
    filled = _fill(out)
    if not ring:
        return filled
    hole = _largest(filled & ~out)
    return filled & ~hole


def _edge(mask: np.ndarray) -> np.ndarray:
    # outside the frame counts as background: a mask touching the border still has an edge there
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & ~(inner > 0)


def boundary_f(a: np.ndarray, b: np.ndarray, tol_px: int = 2) -> float:
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    ea, eb = _edge(a), _edge(b)
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((db[ea] <= tol_px).mean())
    recall = float((da[eb] <= tol_px).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def stop_reason(carried: np.ndarray, refined: np.ndarray, free: np.ndarray, ring: bool, threshold: float,
                tol_px: int = 2) -> str | None:
    if not refined.any():
        return "vanished"
    ratio = refined.sum() / max(int(carried.sum()), 1)
    if ratio < 0.67 or ratio > 1.5:
        return "area"
    if ring and _has_hole(carried) and not _has_hole(refined):  # only a hole that existed can be lost
        return "ring"
    n, lab = cv2.connectedComponents(free.astype(np.uint8))
    touching = np.isin(lab, np.unique(lab[carried & free]))
    if boundary_f(carried, free & touching, tol_px) < threshold:
        return "disagree"
    return None


def object_result(mask: np.ndarray, template: dict, cls: int) -> dict | None:
    if not mask.any():
        return None
    h, w = mask.shape
    label = (_labels(template) or [None])[0]
    found = mask_to_results(mask.astype(np.uint8) * cls, w, h, template.get("from_name"), template.get("to_name"),
                            {cls: label})
    if not found:
        return None

    def area(r):
        p = np.asarray(r["value"]["points"], np.float64)
        return abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))

    best = max(found, key=area)
    best["id"] = template.get("id")
    best["type"] = template.get("type", "polygonlabels")
    return best
