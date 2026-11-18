"""A fixed-size window around the drawn regions: the model sees the embryo, not the whole well."""
from __future__ import annotations

import copy

import numpy as np

from .ls_format import preprocess

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in original-image pixels


def _points(results: list[dict]):
    for r in results or []:
        value = r.get("value") or {}
        yield from value.get("points") or []


def crop_box(results: list[dict], w: int, h: int, size: int = 600) -> Box:
    pts = np.asarray(list(_points(results)), dtype=np.float64).reshape(-1, 2)
    if len(pts):
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2 * w / 100
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2 * h / 100
    else:
        cx, cy = w / 2, h / 2
    sw, sh = min(size, w), min(size, h)
    x0 = int(min(max(round(cx - sw / 2), 0), w - sw))
    y0 = int(min(max(round(cy - sh / 2), 0), h - sh))
    return x0, y0, x0 + sw, y0 + sh


def _move(results: list[dict], fx, fy) -> list[dict]:
    out = []
    for r in results or []:
        value = r.get("value") or {}
        if not value.get("points"):
            out.append(r)
            continue
        r = copy.deepcopy(r)
        v = r["value"]
        v["points"] = [[fx(x), fy(y)] for x, y in v["points"]]
        if v.get("holes"):
            v["holes"] = [[[fx(x), fy(y)] for x, y in hole] for hole in v["holes"]]
        out.append(r)
    return out


def to_box(results: list[dict], w: int, h: int, box: Box) -> list[dict]:
    """% of the original image -> % of the box (shapes with points only; the rest is returned as is)."""
    x0, y0, x1, y1 = box
    return _move(results, lambda x: (x * w / 100 - x0) * 100 / (x1 - x0), lambda y: (y * h / 100 - y0) * 100 / (y1 - y0))


def from_box(results: list[dict], w: int, h: int, box: Box) -> list[dict]:
    """% of the box -> % of the original image."""
    x0, y0, x1, y1 = box
    out = _move(results, lambda x: (x * (x1 - x0) / 100 + x0) * 100 / w, lambda y: (y * (y1 - y0) / 100 + y0) * 100 / h)
    for r in out:
        if (r.get("value") or {}).get("points"):
            r["original_width"], r["original_height"] = w, h
    return out


def crop_input(gray: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return preprocess(gray[y0:y1, x0:x1])
