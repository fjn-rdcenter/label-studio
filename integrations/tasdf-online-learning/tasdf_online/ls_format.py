"""Label Studio polygon/ring results <-> TASDF class maps, and image loading.

Result format follows the Fujinet editor fork (web/libs/editor/src/regions/PolygonRegion.jsx
serialize): value = {points: [[x%, y%], ...], closed, ring?, outerclosed?, holes?: [[[x%, y%], ...]]}.
The fork supports exactly one hole per ring.
"""
from __future__ import annotations

import io
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

BG, ZP, TE, ICM, BC = 0, 1, 2, 3, 4
CLASS_NAMES = {ZP: "ZP", TE: "TE", ICM: "ICM", BC: "BC"}
NAME_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}
# painter order: inner anatomy overwrites outer shells
PAINT_ORDER = (ZP, TE, BC, ICM)
SIZE = 256
MIN_AREA_PX = 12  # at SIZE resolution; smaller components are noise


def _to_px(points, w: int, h: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.round(pts * [w / 100.0, h / 100.0]).astype(np.int32)


def _holes(value: dict) -> list:
    holes = value.get("holes") or []
    # stored as [[[x, y], ...]]; editor also accepts the flat [[x, y], ...] form
    if holes and isinstance(holes[0], (list, tuple)) and holes[0] and isinstance(holes[0][0], (list, tuple)):
        return list(holes)
    return [holes] if holes else []


def results_to_mask(results: list[dict], from_name: str | None = None, size: int = SIZE) -> np.ndarray:
    """Rasterise polygonlabels results into a uint8 class map of shape (size, size)."""
    layers = {c: np.zeros((size, size), np.uint8) for c in PAINT_ORDER}
    for r in results or []:
        if r.get("type") != "polygonlabels" or (from_name and r.get("from_name") != from_name):
            continue
        value = r.get("value") or {}
        labels = value.get("polygonlabels") or []
        cls = NAME_TO_CLASS.get(labels[0]) if labels else None
        points = value.get("points") or []
        if cls is None or len(points) < 3:
            continue
        shape = np.zeros((size, size), np.uint8)
        cv2.fillPoly(shape, [_to_px(points, size, size)], 1)
        for hole in _holes(value):
            if len(hole) >= 3:
                pts = _to_px(hole, size, size)
                cv2.fillPoly(shape, [pts], 0)
                cv2.polylines(shape, [pts], True, 1)  # the hole outline belongs to the ring (as the outer one does)
        layers[cls] |= shape
    mask = np.zeros((size, size), np.uint8)
    for cls in PAINT_ORDER:
        mask[layers[cls] > 0] = cls
    return mask


def _simplify(contour: np.ndarray) -> np.ndarray:
    # same rule as opencv_segmentation/service.py::_simplify_contour (epsilon = max(1.5, 0.3% perimeter))
    eps = max(1.5, 0.003 * cv2.arcLength(contour, True))
    return cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)


def _to_perc(contour: np.ndarray, size: int) -> list[list[float]]:
    return [[round(float(x) * 100.0 / size, 3), round(float(y) * 100.0 / size, 3)] for x, y in contour]


def mask_to_results(mask: np.ndarray, orig_w: int, orig_h: int, from_name: str, to_name: str) -> list[dict]:
    """Class map (size x size) -> polygonlabels results; rings keep their largest hole."""
    size = mask.shape[0]
    results = []
    for cls, name in CLASS_NAMES.items():
        # RETR_CCOMP: level-1 contours are outer boundaries, their children are the holes of that class
        contours, hierarchy = cv2.findContours((mask == cls).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1 or cv2.contourArea(contour) < MIN_AREA_PX:
                continue  # holes are handled with their parent; skip specks
            outer = _simplify(contour)
            if len(outer) < 3:
                continue
            value = {"points": _to_perc(outer, size), "closed": True, "polygonlabels": [name]}
            inside = [contours[j] for j in range(len(contours)) if hierarchy[0][j][3] == i]
            if inside and cv2.contourArea(max(inside, key=cv2.contourArea)) >= MIN_AREA_PX:
                hole = _simplify(max(inside, key=cv2.contourArea))
                if len(hole) >= 3:
                    value.update(ring=True, outerclosed=True, holes=[_to_perc(hole, size)])
            results.append({
                "from_name": from_name, "to_name": to_name, "type": "polygonlabels",
                "original_width": int(orig_w), "original_height": int(orig_h), "image_rotation": 0,
                "value": value,
            })
    return results


def ls_media_url(ls_url: str, source: str) -> str:
    """Only Label Studio-relative media paths are fetched (no arbitrary hosts -> no SSRF)."""
    parsed = urlparse(source)
    if parsed.scheme or parsed.netloc:
        if not ls_url or f"{parsed.scheme}://{parsed.netloc}" != ls_url:
            raise ValueError(f"refusing to fetch image from foreign host: {source}")
        source = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    if not source.startswith("/data/"):
        raise ValueError(f"unsupported image path: {source}")
    return ls_url + source


def fetch_image(ls_url: str, token: str, source: str, timeout: float = 30) -> tuple[np.ndarray, int, int]:
    """Download a task image from Label Studio -> (float32 [0,1] SIZE x SIZE gray, orig_w, orig_h)."""
    resp = requests.get(ls_media_url(ls_url, source), headers={"Authorization": f"Token {token}"}, timeout=timeout)
    resp.raise_for_status()
    try:
        # PIL "L" conversion, as in tasdf BaseEmbryoDataset.load_raw_grayscale
        gray = np.array(Image.open(io.BytesIO(resp.content)).convert("L"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ValueError(f"not an image: {source}") from exc
    return preprocess(gray), gray.shape[1], gray.shape[0]


def preprocess(gray_uint8: np.ndarray) -> np.ndarray:
    # matches EmbryoTransformPipeline(is_train=False) with denoise disabled (M2-B config)
    img = cv2.resize(gray_uint8, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0
