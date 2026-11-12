"""Label Studio polygon/ring results <-> TASDF class maps, and image loading.

Result format follows the Fujinet editor fork (web/libs/editor/src/regions/PolygonRegion.jsx
serialize): value = {points: [[x%, y%], ...], closed, ring?, outerclosed?, holes?: [[[x%, y%], ...]]}.
The fork supports exactly one hole per ring.
"""
from __future__ import annotations

import base64
import io
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

BG, ZP, TE, ICM, BC = 0, 1, 2, 3, 4
CLASS_NAMES = {ZP: "ZP", TE: "TE", ICM: "ICM", BC: "BC"}
NAME_TO_CLASS = {**{v: k for k, v in CLASS_NAMES.items()}, "Blastocoel": BC}
# painter order: inner anatomy overwrites outer shells
PAINT_ORDER = (ZP, TE, BC, ICM)
SIZE = 256
MIN_AREA_PX = 12  # at SIZE resolution; smaller components are noise
# 4 px: ~25 vertices/region at IoU 0.95 on project 37 (3 px: ~30 at 0.96, 5 px: ~22 at 0.94)
# shortcut: one fixed tolerance for every class/project; make it a setting when projects need different trade-offs
SIMPLIFY_TOL_PX = 4.0  # max distance (px of the mask given to mask_to_results) between true boundary and polygon
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def _to_px(points, w: int, h: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.round(pts * [w / 100.0, h / 100.0]).astype(np.int32)


def _holes(value: dict) -> list:
    holes = value.get("holes") or []
    # stored as [[[x, y], ...]]; editor also accepts the flat [[x, y], ...] form
    if holes and isinstance(holes[0], (list, tuple)) and holes[0] and isinstance(holes[0][0], (list, tuple)):
        return list(holes)
    return [holes] if holes else []


def _decode_rle_mask(rle: list[int], width: int, height: int, size: int = SIZE) -> np.ndarray:
    if not rle or width <= 0 or height <= 0:
        return np.zeros((size, size), np.uint8)
    bits = "".join(f"{int(b) & 0xFF:08b}" for b in rle)
    pos = [0]

    def read(n: int) -> int:
        val = int(bits[pos[0] : pos[0] + n] or "0", 2)
        pos[0] += n
        return val

    num = read(32)
    word_size = read(5) + 1
    rle_sizes = [read(4) + 1 for _ in range(4)]
    out = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num and pos[0] < len(bits):
        x = read(1)
        j = min(num, i + 1 + read(rle_sizes[read(2)]))
        if x:
            out[i:j] = read(word_size)
            i = j
        else:
            while i < j:
                out[i] = read(word_size)
                i += 1
    alpha = out.reshape(height, width, 4)[:, :, 3] if num == height * width * 4 else out.reshape(height, width)
    if (height, width) != (size, size):
        alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_NEAREST)
    return (alpha > 0).astype(np.uint8)


def results_to_mask(results: list[dict], from_name: str | None = None, size: int = SIZE) -> np.ndarray:
    """Rasterise polygon/ring/ellipse/rectangle/brush results into a uint8 class map of shape (size, size)."""
    layers = {c: np.zeros((size, size), np.uint8) for c in PAINT_ORDER}
    for r in results or []:
        rtype = r.get("type")
        if from_name and rtype == "polygonlabels" and r.get("from_name") and r.get("from_name") != from_name:
            continue
        value = r.get("value") or {}
        labels = (
            value.get("polygonlabels")
            or value.get("brushlabels")
            or value.get("ellipsebasedpolygonlabels")
            or value.get("rectanglelabels")
            or value.get("labels")
            or []
        )
        cls = NAME_TO_CLASS.get(labels[0]) if labels else None
        if cls is None:
            continue
        shape = np.zeros((size, size), np.uint8)
        if rtype in {"polygonlabels", "ellipsebasedpolygonlabels"}:
            points = value.get("points") or []
            if len(points) < 3:
                continue
            cv2.fillPoly(shape, [_to_px(points, size, size)], 1)
            for hole in _holes(value):
                if len(hole) >= 3:
                    pts = _to_px(hole, size, size)
                    cv2.fillPoly(shape, [pts], 0)
                    cv2.polylines(shape, [pts], True, 1)
        elif rtype == "brushlabels" and value.get("rle"):
            ow = int(r.get("original_width") or size)
            oh = int(r.get("original_height") or size)
            shape = _decode_rle_mask(value["rle"], ow, oh, size)
        elif rtype == "rectanglelabels" and "x" in value and "y" in value:
            x1 = int(round(float(value["x"]) * size / 100.0))
            y1 = int(round(float(value["y"]) * size / 100.0))
            x2 = int(round((float(value["x"]) + float(value.get("width", 0))) * size / 100.0))
            y2 = int(round((float(value["y"]) + float(value.get("height", 0))) * size / 100.0))
            cv2.rectangle(shape, (x1, y1), (x2, y2), 1, -1)
        else:
            continue
        layers[cls] |= shape
    mask = np.zeros((size, size), np.uint8)
    for cls in PAINT_ORDER:
        mask[layers[cls] > 0] = cls
    return mask


IGNORE = 255
SHAPE_TYPES = {"polygonlabels", "ellipsebasedpolygonlabels", "brushlabels"}  # rectangles/keypoints are prompts, not shapes


def annotation_target(results: list[dict], from_name: str | None = None, size: int = SIZE) -> np.ndarray:
    """Training target from an annotation. Unlabelled pixels are background only outside the drawn ZP;
    inside the ZP (or everywhere when no ZP was drawn) they are IGNORE, so a class the annotator skipped
    is never taught as background."""
    shapes = [r for r in results or [] if r.get("type") in SHAPE_TYPES]
    y = results_to_mask(shapes, from_name, size)
    # ZP extent: the filled outer boundary (holes dropped), so the whole embryo counts as "inside"
    zp_outer = [{**r, "value": {k: v for k, v in (r.get("value") or {}).items() if k != "holes"}} for r in shapes]
    zp = (results_to_mask(zp_outer, from_name, size) == ZP).astype(np.uint8)
    contours, _ = cv2.findContours(zp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside = np.zeros_like(zp)
    cv2.drawContours(inside, contours, -1, 1, -1)  # also closes brush-painted ZP rings
    y[(y == BG) & ((inside > 0) | (not contours))] = IGNORE
    return y


def bio_valid(mask: np.ndarray) -> bool:
    """Reject empty foreground masks or single-class foreground saturation (>=95% of image area)."""
    fg = mask[mask > 0]
    if fg.size == 0:
        return False
    return int(np.bincount(fg).max()) < int(0.95 * mask.size)


def _simplify(contour: np.ndarray) -> np.ndarray:
    # Douglas-Peucker: every boundary point stays within SIMPLIFY_TOL_PX of the polygon. Measured on project 37,
    # DP beats an optimal min-vertex (Imai-Iri) polygon on IoU at equal vertex count, at ~1 ms instead of ~200 ms.
    return cv2.approxPolyDP(contour, SIMPLIFY_TOL_PX, True).reshape(-1, 2)


def _to_perc(contour: np.ndarray, w: int, h: int) -> list[list[float]]:
    return [[round(float(x) * 100.0 / w, 3), round(float(y) * 100.0 / h, 3)] for x, y in contour]


def mask_to_results(mask: np.ndarray, orig_w: int, orig_h: int, from_name: str, to_name: str,
                    names: dict = CLASS_NAMES) -> list[dict]:
    """Class map (H x W, ideally the original image size) -> polygonlabels results; rings keep their largest hole."""
    h, w = mask.shape
    min_area = MIN_AREA_PX * (w * h) / (SIZE * SIZE)  # same speck size whatever the mask resolution
    results = []
    for cls, name in names.items():
        # RETR_CCOMP: level-1 contours are outer boundaries, their children are the holes of that class
        contours, hierarchy = cv2.findContours((mask == cls).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1 or cv2.contourArea(contour) < min_area:
                continue  # holes are handled with their parent; skip specks
            outer = _simplify(contour)
            if len(outer) < 3:
                continue
            value = {"points": _to_perc(outer, w, h), "closed": True, "polygonlabels": [name]}
            inside = [contours[j] for j in range(len(contours)) if hierarchy[0][j][3] == i]
            if inside and cv2.contourArea(max(inside, key=cv2.contourArea)) >= min_area:
                hole = _simplify(max(inside, key=cv2.contourArea))
                if len(hole) >= 3:
                    value.update(ring=True, outerclosed=True, holes=[_to_perc(hole, w, h)])
            results.append({
                "id": f"tasdf-{cls}-{i}",
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


def _decode_gray(raw: bytes, what: str) -> np.ndarray:
    # same limits as the OpenCV backend (MAX_IMAGE_BYTES / MAX_IMAGE_PIXELS): /predict is reachable without auth
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"{what} exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"{what} exceeds {MAX_IMAGE_PIXELS} pixels")
            # PIL "L" conversion, as in tasdf BaseEmbryoDataset.load_raw_grayscale
            return np.array(im.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{what} is not a supported image") from exc


def fetch_image(ls_url: str, token: str, source: str, timeout: float = 30) -> tuple[np.ndarray, int, int]:
    """Download or decode a task image from Label Studio -> (float32 [0,1] SIZE x SIZE gray, orig_w, orig_h)."""
    if source.startswith("data:image/") and "," in source:
        try:
            raw = base64.b64decode(source.split(",", 1)[1], validate=False)
        except Exception as exc:
            raise ValueError("not a valid base64 image data URI") from exc
        gray = _decode_gray(raw, "data URI image")
    else:
        resp = requests.get(ls_media_url(ls_url, source), headers={"Authorization": f"Token {token}"}, timeout=timeout)
        resp.raise_for_status()
        gray = _decode_gray(resp.content, source)
    return preprocess(gray), gray.shape[1], gray.shape[0]


def preprocess(gray_uint8: np.ndarray) -> np.ndarray:
    # matches EmbryoTransformPipeline(is_train=False) with denoise disabled (M2-B config)
    img = cv2.resize(gray_uint8, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0
