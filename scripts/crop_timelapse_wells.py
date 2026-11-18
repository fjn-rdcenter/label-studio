"""Crop every image of a timelapse dataset to a fixed window centred on each well's embryo.

Layout in and out: <dataset_dir>/<dhhmmss>/<00001.jpg ...>, every `--slices` consecutive images are one well
(same as timelapse_wells.py). One box per well, found on the middle slice and kept for every frame and slice,
so region coordinates stay comparable over time. Boxes are written to <out>/boxes.json.

    python scripts/crop_timelapse_wells.py ../dataset/OneDrive_4_11-11-2026 ../dataset/OneDrive_4_11-11-2026_crop600
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def embryo_centre(gray: np.ndarray) -> tuple[int, int]:
    """(x, y) of the most textured area inside the microwell; the wall and its halo are left out by finding
    the microwell circle first (radius ~340-400 px on 964x1288 wells)."""
    scale = 4
    small = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    circles = cv2.HoughCircles(cv2.GaussianBlur(small, (5, 5), 0), cv2.HOUGH_GRADIENT, dp=1, minDist=200,
                               param1=60, param2=25, minRadius=60, maxRadius=120)
    f = small.astype(np.float32)
    mean = cv2.blur(f, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) - mean * mean, 0))
    if circles is not None:  # no microwell found: search the whole image
        cx, cy, r = circles[0][0]
        yy, xx = np.mgrid[:f.shape[0], :f.shape[1]]
        std[(xx - cx) ** 2 + (yy - cy) ** 2 >= (0.8 * r) ** 2] = 0
    energy = cv2.GaussianBlur(std, (0, 0), 10)
    # centroid of the energy plateau: steadier than the single brightest pixel of a random texture
    ys, xs = np.nonzero(energy >= 0.7 * energy.max())
    return int(xs.mean() * scale + scale // 2), int(ys.mean() * scale + scale // 2)


def _box(cx: float, cy: float, w: int, h: int, size: int) -> tuple[int, int, int, int]:
    sw, sh = min(size, w), min(size, h)
    x0 = int(min(max(round(cx - sw / 2), 0), w - sw))
    y0 = int(min(max(round(cy - sh / 2), 0), h - sh))
    return x0, y0, x0 + sw, y0 + sh


def crop_dataset(src, dst, slices: int = 11, size: int = 600) -> list[tuple[int, int, int, int]]:
    src, dst = Path(src), Path(dst)
    frames = sorted(p for p in src.iterdir() if p.is_dir() and re.fullmatch(r'\d{7}', p.name))
    images = {f: sorted(f.glob('*.jpg')) for f in frames}
    n = {len(v) for v in images.values()}
    if len(n) != 1 or not n or next(iter(n)) % slices:
        raise ValueError(f'every frame needs the same number of images, a multiple of {slices}: {sorted(n)}')
    wells, ref = next(iter(n)) // slices, slices // 2
    boxes = []
    for w in range(wells):
        centres, shape = [], None
        for f in frames:
            gray = cv2.imread(str(images[f][w * slices + ref]), cv2.IMREAD_GRAYSCALE)
            shape = gray.shape
            centres.append(embryo_centre(gray))
        cx, cy = np.median(np.asarray(centres, np.float64), axis=0)
        boxes.append(_box(cx, cy, shape[1], shape[0], size))
    for f in frames:
        (dst / f.name).mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(images[f]):
            x0, y0, x1, y1 = boxes[i // slices]
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(dst / f.name / path.name), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    (dst / 'boxes.json').write_text(json.dumps({'source': src.name, 'size': size, 'boxes': boxes}))
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--slices', type=int, default=11)
    parser.add_argument('--size', type=int, default=600)
    args = parser.parse_args()
    try:
        boxes = crop_dataset(args.dataset_dir, args.output_dir, args.slices, args.size)
    except ValueError as e:
        sys.exit(f'error: {e}')
    for w, b in enumerate(boxes, 1):
        print(f'W{w}: box {b}')


if __name__ == '__main__':
    main()
