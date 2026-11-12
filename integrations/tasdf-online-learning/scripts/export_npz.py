"""Export a labelled D10 split to the npz the backend reads (HOLDOUT_NPZ / ANCHOR_NPZ).

    python scripts/export_npz.py --root <Research>/dataset/D10 --split valid --out holdout.npz   # anti-forgetting gate
    python scripts/export_npz.py --root <Research>/dataset/D10 --split train --out anchors.npz   # anchor samples

Reads <root>/<split>/masks/*.png (class ids 0..4 = BG, ZP, TE, ICM, BC) and the image of the same name in
<root>/<split>/images/. Only images that have a mask are exported (D10 test has none: never use it here).
Output: images uint8 (N, 256, 256) gray, masks uint8 (N, 256, 256).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
CLASSES = {0, 1, 2, 3, 4}


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    mask_paths = sorted((root / split / "masks").glob("*.png"))
    if not mask_paths:
        raise SystemExit(f"no masks in {root / split / 'masks'}: this split has no labels")
    images, masks = [], []
    for mp in mask_paths:
        candidates = sorted((root / split / "images").glob(mp.stem + ".*"))
        if not candidates:
            raise SystemExit(f"no image for mask {mp.name}")
        gray = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if gray is None or mask is None or mask.ndim != 2:
            raise SystemExit(f"unreadable or non single-channel pair: {mp.name}")
        images.append(cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
        masks.append(cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))
    images, masks = np.stack(images).astype(np.uint8), np.stack(masks).astype(np.uint8)
    present = set(np.unique(masks).tolist())
    # the old export took D10 test via another source: no BC, TE covered the cavity. Refuse such sets.
    if present != CLASSES:
        raise SystemExit(f"masks must contain exactly classes {sorted(CLASSES)}, found {sorted(present)}")
    return images, masks


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="D10 dataset root (contains train/ valid/)")
    ap.add_argument("--split", required=True, choices=["train", "valid"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    images, masks = load_split(Path(args.root), args.split)
    np.savez_compressed(args.out, images=images, masks=masks)
    print(f"wrote {args.out}: {len(images)} samples from {args.split}, classes {sorted(np.unique(masks).tolist())}")


if __name__ == "__main__":
    sys.exit(main())
