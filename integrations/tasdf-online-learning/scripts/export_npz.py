"""Export a labeled TASDF split to the npz the backend reads (HOLDOUT_NPZ / ANCHOR_NPZ).

Runs where the research repo is importable, with the exact dataset config of the baseline run:

    python scripts/export_npz.py --tasdf-repo D:/Workspace/model/tasdf-audit \
        --run-json D:/Workspace/model/tasdf-audit/experiments/fix_m2b/run/run.json \
        --split test --root-dir D:/Workspace/model/tasdf-audit/data/D10 --out holdout.npz

Output: images uint8 (N, 256, 256), masks uint8 (N, 256, 256) with 255 = ignore.
"""
import argparse
import json
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasdf-repo", required=True)
    ap.add_argument("--run-json", required=True, help="run.json of the baseline run (dataset section is reused)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root-dir", help="override dataset.root_dir (run.json may hold a server path)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.tasdf_repo)
    from tasdf.data.builder import build_dataset

    cfg = dict(json.load(open(args.run_json, encoding="utf-8"))["config"]["dataset"])
    if args.root_dir:
        cfg["root_dir"] = args.root_dir
    ds = build_dataset(cfg, split=args.split, is_labeled=True, is_train=False)
    images, masks = [], []
    for i in range(len(ds)):
        img, mask = ds[i][:2]
        images.append(np.round(img.squeeze(0).numpy() * 255).astype(np.uint8))
        masks.append(mask.numpy().astype(np.uint8))
    np.savez_compressed(args.out, images=np.stack(images), masks=np.stack(masks))
    print(f"wrote {args.out}: {len(images)} samples from split={args.split}")


if __name__ == "__main__":
    main()
