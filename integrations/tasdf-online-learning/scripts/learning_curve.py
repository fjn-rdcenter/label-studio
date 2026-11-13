"""Learning curve: does light / heavy retraining improve as reviewed labels accumulate?

Read-only (never touches the registry). Runs where the backend runs, e.g.:

    docker exec -i label-studio-tasdf-online-1 python - --projects 36,37 < scripts/learning_curve.py

For every k in {4, 8, 16, ...} reviewed train-split annotations (3 random subsets each) it trains
  light      : BN + up4 + heads, from the baseline
  heavy-base : whole decoder + BN + heads, from the baseline
  heavy-warm : whole decoder + BN + heads, from the current production model
with a fixed epoch budget (no early stopping, so nothing is selected on the evaluation data), and reports
  val   : Dice on the fixed project validation images (content-hash split, never trained on)
  d10   : mDice on the D10/valid holdout (forgetting)
  train : Dice on the k training images; train - val is the memorization gap
With fewer than VAL_MIN fixed validation images it switches to leave-one-out cross-validation over all
reviewed annotations (each image is the held-out one in turn) and labels the result as a CV estimate.
Memory: one torch process at a time; base/production stay on the GPU, each candidate is freed after scoring,
and every measured row is appended to <reports>/learning_curve-<timestamp>.jsonl right away.
Results: <reports>/learning_curve-<timestamp>.{jsonl,md} and one line appended to <reports>/history.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODES = (("light", "light", "base"), ("heavy-base", "heavy", "base"), ("heavy-warm", "heavy", "production"))
VAL_MIN = 10


def ks_for(n: int, start: int = 4) -> list[int]:
    ks, k = [], start
    while k < n:
        ks.append(k)
        k *= 2
    return ks + [n] if n else []


def dice_per_image(model, xs: torch.Tensor, ys: list[np.ndarray]) -> list[float]:
    from tasdf_online import trainer

    return trainer.mean_dice_per_image(model, xs, ys)


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_curve(train: list[dict], val: list[dict], holdout, anchors, models: dict, *, ks: list[int], seeds: int,
              epochs: int, lr: float, amp: bool, fold=None, on_row=None) -> list[dict]:
    """models: {"base": nn.Module, "production": nn.Module}. Each sample: {"x": (1,H,W) tensor, "y": np.ndarray}.
    `on_row` gets every measured row as soon as it exists (persist it: a crash keeps what was done)."""
    from tasdf_online import learner, trainer

    vx = torch.stack([d["x"] for d in val]) if val else torch.zeros((0, 1, 256, 256))
    vys = [d["y"] for d in val]
    rows = []

    def emit(row):
        row["fold"] = fold
        rows.append(row)
        if on_row:
            on_row(row)

    for ref_name in ("base", "production"):
        ref = models[ref_name]
        emit({"mode": f"ref:{ref_name}", "k": 0, "seed": None,
              "val": float(np.mean(dice_per_image(ref, vx, vys))) if val else None,
              "d10": learner.evaluate(ref, *holdout)["pooled_mDice"] if holdout is not None else None,
              "train": None, "seconds": 0.0})
    for k in ks:
        for seed in range(seeds):
            subset = random.Random(1000 * k + seed).sample(train, k)
            x = torch.stack([d["x"] for d in subset])
            y = torch.from_numpy(np.stack([d["y"] for d in subset]).astype(np.int64))
            for name, mode, start in MODES:
                t0 = time.time()
                cand, _ = trainer.train_epochs(models[start], x, y, anchors, mode=mode, epochs=epochs, lr=lr, amp=amp,
                                               seed=seed)
                emit({"mode": name, "k": k, "seed": seed,
                      "val": float(np.mean(dice_per_image(cand, vx, vys))) if val else None,
                      "d10": learner.evaluate(cand, *holdout)["pooled_mDice"] if holdout is not None else None,
                      "train": float(np.mean(dice_per_image(cand, x, [d["y"] for d in subset]))),
                      "seconds": round(time.time() - t0, 1)})
                del cand
                _free()
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for name in [r["mode"] for r in rows if r["k"] == 0] + [m for m, _, _ in MODES]:
        for k in sorted({r["k"] for r in rows if r["mode"] == name}):
            g = [r for r in rows if r["mode"] == name and r["k"] == k]
            agg = {"mode": name, "k": k, "n": len(g)}
            for key in ("val", "d10", "train", "seconds"):
                vals = [r[key] for r in g if r[key] is not None]
                agg[key] = (st.mean(vals), st.pstdev(vals)) if vals else None
            agg["gap"] = (agg["train"][0] - agg["val"][0]) if agg["train"] and agg["val"] else None
            out.append(agg)
    return out


def to_markdown(summary: list[dict], meta: dict) -> str:
    def f(v):
        return "—" if v is None else (f"{v[0]:.4f} ± {v[1]:.4f}" if isinstance(v, tuple) else f"{v:.4f}")

    lines = [f"# Learning curve {meta['at']}", "",
             f"- Train/val annotations: {meta['n_train']} / {meta['n_val']} (projects {meta['projects']}, "
             f"annotators {meta['annotators']}); epochs {meta['epochs']}, seeds {meta['seeds']}",
             f"- Evaluation: {meta['evaluation']}",
             "- val = project validation Dice (never trained on), d10 = D10/valid mDice, gap = train − val", "",
             "| mode | k | val | d10 | train | gap | s |", "|---|---|---|---|---|---|---|"]
    for a in summary:
        gap = "—" if a["gap"] is None else f"{a['gap']:+.4f}"
        secs = "—" if not a["seconds"] else str(round(a["seconds"][0]))
        lines.append(f"| {a['mode']} | {a['k']} | {f(a['val'])} | {f(a['d10'])} | {f(a['train'])} | {gap} | {secs} |")
    return "\n".join(lines) + "\n"


def collect(s, projects: list[int]) -> tuple[list[dict], list[dict], set]:
    """Reviewed annotations of `projects` from LS -> (train, val) samples with 256x256 inputs."""
    import requests

    from tasdf_online import trainer
    from tasdf_online.app import parse_label_config
    from tasdf_online.ls_format import fetch_image

    headers = {"Authorization": f"Token {s.ls_token}"}
    train, val, annotators = [], [], set()
    for pid in projects:
        cfg = requests.get(f"{s.ls_url}/api/projects/{pid}", headers=headers, timeout=60).json()["label_config"]
        from_name, _, key, _ = parse_label_config(cfg)
        tasks, page = [], 1
        while True:
            r = requests.get(f"{s.ls_url}/api/projects/{pid}/tasks/", params={"page": page, "page_size": 100},
                             headers=headers, timeout=60)
            if r.status_code == 404 or not r.json():
                break
            r.raise_for_status()
            tasks += r.json()
            page += 1
        for d in trainer.select_samples(tasks, pid, from_name, s.min_quality_level):
            img, _, _ = fetch_image(s.ls_url, s.ls_token, d["task"]["data"][key], s.request_timeout)
            ann = next(a for a in d["task"]["annotations"] if a.get("id") == d["annotation_id"])
            annotators.add(ann.get("completed_by"))
            sample = {"x": torch.from_numpy(img)[None], "y": d["y"], "project": pid, "task_id": d["task_id"]}
            (val if trainer.image_split(img) == "val" else train).append(sample)
    return train, val, annotators


def production_model(s, base, stage: str = "blastocyst"):
    """Current production weights from the bundle registry (the baseline when no registry exists yet)."""
    import copy
    import os

    from tasdf_online.bundle_registry import BundleRegistry

    model = copy.deepcopy(base)
    path = os.path.join(s.registry_dir, "registry.json")
    if not os.path.exists(path):
        return model, "baseline (no registry yet)"
    cpu_base = {k: v.cpu() for k, v in base.state_dict().items()}
    reg = BundleRegistry(s.registry_dir, cpu_base, json.load(open(path))["base_sha"], [stage])
    model.load_state_dict(reg.state_for(reg.production, stage))
    name = reg.model_version(reg.production, stage)
    del cpu_base
    gc.collect()
    return model.to(next(base.parameters()).device).eval(), name


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", default="36,37")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reports", default="/data/tasdf-online/reports")
    args = ap.parse_args(argv)

    from tasdf_online import learner
    from tasdf_online.app import _load_npz
    from tasdf_online.config import Settings, resolve_device

    s = Settings.from_env()
    torch.set_num_threads(4)
    device = resolve_device(s.device)
    amp = device.type == "cuda"
    base = learner.build_model(s.base_channels, s.erosion)
    ckpt = learner.load_state_dict(s.baseline_ckpt)
    base.load_state_dict(ckpt)
    del ckpt  # the checkpoint file is ~745 MB: never keep a second CPU copy around
    gc.collect()
    base = base.to(device).eval()
    prod, prod_name = production_model(s, base)
    holdout = _load_npz(s.holdout_npz, s.holdout_limit)
    anchors = _load_npz(s.anchor_npz) if s.anchor_npz else None
    projects = [int(p) for p in args.projects.split(",") if p]
    train, val, annotators = collect(s, projects)
    print(f"train {len(train)} / val {len(val)} annotations from projects {projects}", flush=True)
    out = Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    jsonl = out / f"learning_curve-{stamp}.jsonl"
    models = {"base": base, "production": prod}

    def persist(row):
        with jsonl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if len(val) >= VAL_MIN:
        evaluation = f"fixed validation split ({len(val)} images, content hash, never trained on)"
        rows = run_curve(train, val, holdout, anchors, models, ks=ks_for(len(train)), seeds=args.seeds,
                         epochs=args.epochs, lr=args.lr, amp=amp, on_row=persist)
    else:  # too few validation images for a fixed split: leave-one-out over everything reviewed
        pool = train + val
        evaluation = f"leave-one-out CV estimate over {len(pool)} images (fewer than {VAL_MIN} fixed val images)"
        rows = []
        for i, held in enumerate(pool):
            rest = pool[:i] + pool[i + 1:]
            rows += run_curve(rest, [held], holdout, anchors, models, ks=ks_for(len(rest)), seeds=1,
                              epochs=args.epochs, lr=args.lr, amp=amp, fold=i, on_row=persist)
    summary = summarize(rows)
    at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = {"at": at, "projects": projects, "n_train": len(train), "n_val": len(val), "epochs": args.epochs,
            "seeds": args.seeds, "production": prod_name, "annotators": len(annotators - {None}),
            "evaluation": evaluation}
    md = to_markdown(summary, meta)
    (out / f"learning_curve-{stamp}.md").write_text(md)
    best = max((a for a in summary if a["mode"] in ("light", "heavy-base", "heavy-warm") and a["val"]),
               key=lambda a: a["val"][0], default=None)
    ref = next((a for a in summary if a["mode"] == "ref:production"), None)
    line = (f"| {at} | {len(train)} | {len(val)} | {ref['val'][0]:.4f} | "
            f"{best['mode'] + ' k=' + str(best['k']) if best else '—'} | {best['val'][0]:.4f} |\n"
            if best and ref and ref["val"] else f"| {at} | {len(train)} | {len(val)} | — | — | — |\n")
    hist = out / "history.md"
    if not hist.exists():
        hist.write_text("| at | train | val | production val | best mode | best val |\n|---|---|---|---|---|---|\n")
    with hist.open("a") as fh:
        fh.write(line)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
