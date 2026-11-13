import sys
from pathlib import Path

import numpy as np
import torch
from conftest import blastocyst_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import learning_curve  # noqa: E402

from tasdf_online import learner  # noqa: E402


def test_ks_double_up_to_all():
    assert learning_curve.ks_for(3) == [3]
    assert learning_curve.ks_for(10) == [4, 8, 10]
    assert learning_curve.ks_for(0) == []


def test_run_curve_reports_every_mode_and_reference():
    torch.manual_seed(0)
    base = learner.build_model(base_channels=8).eval()
    sample = lambda: {"x": torch.rand(1, 256, 256), "y": blastocyst_mask()}  # noqa: E731
    train, val = [sample() for _ in range(3)], [sample()]
    holdout = (torch.rand(2, 1, 256, 256), np.stack([blastocyst_mask()] * 2).astype(np.int64))
    seen = []
    rows = learning_curve.run_curve(train, val, holdout, None, {"base": base, "production": base}, ks=[2, 3], seeds=1,
                                    epochs=1, lr=1e-3, amp=False, fold=0, on_row=seen.append)
    assert seen == rows and all(r["fold"] == 0 for r in rows)  # every row is handed out as soon as it exists
    modes = {(r["mode"], r["k"]) for r in rows}
    assert {("ref:base", 0), ("ref:production", 0), ("light", 2), ("heavy-base", 3), ("heavy-warm", 3)} <= modes
    assert all(0.0 <= r["val"] <= 1.0 for r in rows)
    md = learning_curve.to_markdown(learning_curve.summarize(rows), {
        "at": "t", "projects": [37], "n_train": 3, "n_val": 1, "epochs": 1, "seeds": 1, "annotators": 1,
        "evaluation": "leave-one-out"})
    assert "| heavy-warm | 3 |" in md and "gap" in md
    done = {(r["mode"], r["k"], r["seed"], r["fold"]) for r in rows if r["k"] < 3}
    rest = learning_curve.run_curve(train, val, holdout, None, {"base": base, "production": base}, ks=[2, 3],
                                    seeds=1, epochs=1, lr=1e-3, amp=False, fold=0, done=done)
    assert {(r["mode"], r["k"]) for r in rest} == {(m, 3) for m in ("light", "heavy-base", "heavy-warm")}
