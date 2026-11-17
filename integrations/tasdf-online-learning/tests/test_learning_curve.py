import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import blastocyst_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import learning_curve  # noqa: E402

from tasdf_online import learner  # noqa: E402
from tasdf_online.bundle_registry import BASE, BundleRegistry  # noqa: E402


def test_ks_double_up_to_all():
    assert learning_curve.ks_for(3) == [3]
    assert learning_curve.ks_for(10) == [4, 8, 10]
    assert learning_curve.ks_for(0) == []


def test_pick_ks_keeps_requested_sizes_that_fit():
    assert learning_curve.pick_ks(18, "8,18") == [8, 18]
    assert learning_curve.pick_ks(10, "8, 18") == [8]
    assert learning_curve.pick_ks(10, None) == learning_curve.ks_for(10)


def test_drop_tasks_leaves_listed_tasks_out():
    samples = [{"task_id": t} for t in (1, 2, 3)]
    assert [d["task_id"] for d in learning_curve.drop_tasks(samples, "2, 3")] == [1]
    assert learning_curve.drop_tasks(samples, None) == samples


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
    two_folds = rows + [dict(r, fold=1) for r in rows]
    assert [a["mode"] for a in learning_curve.summarize(two_folds)].count("ref:base") == 1
    md = learning_curve.to_markdown(learning_curve.summarize(rows), {
        "at": "t", "projects": [37], "n_train": 3, "n_val": 1, "epochs": 1, "seeds": 1, "annotators": 1,
        "evaluation": "leave-one-out"})
    assert "| heavy-warm | 3 |" in md and "gap" in md
    done = {(r["mode"], r["k"], r["seed"], r["fold"]) for r in rows if r["k"] < 3}
    rest = learning_curve.run_curve(train, val, holdout, None, {"base": base, "production": base}, ks=[2, 3],
                                    seeds=1, epochs=1, lr=1e-3, amp=False, fold=0, done=done)
    assert {(r["mode"], r["k"]) for r in rest} == {(m, 3) for m in ("light", "heavy-base", "heavy-warm")}


def test_resume_refuses_rows_measured_with_other_settings(capsys):
    cfg = {"epochs": 40, "lr": 3e-5, "seeds": 3, "anchor_frac": 0.0, "no_d10": False, "exclude": [], "pool": "abc"}
    learning_curve.check_resume([{"mode": "light", "cfg": dict(cfg)}], cfg)  # same settings: fine
    with pytest.raises(SystemExit, match="epochs"):
        learning_curve.check_resume([{"mode": "light", "cfg": dict(cfg, epochs=20)}], cfg)
    with pytest.raises(SystemExit, match="exclude"):  # fold i would be another image
        learning_curve.check_resume([{"mode": "light", "cfg": dict(cfg, exclude=[10221])}], cfg)
    learning_curve.check_resume([{"mode": "light"}], cfg)  # rows from before cfg existed: warn, keep
    assert "warning" in capsys.readouterr().out


def test_production_model_reads_the_configured_stage_without_writing(settings, tmp_path):
    (tmp_path / "stages.json").write_text(json.dumps({"day5": {"projects": [1]}}))
    base = learner.build_model(base_channels=8).eval()
    BundleRegistry(settings.registry_dir, base.state_dict(), "sha-x", ["day5"])
    reg_file = Path(settings.registry_dir) / "registry.json"
    before = reg_file.read_bytes(), reg_file.stat().st_mtime_ns
    model, name = learning_curve.production_model(settings, base)
    assert name == f"{BASE}.day5" and model is not base
    assert (reg_file.read_bytes(), reg_file.stat().st_mtime_ns) == before  # a reader never rewrites the registry


def test_read_rows_drops_lines_torn_by_a_power_cut(tmp_path):
    p = tmp_path / "lc.jsonl"
    p.write_bytes(b'{"mode": "light", "k": 4}\n{"mode": "heavy-b' + b"\x00" * 20 + b"\n")
    assert learning_curve.read_rows(p) == [{"mode": "light", "k": 4}]
