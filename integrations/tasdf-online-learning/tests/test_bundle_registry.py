import json

import numpy as np
import pytest
import torch

from tasdf_online import learner
from tasdf_online.bundle_registry import BASE, BundleRegistry, adapter_state, decoder_state
from tasdf_online.config import load_stages, stage_of


def _model(seed=0):
    torch.manual_seed(seed)
    return learner.build_model(base_channels=8).eval()


def _perturb(model, mode="heavy"):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if learner.is_trainable(n, mode):
                p.add_(0.05 * torch.randn_like(p))
    return model


def test_base_bundle_and_unique_names(tmp_path):
    base = _model().state_dict()
    reg = BundleRegistry(tmp_path / "registry", base, "sha-base", ["blastocyst"])
    assert reg.production == BASE == reg.previous
    assert all(torch.equal(v, base[k]) for k, v in reg.state_for(BASE, "blastocyst").items())
    m = _perturb(_model())
    b1 = reg.save_bundle(m.state_dict(), ["blastocyst"], {"metrics": {}})
    assert b1.startswith("tasdf-") and len(b1.split("-")[-1]) == 6
    assert reg.save_bundle(m.state_dict(), ["blastocyst"], {}) == b1  # same content -> same name, no duplicate
    b2 = reg.save_bundle(_perturb(_model(1)).state_dict(), ["blastocyst"], {})
    assert b2 != b1 and reg.model_version(b2, "blastocyst") == f"{b2}.blastocyst"


def test_bundle_round_trip_is_close_to_fp32(tmp_path):
    base = _model().state_dict()
    reg = BundleRegistry(tmp_path / "registry", base, "sha", ["blastocyst"])
    trained = _perturb(_model())
    b = reg.save_bundle(trained.state_dict(), ["blastocyst"], {})
    loaded = _model()
    loaded.load_state_dict(reg.state_for(b, "blastocyst"))
    x = torch.rand(2, 1, 256, 256)
    p1 = learner.predict_probs(trained, x)[0].argmax(1)
    p2 = learner.predict_probs(loaded, x)[0].argmax(1)
    assert (p1 == p2).float().mean() > 0.995  # fp16 decoder storage does not change the prediction
    # encoder untouched by a bundle: comes from the baseline
    enc = [k for k in base if not k.startswith(("net1.heads", "net2.heads")) and ".down" in k and "double_conv.0" in k]
    assert enc and all(torch.equal(reg.state_for(b, "blastocyst")[k], base[k]) for k in enc)


def test_split_keys_cover_decoder_and_adapter(tmp_path):
    state = _model().state_dict()
    dec, ad = decoder_state(state), adapter_state(state)
    assert dec and ad and not (set(dec) & set(ad))
    assert any("running_mean" in k for k in ad) and any(".heads." in k for k in ad)
    assert all(".up" in k for k in dec)


def test_promote_rollback_history_and_gc(tmp_path):
    reg = BundleRegistry(tmp_path / "registry", _model().state_dict(), "sha", ["blastocyst"])
    names = []
    for i in range(5):  # each job saves a bundle and the gate decides right away
        names.append(reg.save_bundle(_perturb(_model(i)).state_dict(), ["blastocyst"], {}))
        if i < 2:
            reg.promote(names[-1], "gate")
        else:
            reg.record(names[-1], status="rejected")
    assert (reg.production, reg.previous) == (names[1], names[0])
    assert reg.rollback() == names[0] and reg.production == names[0]
    assert [h["reason"] for h in reg.data["history"]] == ["gate", "gate", "rollback"]
    kept = set(reg.data["bundles"])
    assert {BASE, names[0], names[1]} <= kept and len(kept) <= 6
    files = {f.stem for f in (tmp_path / "registry" / "decoders").glob("*.pt")}
    assert files == {reg.info(b)["decoder"] for b in kept} - {"dec-base"}
    again = BundleRegistry(tmp_path / "registry", _model().state_dict(), "sha", ["blastocyst"])  # survives restart
    assert again.production == names[0]
    with pytest.raises(RuntimeError, match="BASELINE_CKPT changed"):
        BundleRegistry(tmp_path / "registry", _model().state_dict(), "other", ["blastocyst"])


def test_migrate_legacy_moves_old_registries(tmp_path):
    state = tmp_path / "state"
    (state / "37" / "versions").mkdir(parents=True)
    (state / "37" / "registry.json").write_text(json.dumps({"active": "v8", "versions": {"v0": {}, "v8": {}}}))
    reg = BundleRegistry(tmp_path / "registry", _model().state_dict(), "sha", ["blastocyst"])
    assert reg.migrate_legacy(state) == ["37"]
    assert not (state / "37").exists() and (tmp_path / "registry" / "legacy" / "37" / "registry.json").exists()
    assert reg.data["legacy"]["37"]["active_was"] == "v8" and reg.production == BASE
    assert reg.migrate_legacy(state) == []


def test_load_stages_validates(tmp_path):
    (tmp_path / "holdout.npz").write_bytes(b"x")
    good = {"blastocyst": {"projects": [36, 37], "holdout_npz": "holdout.npz"}}
    (tmp_path / "stages.json").write_text(json.dumps(good))
    stages = load_stages(str(tmp_path / "stages.json"))
    assert stage_of(stages, "37.1700000") == "blastocyst" and stage_of(stages, 99) is None
    (tmp_path / "stages.json").write_text(json.dumps({**good, "pn": {"projects": [37]}}))
    with pytest.raises(ValueError, match="both"):
        load_stages(str(tmp_path / "stages.json"))
    (tmp_path / "stages.json").write_text(json.dumps({"blastocyst": {"projects": [1], "anchors_npz": "missing.npz"}}))
    with pytest.raises(ValueError, match="does not exist"):
        load_stages(str(tmp_path / "stages.json"))
