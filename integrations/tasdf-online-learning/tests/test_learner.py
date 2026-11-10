import numpy as np
import torch
from conftest import blastocyst_mask

from tasdf_online import learner


def test_trainable_count_matches_spec():
    model = learner.build_model(base_channels=64)
    assert sum(p.numel() for p in model.parameters()) == 62_074_002
    assert learner.select_trainable(model) == learner.EXPECTED_TRAINABLE_BASE64


def test_episode_moves_only_trainable_weights_and_keeps_bn_stats():
    torch.manual_seed(0)
    teacher = learner.build_model(base_channels=8).eval()
    base = {k: v.clone() for k, v in teacher.state_dict().items()}
    y = torch.from_numpy(blastocyst_mask().astype(np.int64))[None].repeat(2, 1, 1)
    x = torch.rand(2, 1, 256, 256)
    w = torch.ones(y.shape)
    cand, losses = learner.run_episode(teacher, x, y, w, epochs=2, lr=1e-2, ema_alpha=0.65, topo_sup_weight=0.1,
                                       erosion="cross", amp=False)
    assert len(losses) == 2 and np.isfinite(losses[-1]["total"])
    shifts = learner.layer_shifts(teacher, cand, base)
    assert shifts["encoder_frozen"]["l2_shift"] == 0.0 and shifts["decoder_frozen"]["l2_shift"] == 0.0
    for g in ("bn_affine", "decoder_up4", "pixel_head", "topo_head"):
        assert shifts[g]["l2_shift"] > 0, g
    for name, buf in cand.named_buffers():
        if "running_" in name:
            assert torch.equal(buf, base[name]), name
    # teacher untouched
    assert all(torch.equal(v, base[k]) for k, v in teacher.state_dict().items())


def test_ignore_index_in_replay_masks():
    teacher = learner.build_model(base_channels=8).eval()
    y = torch.full((1, 256, 256), 255, dtype=torch.long)
    y[0, 100:150, 100:150] = 2
    cand, losses = learner.run_episode(teacher, torch.rand(1, 1, 256, 256), y, torch.ones(y.shape), epochs=1, lr=1e-3,
                                       ema_alpha=0.65, topo_sup_weight=0.1, erosion="cross", amp=False)
    assert np.isfinite(losses[0]["total"])


def _m(mdice, d):
    return {"pooled_mDice": mdice, "dice": dict(enumerate(d, start=1))}


def test_gate_thresholds():
    base = prev = _m(0.80, [0.8, 0.8, 0.8, 0.8])
    assert learner.gate(_m(0.796, [0.8, 0.79, 0.8, 0.8]), prev, base)[0]
    ok, reasons = learner.gate(_m(0.79, [0.8, 0.8, 0.8, 0.8]), prev, base)
    assert not ok and "vs prev" in reasons[0]
    ok, reasons = learner.gate(_m(0.80, [0.8, 0.74, 0.8, 0.86]), prev, base)
    assert not ok and any("class 2 drop vs prev" in r for r in reasons)
    ok, reasons = learner.gate(_m(0.74, [0.74] * 4), _m(0.745, [0.745] * 4), base)
    assert not ok and any("vs v0" in r for r in reasons)


def test_gate_per_class_after_registry_round_trip():
    # registry.json turns int class keys into "1".."4"; the per-class checks must still fire after a restart
    import json

    prev = json.loads(json.dumps(_m(0.80, [0.8, 0.8, 0.8, 0.8])))
    ok, reasons = learner.gate(_m(0.80, [0.8, 0.74, 0.8, 0.86]), prev, prev)
    assert not ok and any("class 2 drop vs prev" in r for r in reasons)


def test_local_probe_absorption():
    y = blastocyst_mask()
    before = torch.zeros(5, 256, 256); before[0] = 1
    after = torch.nn.functional.one_hot(torch.from_numpy(y.astype(np.int64)), 5).permute(2, 0, 1).float()
    probe = learner.local_probe(before, after, y)
    assert probe["absorption_rate_pct"] == 100.0 and probe["dice_after"] == 1.0
