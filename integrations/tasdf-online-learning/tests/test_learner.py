from tasdf_online import learner


def test_trainable_count_matches_spec():
    model = learner.build_model(base_channels=64)
    assert sum(p.numel() for p in model.parameters()) == 62_074_002
    assert learner.select_trainable(model) == learner.EXPECTED_TRAINABLE_BASE64


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
