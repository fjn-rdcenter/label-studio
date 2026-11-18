import time

import torch
from conftest import blastocyst_mask

from tasdf_online import learner
from tasdf_online.adapter import Sample, WellAdapters, WellState, adapt, adapter_loss, choose_source
from tasdf_online.bundle_registry import adapter_state, is_adapter_key


def setup():
    torch.manual_seed(0)
    m = learner.build_model(8).eval()
    prod = {k: v.clone() for k, v in adapter_state(m.state_dict()).items()}
    y = torch.from_numpy(blastocyst_mask().astype("int64"))
    s = Sample(torch.rand(1, 256, 256), y, torch.zeros(256, 256, dtype=torch.bool), y.clone())
    return m, prod, WellState(dict(prod), [s], {}, [], time.time())


def test_adapt_changes_only_adapter_keys_and_freezes_bn_stats():
    m, prod, st = setup()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    out = adapt(m, st, prod, steps=3, rollback_margin=1.0)
    assert not out["reverted"] and out["samples"] == 1
    for k, v in m.state_dict().items():
        if "running_" in k or "num_batches" in k or not is_adapter_key(k):
            assert torch.equal(v, before[k]), k
    assert any(not torch.equal(st.adapter[k], prod[k]) for k in prod)


def test_loss_goes_down_on_the_well():
    m, prod, st = setup()
    l0 = float(adapter_loss(m, prod, st.samples).detach())
    adapt(m, st, prod, steps=10, rollback_margin=1.0)
    m.load_state_dict(st.adapter, strict=False)
    assert float(adapter_loss(m, prod, st.samples)) < l0


def test_adapt_trains_in_micro_batches_to_bound_gpu_memory():
    m, prod, st = setup()
    st.samples = st.samples * 5
    seen = []
    forward = m.forward

    def spy(x, *a, **k):
        if torch.is_grad_enabled():
            seen.append(len(x))
        return forward(x, *a, **k)

    m.forward = spy
    adapt(m, st, prod, steps=1, rollback_margin=1.0)
    assert seen and max(seen) <= 2 and sum(seen) == 5


def test_rollback_keeps_old_adapter(monkeypatch):
    import tasdf_online.adapter as A
    m, prod, st = setup()
    calls = iter([0.9, 0.1])
    monkeypatch.setattr(A, "_mean_dice", lambda *a, **k: next(calls))
    out = adapt(m, st, prod, steps=1)
    assert out["reverted"] and all(torch.equal(st.adapter[k], prod[k]) for k in prod)


def test_well_store_evicts_lru_and_idle():
    w = WellAdapters(max_wells=2, ttl_s=3600)
    for key in [(1, 1), (1, 2), (1, 3)]:
        w.get(key, dict)
    assert len(w) == 2
    w2 = WellAdapters(max_wells=8, ttl_s=0)
    w2.get((1, 1), dict)
    time.sleep(0.01)
    w2.get((1, 2), dict)
    assert len(w2) == 1


def test_choose_source():
    assert choose_source([{"copy": 1, "production": 0, "adapter": 0}] * 2) == "production"
    s = [{"copy": .5, "production": .6, "adapter": .8}] * 3
    assert choose_source(s) == "adapter"
    assert choose_source([{"copy": .9, "production": .1, "adapter": .1}] * 5
                         + [{"copy": 0, "production": 1, "adapter": 0}] * 5) == "production"
