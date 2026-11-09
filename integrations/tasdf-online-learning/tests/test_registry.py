import torch

from tasdf_online import learner
from tasdf_online.registry import Registry


def test_versions_survive_restart_and_rollback(tmp_path):
    base_model = learner.build_model(base_channels=8)
    base = {k: v.clone() for k, v in base_model.state_dict().items()}
    cand = learner.build_model(base_channels=8)
    cand.load_state_dict(base)
    with torch.no_grad():
        for n, p in cand.named_parameters():
            if learner.is_trainable(n):
                p.add_(0.5)

    reg = Registry(str(tmp_path), 3)
    v1 = reg.add(learner.trainable_state(cand), {"metrics": {"pooled_mDice": 0.9}}, accepted=True)
    v2 = reg.add(learner.trainable_state(cand), {"metrics": {"pooled_mDice": 0.1}}, accepted=False)
    assert (reg.active, v1, v2) == ("v1", "v1", "v2")
    assert not (tmp_path / "3" / "versions" / "v2.pt").exists()

    reloaded = Registry(str(tmp_path), 3)  # simulated restart
    assert reloaded.active == "v1" and reloaded.info("v2")["status"] == "rejected"
    model = learner.build_model(base_channels=8)
    reloaded.load_into(model, base, reloaded.active)
    assert all(torch.equal(v, cand.state_dict()[k]) for k, v in model.state_dict().items())

    reloaded.rollback("v0")
    reloaded.load_into(model, base, Registry(str(tmp_path), 3).active)
    assert all(torch.equal(v, base[k]) for k, v in model.state_dict().items())


def test_delta_is_small_at_full_size(tmp_path):
    model = learner.build_model(base_channels=64)
    path = tmp_path / "delta.pt"
    torch.save(learner.trainable_state(model), path)
    assert path.stat().st_size < 2 * 1024 * 1024
