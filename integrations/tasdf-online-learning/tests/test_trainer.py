import numpy as np
import pytest
import torch
from conftest import blastocyst_mask

from tasdf_online import learner, trainer
from tasdf_online.ls_format import IGNORE, mask_to_results


def _ann(ann_id, result, quality=2, updated="2026-11-12T10:00", cancelled=False):
    return {"id": ann_id, "result": result, "quality_level": quality, "updated_at": updated, "was_cancelled": cancelled}


FULL = mask_to_results(blastocyst_mask(), 256, 256, "seg", "image")


def test_split_is_fixed_per_task_and_about_a_fifth():
    splits = [trainer.split_of(37, t) for t in range(1000)]
    assert splits == [trainer.split_of(37, t) for t in range(1000)]
    assert 150 < splits.count("val") < 250


def test_select_samples_takes_latest_usable_annotation():
    tasks = [
        {"id": 1, "annotations": [_ann(10, FULL, updated="2026-11-12T09:00"), _ann(11, FULL[:1], updated="2026-11-12T11:00")]},
        {"id": 2, "annotations": [_ann(20, FULL, quality=1)]},           # below MIN_QUALITY_LEVEL
        {"id": 3, "annotations": [_ann(30, FULL, cancelled=True)]},      # skipped
        {"id": 4, "annotations": [_ann(40, [{"type": "rectanglelabels", "from_name": "seg",
                                             "value": {"x": 10, "y": 10, "width": 50, "height": 50,
                                                       "rectanglelabels": ["ZP"]}}])]},  # prompts only
        {"id": 5, "annotations": []},
    ]
    samples = trainer.select_samples(tasks, 37, "seg", min_quality=2)
    assert [s["task_id"] for s in samples] == [1]
    assert samples[0]["annotation_id"] == 11 and samples[0]["digest"] == trainer.digest(FULL[:1])


def test_mean_dice_ignore_skips_ignored_pixels():
    y = blastocyst_mask()
    noisy = y.copy()
    target = y.copy()
    target[:, :128] = IGNORE
    noisy[:, :128] = 0  # wrong only where the target is ignored
    assert trainer.mean_dice_ignore(noisy, target) == pytest.approx(1.0)
    assert trainer.mean_dice_ignore(noisy, y) < 1.0


def test_train_moves_only_trainable_weights_and_keeps_bn_stats():
    torch.manual_seed(0)
    base = learner.build_model(base_channels=8).eval()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    y = torch.from_numpy(blastocyst_mask().astype(np.int64))[None].repeat(2, 1, 1)
    y[1, :, :100] = IGNORE  # a partial annotation must not break the loss
    x = torch.rand(2, 1, 256, 256)
    anchors = (torch.rand(3, 1, 256, 256), blastocyst_mask()[None].repeat(3, 0).astype(np.int64))
    scores = iter(range(100))
    cand, history = trainer.train(base, x, y, anchors, steps=3, lr=1e-2, eval_every=1,
                                  score_fn=lambda m: float(next(scores)), amp=False)
    assert len(history) == 3 and all(np.isfinite(h["loss"]) for h in history)
    shifts = learner.layer_shifts(base, cand, before)
    assert shifts["encoder_frozen"]["l2_shift"] == 0.0 and shifts["decoder_frozen"]["l2_shift"] == 0.0
    assert shifts["pixel_head"]["l2_shift"] > 0
    for name, buf in cand.named_buffers():
        if "running_" in name:
            assert torch.equal(buf, before[name]), name
    assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items())  # base untouched


def test_train_keeps_the_best_checkpoint():
    base = learner.build_model(base_channels=8).eval()
    y = torch.from_numpy(blastocyst_mask().astype(np.int64))[None]
    cand, _ = trainer.train(base, torch.rand(1, 1, 256, 256), y, None, steps=3, lr=1e-2, eval_every=1,
                            score_fn=lambda m: 0.0, amp=False)  # never better than the start
    assert all(torch.equal(p, q) for p, q in zip(cand.parameters(), base.parameters()))


def test_image_split_is_content_based():
    a = np.random.default_rng(0).random((256, 256)).astype(np.float32)
    assert trainer.image_split(a) == trainer.image_split(a.copy())
    splits = [trainer.image_split(np.full((4, 4), i, np.float32)) for i in range(500)]
    assert 60 < splits.count("val") < 140


def test_heavy_mode_trains_decoder_and_keeps_encoder():
    torch.manual_seed(0)
    base = learner.build_model(base_channels=8).eval()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    y = torch.from_numpy(blastocyst_mask().astype(np.int64))[None].repeat(3, 1, 1)
    anchors = (torch.rand(4, 1, 256, 256), blastocyst_mask()[None].repeat(4, 0).astype(np.int64))
    cand, history = trainer.train_epochs(base, torch.rand(3, 1, 256, 256), y, anchors, mode="heavy", epochs=2,
                                         lr=1e-2, amp=False)
    assert len(history) == 2 and all(np.isfinite(h["loss"]) for h in history)
    shifts = learner.layer_shifts(base, cand, before)
    assert shifts["encoder_frozen"]["l2_shift"] == 0.0
    assert shifts["decoder_frozen"]["l2_shift"] > 0  # up1-up3 move in heavy mode only
    light, _ = trainer.train_epochs(base, torch.rand(3, 1, 256, 256), y, None, mode="light", epochs=1, lr=1e-2, amp=False)
    assert learner.layer_shifts(base, light, before)["decoder_frozen"]["l2_shift"] == 0.0


def test_train_epochs_early_stops_and_keeps_best():
    base = learner.build_model(base_channels=8).eval()
    y = torch.from_numpy(blastocyst_mask().astype(np.int64))[None]
    cand, history = trainer.train_epochs(base, torch.rand(1, 1, 256, 256), y, None, mode="light", epochs=10, lr=1e-2,
                                         amp=False, score_fn=lambda m: 0.0, patience=2)
    assert len(history) == 2  # no gain over the start for 2 epochs
    assert all(torch.equal(p, q) for p, q in zip(cand.parameters(), base.parameters()))
