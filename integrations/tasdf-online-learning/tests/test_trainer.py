import numpy as np
import pytest
import torch
from conftest import blastocyst_mask

from tasdf_online import learner, trainer
from tasdf_online.ls_format import IGNORE, mask_to_results


def _ann(ann_id, result, quality=2, updated="2026-11-12T10:00", cancelled=False):
    return {"id": ann_id, "result": result, "quality_level": quality, "updated_at": updated, "was_cancelled": cancelled}


FULL = mask_to_results(blastocyst_mask(), 256, 256, "seg", "image")


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


def test_mean_dice_ignore_skips_classes_the_annotator_did_not_draw():
    y = blastocyst_mask()
    bc_only = np.where(y == 4, 4, IGNORE).astype(np.uint8)  # only BC drawn, no ZP: the rest is IGNORE
    pred = y.copy()
    pred[y == 4] = 4
    pred[np.argwhere(y == 4)[0][0], np.argwhere(y == 4)[0][1]] = 3  # one stray ICM pixel on drawn BC
    assert trainer.mean_dice_ignore(pred, bc_only) > 0.99  # was 0.5: the stray pixel scored ICM 0


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


def test_recalibrate_bn_changes_only_running_stats():
    model = learner.build_model(base_channels=8).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer.recalibrate_bn(model, torch.rand(4, 1, 256, 256) * 0.2)
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all("running_" in k or "num_batches" in k for k in changed)
    assert not model.training


def test_bootstrap_std_shrinks_with_more_images():
    rng = np.random.default_rng(0)
    assert trainer.bootstrap_std([0.1]) == 0.0
    few, many = list(rng.normal(0, 0.05, 5)), list(rng.normal(0, 0.05, 200))
    assert trainer.bootstrap_std(many) < trainer.bootstrap_std(few)


def _decide(cand_val, prod_val, cand_train=None, **kw):
    args = dict(d10_ok=True, d10_reasons=[], encoder_moved=False, val_min=10, min_gain=0.005, max_gap=0.15)
    args.update(kw)
    return trainer.promotion_decision(cand_val=cand_val, prod_val=prod_val, cand_train=cand_train, **args)


def test_promotion_requires_a_gain_beyond_noise():
    prod = [0.80] * 12
    ok, reasons, note = _decide([0.86] * 12, prod, cand_train=0.9)
    assert ok and not reasons and note is None
    noisy = [0.80 + (0.3 if i % 2 else -0.28) for i in range(12)]  # mean +0.01 but huge spread
    ok, reasons, _ = _decide(noisy, prod, cand_train=0.9)
    assert not ok and "noise-aware" in reasons[0]


def test_promotion_blocks_memorization_forgetting_and_encoder_drift():
    ok, reasons, _ = _decide([0.86] * 12, [0.80] * 12, cand_train=0.99 + 0.1)
    assert not ok and any("memorization" in r for r in reasons)
    ok, reasons, _ = _decide([0.86] * 12, [0.80] * 12, d10_ok=False, d10_reasons=["mDice vs v0 -0.06"])
    assert not ok and reasons == ["mDice vs v0 -0.06"]
    ok, reasons, _ = _decide([0.86] * 12, [0.80] * 12, encoder_moved=True)
    assert not ok and "encoder" in reasons[0]


def test_small_or_missing_val_is_never_promoted():
    ok, reasons, note = _decide([0.95, 0.95], [0.80, 0.80])  # a big gain on 2 images is still no evidence
    assert not ok and note.startswith("insufficient_val") and "not promoted" in reasons[0]
    ok, reasons, note = _decide([], [])
    assert not ok and note == "no_val"


def test_cool_down_waits_only_when_hot():
    naps = []
    assert trainer.cool_down(temp=lambda: 60, sleep=naps.append) == 0.0
    temps = iter([85, 78, 72, 65])
    assert trainer.cool_down(temp=lambda: next(temps), sleep=naps.append) == 30.0 and naps == [10, 10, 10]
    assert trainer.cool_down(temp=lambda: None, sleep=naps.append) == 0.0
