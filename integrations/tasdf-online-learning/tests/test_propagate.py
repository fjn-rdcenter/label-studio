import cv2
import numpy as np

from tasdf_online.propagate import boundary_f, object_result, refine_object, stop_reason


def disk(r, c=(128, 128)):
    m = np.zeros((256, 256), np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(bool)


def holes(m):
    _, h = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return 0 if h is None else int((h[0][:, 3] != -1).sum())


def test_refine_never_leaves_the_band():
    out = refine_object(disk(60), np.ones((256, 256), bool), 8, ring=False)
    assert not (out & ~disk(69)).any()  # band 8 px around r=60, +1 px for raster rounding


def test_refine_keeps_core_when_model_sees_nothing():
    out = refine_object(disk(60), np.zeros((256, 256), bool), 8, ring=False)
    assert out.sum() > 0 and not (out & ~disk(60)).any()


def test_refine_drops_far_junk():
    junk = disk(60) | disk(10, (20, 20))
    assert not refine_object(disk(60), junk, 8, ring=False)[20, 20]


def test_ring_keeps_exactly_one_hole():
    ring = disk(80) & ~disk(60)
    model = ring.copy()
    model[128, 128] = False
    assert holes(refine_object(ring, model, 8, ring=True)) == 1


def test_boundary_f():
    assert boundary_f(disk(60), disk(60)) == 1.0
    assert boundary_f(disk(60), disk(60, (138, 128))) < 0.5
    assert boundary_f(np.zeros((256, 256), bool), np.zeros((256, 256), bool)) == 1.0


def test_stop_reasons_in_priority_order():
    c = disk(60)
    assert stop_reason(c, np.zeros_like(c), c, False, 0.8) == "vanished"
    assert stop_reason(c, disk(90), c, False, 0.8) == "area"
    assert stop_reason(c, c, c, True, 0.8) == "ring"
    assert stop_reason(c, c, disk(60, (150, 128)), False, 0.8) == "disagree"
    assert stop_reason(c, c, c, False, 0.8) is None


def test_object_result_keeps_identity_and_ring():
    tpl = {"id": "zp1", "type": "polygonlabels", "from_name": "pl-3", "to_name": "timelapse_3",
           "value": {"points": [], "polygonlabels": ["ZP"], "ring": True}}
    r = object_result(disk(80) & ~disk(60), tpl, 1)
    assert r["id"] == "zp1" and r["from_name"] == "pl-3" and r["value"]["polygonlabels"] == ["ZP"]
    assert len(r["value"]["holes"]) == 1
    assert object_result(np.zeros((256, 256), bool), tpl, 1) is None


def test_boundary_f_of_a_mask_filling_the_frame_is_finite():
    full = np.ones((256, 256), bool)
    assert boundary_f(full, full) == 1.0
    assert 0.0 <= boundary_f(full, disk(60)) < 0.5
