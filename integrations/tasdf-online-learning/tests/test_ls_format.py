import numpy as np
import pytest
from conftest import blastocyst_mask

from tasdf_online.learner import mean_dice
from tasdf_online.ls_format import BC, ICM, TE, ZP, bio_valid, ls_media_url, mask_to_results, results_to_mask


def test_round_trip_keeps_rings_and_classes():
    mask = blastocyst_mask()
    results = mask_to_results(mask, 800, 600, "seg", "image")
    rings = {r["value"]["polygonlabels"][0] for r in results if r["value"].get("ring")}
    assert {"ZP", "TE"} <= rings
    assert all(len(r["value"]["holes"]) == 1 for r in results if r["value"].get("ring"))
    assert all(r["original_width"] == 800 and r["type"] == "polygonlabels" for r in results)
    back = results_to_mask(results, "seg")
    mdice, per = mean_dice(back, mask)
    assert mdice >= 0.95, per
    assert min(per.values()) >= 0.9, per


def test_ring_around_background_keeps_its_own_hole():
    # regression: ZP ring enclosing background + an ICM blob (no TE); the hole must be ZP's interior, not the ICM
    mask = blastocyst_mask()
    mask[(mask == TE) | (mask == BC)] = 0
    back = results_to_mask(mask_to_results(mask, 256, 256, "seg", "image"), "seg")
    _, per = mean_dice(back, mask)
    assert per[ZP] >= 0.9 and per[ICM] >= 0.9, per


def test_painter_order_icm_over_bc():
    square = lambda x0, x1: [[x0, x0], [x1, x0], [x1, x1], [x0, x1]]  # noqa: E731
    results = [
        {"type": "polygonlabels", "from_name": "seg", "value": {"points": square(10, 90), "polygonlabels": ["BC"]}},
        {"type": "polygonlabels", "from_name": "seg", "value": {"points": square(40, 60), "polygonlabels": ["ICM"]}},
    ]
    m = results_to_mask(results, "seg")
    assert m[128, 128] == ICM and m[50, 50] == BC


@pytest.mark.parametrize("holes", [[[[30, 30], [70, 30], [70, 70], [30, 70]]], [[30, 30], [70, 30], [70, 70], [30, 70]]])
def test_ring_hole_formats(holes):
    value = {"points": [[10, 10], [90, 10], [90, 90], [10, 90]], "polygonlabels": ["TE"], "ring": True, "holes": holes}
    m = results_to_mask([{"type": "polygonlabels", "from_name": "seg", "value": value}], "seg")
    assert m[128, 128] == 0 and m[40, 40] == TE


def test_other_controls_ignored():
    r = {"type": "polygonlabels", "from_name": "other", "value": {"points": [[0, 0], [99, 0], [99, 99]], "polygonlabels": ["ZP"]}}
    assert results_to_mask([r], "seg").max() == 0
    assert ZP == 1


def test_media_url_only_label_studio():
    assert ls_media_url("http://webapp:8080", "/data/local-files/?d=AI01/a.png") == "http://webapp:8080/data/local-files/?d=AI01/a.png"
    assert ls_media_url("http://webapp:8080", "http://webapp:8080/data/upload/1/a.png").endswith("/data/upload/1/a.png")
    with pytest.raises(ValueError):
        ls_media_url("http://webapp:8080", "http://evil.example/data/x.png")
    with pytest.raises(ValueError):
        ls_media_url("http://webapp:8080", "/api/projects/")


def test_brush_rle_round_trip_with_reference_encoder():
    import cv2

    brush = pytest.importorskip("label_studio_converter.brush")
    full = cv2.resize(blastocyst_mask(), (320, 240), interpolation=cv2.INTER_NEAREST)  # non-square original size
    results = [
        {"type": "brushlabels", "from_name": "mask", "original_width": 320, "original_height": 240,
         "value": {"format": "rle", "rle": brush.mask2rle(((full == c) * 255).astype(np.uint8)), "brushlabels": [name]}}
        for c, name in ((ZP, "ZP"), (TE, "TE"), (BC, "BC"), (ICM, "ICM"))
    ]
    expected = cv2.resize(full, (256, 256), interpolation=cv2.INTER_NEAREST)
    mdice, per = mean_dice(results_to_mask(results, "seg"), expected)
    assert mdice >= 0.98, per


def test_bio_valid_rejects_empty_and_saturated():
    assert bio_valid(blastocyst_mask())
    assert not bio_valid(np.zeros((256, 256), np.uint8))
    assert not bio_valid(np.full((256, 256), ZP, np.uint8))


def test_fetch_image_data_uri_and_multi_tool_mask():
    import base64
    import cv2
    from tasdf_online.ls_format import fetch_image

    raw = np.full((64, 80), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", raw)
    assert ok
    uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    img, w, h = fetch_image("", "", uri)
    assert img.shape == (256, 256) and (w, h) == (80, 64)

    results = [
        {"type": "rectanglelabels", "from_name": "rectangle", "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": ["ZP"]}},
        {"type": "ellipsebasedpolygonlabels", "from_name": "ellipse", "value": {"points": [[50, 50], [70, 50], [70, 70], [50, 70]], "ellipsebasedpolygonlabels": ["ICM"]}},
    ]
    m = results_to_mask(results, "polygon")
    assert m[40, 40] == ZP and m[150, 150] == ICM

