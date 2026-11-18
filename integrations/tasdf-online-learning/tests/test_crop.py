import numpy as np

from tasdf_online.crop import crop_box, crop_input, from_box, to_box


def poly(cx, cy, r=2.0, label="ZP"):
    pts = [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]
    return {"id": "a", "type": "polygonlabels", "from_name": "pl-0", "to_name": "timelapse_0",
            "value": {"points": pts, "polygonlabels": [label]}}


def test_crop_box_centres_on_regions_and_clamps():
    assert crop_box([poly(90, 90)], 964, 1288) == (364, 688, 964, 1288)


def test_crop_box_small_image_and_no_regions():
    assert crop_box([poly(50, 50)], 400, 300) == (0, 0, 400, 300)
    assert crop_box([], 964, 1288) == (182, 344, 782, 944)


def test_box_roundtrip_and_passthrough():
    rect = {"id": "r", "type": "rectanglelabels", "value": {"x": 1, "y": 2, "width": 3, "height": 4}}
    ring = poly(50, 50, 10)
    ring["value"]["holes"] = [[[48, 48], [52, 48], [52, 52]]]
    box = crop_box([ring], 964, 1288)
    back = from_box(to_box([ring, rect], 964, 1288, box), 964, 1288, box)
    assert back[1] is rect
    for p, q in zip(back[0]["value"]["points"] + back[0]["value"]["holes"][0],
                    ring["value"]["points"] + ring["value"]["holes"][0]):
        assert abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6
    assert back[0]["original_width"] == 964 and ring["value"]["points"][0] == [40.0, 40.0]


def test_crop_input_is_256_float():
    x = crop_input(np.full((1288, 964), 255, np.uint8), (364, 688, 964, 1288))
    assert x.shape == (256, 256) and x.dtype == np.float32 and float(x.max()) == 1.0
