import cv2
import numpy as np
import pytest
from label_studio_converter.brush import decode_rle

from opencv_segmentation.app import MODEL_VERSION, create_app
from opencv_segmentation.config import Settings
from opencv_segmentation.service import _pin_external_url, _resolve_image_url, parse_prompts


LABEL_CONFIG = """
<View>
  <Image name="scan" value="$scan_url"/>
  <BrushLabels name="membrane" toName="scan">
    <Label value="Cell"/>
    <Label value="Background"/>
  </BrushLabels>
  <RectangleLabels name="box" toName="scan"><Label value="Cell"/></RectangleLabels>
  <KeyPointLabels name="point" toName="scan"><Label value="Cell"/><Label value="Background"/></KeyPointLabels>
  <PolygonLabels name="outline" toName="scan"><Label value="Cell"/></PolygonLabels>
</View>
"""

VALUE_LIST_CONFIG = """
<View>
  <Image name="scan" valueList="$images"/>
  <BrushLabels name="brush" toName="scan"><Label value="Object"/><Label value="Membrane"/></BrushLabels>
</View>
"""

DYNAMIC_CONFIG = """
<View>
  <Image name="timelapse_{{idx}}" valueList="$timelapse[{{idx}}].image_urls"/>
  <BrushLabels name="membrane_{{idx}}" toName="timelapse_{{idx}}">
    <Label value="Object"/><Label value="Membrane"/>
  </BrushLabels>
</View>
"""


@pytest.fixture
def synthetic_image():
    image = np.full((120, 160, 3), 25, dtype=np.uint8)
    cv2.circle(image, (80, 60), 28, (225, 225, 225), -1)
    return image


@pytest.fixture
def settings():
    return Settings(
        label_studio_url="http://label-studio:8080",
        label_studio_api_key="test-key",
        allowed_image_hosts=frozenset({"images.test", "label-studio"}),
        request_timeout=1,
        max_image_bytes=1_000_000,
        max_image_pixels=1_000_000,
        max_tasks=4,
        max_request_bytes=1_000_000,
        default_ring=False,
        ring_width=3,
    )


@pytest.fixture
def app(monkeypatch, synthetic_image, settings):
    monkeypatch.setattr("opencv_segmentation.service.fetch_image", lambda _source, _settings: synthetic_image.copy())
    application = create_app(settings)
    application.config["TESTING"] = True
    return application


def _payload(ring=False):
    return {
        "tasks": [{"id": 1, "data": {"scan_url": "https://images.test/cell.png"}}],
        "label_config": LABEL_CONFIG,
        "params": {
            "context": {
                "ring": ring,
                "ring_width": 4,
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {"x": 25, "y": 15, "width": 50, "height": 70, "rectanglelabels": ["Cell"]},
                    },
                    {"type": "keypointlabels", "value": {"x": 50, "y": 50, "keypointlabels": ["Cell"]}},
                    {
                        "type": "keypointlabels",
                        "value": {"x": 28, "y": 18, "keypointlabels": ["Background"]},
                    },
                ],
            }
        },
    }


def _decoded_mask(response):
    prediction = response.get_json()["results"][0]
    result = prediction["result"][0]
    decoded = decode_rle(result["value"]["rle"])
    return decoded.reshape(result["original_height"], result["original_width"], 4)[:, :, 3]


def test_health_and_setup(app):
    client = app.test_client()
    health = client.get("/health")
    setup = client.post("/setup", json={"schema": LABEL_CONFIG, "project": "1.1"})

    assert health.status_code == 200
    assert health.get_json()["status"] == "UP"
    assert setup.status_code == 200
    assert setup.get_json()["model_version"] == MODEL_VERSION


def test_validate_rejects_config_without_brush(app):
    response = app.test_client().post("/validate", json={"label_config": '<View><Image name="i" value="$image"/></View>'})
    assert response.status_code == 400


def test_predict_rectangle_returns_non_empty_rle(app):
    response = app.test_client().post("/predict", json=_payload())
    assert response.status_code == 200
    result = response.get_json()["results"][0]["result"][0]
    assert result["from_name"] == "membrane"
    assert result["to_name"] == "scan"
    assert result["value"]["brushlabels"] == ["Cell"]
    assert np.count_nonzero(_decoded_mask(response)) > 0


def test_polygon_prompt_returns_edge_traced_polygon(app):
    payload = _payload()
    payload["params"]["context"]["result"] = [
        {
            "type": "polygonlabels",
            "to_name": "scan",
            "value": {
                "points": [[50, 25], [68, 50], [50, 75], [32, 50]],
                "polygonlabels": ["Cell"],
            },
        }
    ]

    response = app.test_client().post("/predict", json=payload)
    result = response.get_json()["results"][0]["result"][0]

    assert response.status_code == 200
    assert result["from_name"] == "outline"
    assert result["type"] == "polygonlabels"
    assert result["value"]["polygonlabels"] == ["Cell"]
    assert result["value"]["closed"] is True
    assert len(result["value"]["points"]) >= 4


def test_ring_mode_has_empty_center(app):
    response = app.test_client().post("/predict", json=_payload(ring=True))
    assert response.status_code == 200
    mask = _decoded_mask(response)
    assert np.count_nonzero(mask) > 0
    assert mask[60, 80] == 0
    assert mask[60, 106] > 0


def test_label_studio_relative_media_urls_are_supported(settings):
    local_url, local_auth = _resolve_image_url("/data/local-files/?d=sample.png", settings)
    upload_url, upload_auth = _resolve_image_url("/data/upload/1/sample.png", settings)

    assert local_url == "http://label-studio:8080/data/local-files/?d=sample.png"
    assert upload_url == "http://label-studio:8080/data/upload/1/sample.png"
    assert local_auth is True
    assert upload_auth is True


def test_browser_localhost_media_url_uses_internal_label_studio_url(settings):
    url, requires_auth = _resolve_image_url(
        "http://localhost:8080/data/local-files?d=timelapse/frame.jpg",
        settings,
    )

    assert url == "http://label-studio:8080/data/local-files?d=timelapse/frame.jpg"
    assert requires_auth is True


def test_top_level_negative_keypoint_is_background_prompt():
    context = {
        "result": [
            {
                "type": "keypointlabels",
                "is_positive": False,
                "value": {"x": 25, "y": 50, "keypointlabels": ["Object"]},
            }
        ]
    }

    _, positives, backgrounds, _ = parse_prompts(context, 200, 100)

    assert positives == []
    assert backgrounds == [(50, 50)]


def test_external_url_is_pinned_to_validated_ip(settings, monkeypatch):
    monkeypatch.setattr(
        "opencv_segmentation.service.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )

    pinned_url, host_header = _pin_external_url("https://images.test:8443/sample.png?token=1", settings)

    assert pinned_url == "https://93.184.216.34:8443/sample.png?token=1"
    assert host_header == "images.test"


def test_simple_value_list_uses_prompt_item_index(app, monkeypatch, synthetic_image):
    fetched = []
    monkeypatch.setattr(
        "opencv_segmentation.service.fetch_image",
        lambda source, _settings: fetched.append(source) or synthetic_image.copy(),
    )
    payload = {
        "tasks": [{"data": {"images": ["https://images.test/first.png", "https://images.test/second.png"]}}],
        "label_config": VALUE_LIST_CONFIG,
        "params": {
            "context": {
                "result": [
                    {
                        "to_name": "scan",
                        "item_index": 1,
                        "type": "rectanglelabels",
                        "value": {"x": 25, "y": 15, "width": 50, "height": 70, "rectanglelabels": ["Object"]},
                    }
                ]
            }
        },
    }

    response = app.test_client().post("/predict", json=payload)
    result = response.get_json()["results"][0]["result"][0]
    assert response.status_code == 200
    assert fetched == ["https://images.test/second.png"]
    assert result["item_index"] == 1
    assert result["value"]["brushlabels"] == ["Object"]


def test_nested_timelapse_expression_renders_dynamic_names(app, monkeypatch, synthetic_image):
    fetched = []
    monkeypatch.setattr(
        "opencv_segmentation.service.fetch_image",
        lambda source, _settings: fetched.append(source) or synthetic_image.copy(),
    )
    payload = {
        "tasks": [
            {
                "data": {
                    "timelapse": [
                        {"image_urls": ["https://images.test/t0-f0.png"]},
                        {"image_urls": ["https://images.test/t1-f0.png", "https://images.test/t1-f1.png"]},
                    ]
                }
            }
        ],
        "label_config": DYNAMIC_CONFIG,
        "params": {
            "context": {
                "result": [
                    {
                        "from_name": "box_1",
                        "to_name": "timelapse_1",
                        "item_index": 1,
                        "type": "rectanglelabels",
                        "value": {"x": 25, "y": 15, "width": 50, "height": 70, "rectanglelabels": ["Object"]},
                    }
                ]
            }
        },
    }

    response = app.test_client().post("/predict", json=payload)
    result = response.get_json()["results"][0]["result"][0]
    assert response.status_code == 200
    assert fetched == ["https://images.test/t1-f1.png"]
    assert result["from_name"] == "membrane_1"
    assert result["to_name"] == "timelapse_1"
    assert result["item_index"] == 1


def test_membrane_label_enables_ring_and_context_false_overrides(app):
    payload = {
        "tasks": [{"data": {"images": ["https://images.test/cell.png"]}}],
        "label_config": VALUE_LIST_CONFIG,
        "params": {
            "context": {
                "label": "membrane",
                "result": [
                    {
                        "to_name": "scan",
                        "item_index": 0,
                        "type": "rectanglelabels",
                        "value": {"x": 25, "y": 15, "width": 50, "height": 70},
                    }
                ],
            }
        },
    }

    automatic = app.test_client().post("/predict", json=payload)
    automatic_mask = _decoded_mask(automatic)
    assert automatic.get_json()["results"][0]["result"][0]["value"]["brushlabels"] == ["Membrane"]
    assert automatic_mask[60, 80] == 0
    assert automatic_mask[60, 106] > 0

    payload["params"]["context"]["ring"] = False
    overridden = app.test_client().post("/predict", json=payload)
    assert _decoded_mask(overridden)[60, 80] > 0
