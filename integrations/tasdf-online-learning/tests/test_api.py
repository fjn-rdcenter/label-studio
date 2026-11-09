import dataclasses
import time

import numpy as np
import pytest
import torch
from conftest import blastocyst_mask

from tasdf_online import app as app_module
from tasdf_online.ls_format import mask_to_results

AUTH = {"X-OL-Token": "secret"}


@pytest.fixture
def client(settings, monkeypatch):
    img = np.random.default_rng(1).random((256, 256)).astype(np.float32)
    monkeypatch.setattr(app_module, "fetch_image", lambda *a, **k: (img, 800, 600))
    application = app_module.create_app(settings, app_module.OnlineService(settings))
    return application.test_client()


def _event(quality=2, ann_id=11, action="ANNOTATION_CREATED"):
    result = mask_to_results(blastocyst_mask(), 800, 600, "seg", "image")
    return {"action": action, "project": {"id": 7}, "task": {"id": 5, "data": {"image": "/data/local-files/?d=a.png"}},
            "annotation": {"id": ann_id, "result": result, "quality_level": quality, "parent_prediction": None}}


def test_health_setup_predict(client):
    assert client.get("/health").get_json()["model_version"] == "tasdf-v0"
    cfg = '<View><Image name="img" value="$picture"/><PolygonLabels name="p" toName="img"><Label value="ZP"/></PolygonLabels></View>'
    assert client.post("/setup", json={"schema": cfg}).get_json()["model_version"] == "tasdf-v0"
    body = client.post("/predict", json={"tasks": [{"id": 1, "data": {"picture": "/data/x.png"}}]}).get_json()
    (res,) = body["results"]
    assert res["model_version"] == "tasdf-v0" and isinstance(res["result"], list) and 0.0 <= res["score"] <= 1.0
    assert all(r["from_name"] == "p" and r["to_name"] == "img" for r in res["result"])
    assert client.post("/predict", json={"tasks": []}).status_code == 400


def test_webhook_fast_filters_and_trains(client):
    t0 = time.perf_counter()
    assert client.post("/webhook", json=_event()).get_json()["status"] == "queued"
    assert (time.perf_counter() - t0) < 0.05
    assert client.post("/webhook", json=_event(quality=1, ann_id=12)).get_json()["status"] == "ignored-quality"
    assert client.post("/webhook", json={"action": "TASKS_CREATED"}).get_json()["status"] == "ignored-action"
    assert client.post("/webhook", json={**_event(ann_id=13), "project": {"id": 99}}).get_json()["status"] == "ignored-project"

    assert client.get("/telemetry/live").status_code == 401
    deadline = time.time() + 120
    while time.time() < deadline:
        tel = client.get("/telemetry/live", headers=AUTH).get_json()
        if tel["episodes"] or tel.get("last_error"):
            break
        time.sleep(0.5)
    assert "last_error" not in tel, tel.get("last_error")
    (ep,) = tel["episodes"]
    assert ep["frozen_unchanged"] and "metrics" in ep and ep["version"] == "v1"
    assert ep["layer_shifts"]["pixel_head"]["l2_shift"] > 0
    # same result again is not retrained
    assert client.post("/webhook", json=_event()).get_json()["status"] == "ignored-unchanged"


def test_rollback_requires_token(client):
    assert client.post("/rollback", json={"version": "v0"}).status_code == 401
    body = client.post("/rollback", json={"version": "v0"}, headers=AUTH).get_json()
    assert body["active"] == "tasdf-v0" and "skipped" in body["refresh"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predict_on_cuda(settings, monkeypatch):
    img = np.zeros((256, 256), np.float32)
    monkeypatch.setattr(app_module, "fetch_image", lambda *a, **k: (img, 256, 256))
    s = dataclasses.replace(settings, device="cuda", amp="auto")
    client = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert client.get("/health").get_json()["device"].startswith("cuda")
    assert client.post("/predict", json={"tasks": [{"id": 1, "data": {"image": "/data/x.png"}}]}).status_code == 200
