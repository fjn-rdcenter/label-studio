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
    assert "input_overlay" in tel["last_visuals"] and tel["last_visuals"]["input_overlay"].startswith("data:image/png;base64,")
    # same result again is not retrained
    assert client.post("/webhook", json=_event()).get_json()["status"] == "ignored-unchanged"
    # degenerate (empty or saturated >=95%) mask is rejected by bio_valid gate
    sat = [{"type": "polygonlabels", "from_name": "seg", "value": {"points": [[0, 0], [100, 0], [100, 100], [0, 100]], "polygonlabels": ["ZP"]}}]
    assert client.post("/webhook", json={**_event(ann_id=99), "annotation": {"id": 99, "result": sat, "quality_level": 2}}).get_json()["status"] == "ignored-invalid-mask"


def test_dashboard_toggle_and_standalone(client):
    assert client.get("/dashboard").status_code == 200
    assert client.post("/dashboard/toggle", json={"enabled": False}).status_code == 401
    assert client.post("/dashboard/toggle", json={"enabled": False}, headers=AUTH).get_json()["dashboard_enabled"] is False
    assert client.get("/dashboard").status_code == 404
    assert "last_visuals" not in client.get("/telemetry/live", headers=AUTH).get_json()
    assert client.post("/dashboard/toggle", json={"enabled": True}, headers=AUTH).get_json()["dashboard_enabled"] is True
    assert client.get("/dashboard").status_code == 200


@pytest.mark.parametrize("next_n", [1, 0])
def test_refresh_ls_creates_predictions_before_patch(settings, monkeypatch, next_n):
    img = np.zeros((256, 256), np.float32)
    monkeypatch.setattr(app_module, "fetch_image", lambda *a, **k: (img, 256, 256))
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="tok", ls_project_id=7, refresh_next_n=next_n)
    svc = app_module.OnlineService(s)
    calls = []

    def fake_get(url, **kw):
        class R:
            status_code = 200 if kw.get("params", {}).get("page") == 1 else 404
            def raise_for_status(self): pass
            def json(self):
                return [{"id": 10, "data": {"image": "/data/a.png"}, "annotations": [], "predictions": [{"id": 3, "model_version": "old"}]}] if self.status_code == 200 else []
        return R()

    def fake_req(method, url, **kw):
        calls.append((method, url))
        class R:
            status_code = 200
            def raise_for_status(self): pass
        return R()

    monkeypatch.setattr(app_module.requests, "get", fake_get)
    monkeypatch.setattr(app_module.requests, "request", fake_req)
    stats = svc.refresh_ls()
    expected_created = 1 if next_n > 0 else 0
    assert stats == {"created": expected_created, "errors": 0}
    expected_calls = ["POST", "PATCH"] if next_n > 0 else ["PATCH"]
    assert [m for m, _ in calls] == expected_calls


def test_refresh_ls_skips_when_already_running(settings):
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="tok")
    svc = app_module.OnlineService(s)
    with svc.refresh_lock:
        assert svc.refresh_ls() == {"skipped": "a refresh is already running"}


def test_forged_webhook_is_checked_against_label_studio(settings, monkeypatch):
    img = np.zeros((256, 256), np.float32)
    monkeypatch.setattr(app_module, "fetch_image", lambda *a, **k: (img, 256, 256))
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="tok")
    svc = app_module.OnlineService(s)
    real = {"id": 11, "task": 5, "project": 7, "quality_level": 1, "was_cancelled": False,
            "result": mask_to_results(blastocyst_mask(), 256, 256, "seg", "image")}

    def fake_req(method, url, **kw):
        class R:
            def raise_for_status(self): pass
            def json(self):
                return real if "/api/annotations/" in url else {"id": 5, "data": {"image": "/data/a.png"}}
        return R()

    monkeypatch.setattr(app_module.requests, "request", fake_req)
    assert svc.enqueue(_event(quality=3)) == "queued"  # payload claims level 3, LS says level 1
    deadline = time.time() + 30
    while time.time() < deadline and "last_skipped" not in svc.telemetry:
        time.sleep(0.1)
    assert svc.telemetry["last_skipped"] == {"annotation_id": 11, "reason": "ignored-quality"}
    assert svc.telemetry["episodes"] == []


def test_webhook_without_project_is_ignored(client):
    event = _event()
    del event["project"]
    assert client.post("/webhook", json=event).get_json()["status"] == "ignored-project"


def test_predict_survives_one_bad_task(client, monkeypatch):
    good = np.zeros((256, 256), np.float32)

    def fake_fetch(ls_url, token, source, timeout=30):
        if "bad" in source:
            raise ValueError("broken image")
        return good, 256, 256

    monkeypatch.setattr(app_module, "fetch_image", fake_fetch)
    tasks = [{"id": 1, "data": {"image": "/data/ok.png"}}, {"id": 2, "data": {"image": "/data/bad.png"}}]
    body = client.post("/predict", json={"tasks": tasks}).get_json()
    assert len(body["results"]) == 2 and body["results"][1]["result"] == []


def test_setup_from_other_project_keeps_label_config(client):
    svc = client.application.config["svc"]
    before = svc.label
    other = '<View><Image name="x" value="$pic"/><PolygonLabels name="q" toName="x"><Label value="ZP"/></PolygonLabels></View>'
    client.post("/setup", json={"project": "99.1700000000", "schema": other})
    assert svc.label == before
    client.post("/setup", json={"project": "7.1700000000", "schema": other})
    assert svc.label == ("q", "x", "pic")


def test_rollback_validates_version(client):
    assert client.post("/rollback", json={"version": "../../etc"}, headers=AUTH).status_code == 400
    assert client.post("/rollback", json={"version": "v5"}, headers=AUTH).status_code == 404


def test_ops_token_not_accepted_in_query(client):
    assert client.get("/telemetry/live?token=secret").status_code == 401
    assert client.get("/telemetry/live", headers=AUTH).status_code == 200


def test_mask_cache_bounded_and_consumed(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_CACHED_MASKS", 8)  # keep the test fast; the bound logic is size-independent
    svc = client.application.config["svc"]
    n = app_module.MAX_CACHED_MASKS + 5
    for i in range(n):
        svc.predict_task({"id": i, "data": {"image": "/data/x.png"}})
    assert len(svc.last_predicted_mask) == app_module.MAX_CACHED_MASKS
    assert 0 not in svc.last_predicted_mask and n - 1 in svc.last_predicted_mask
    assert svc._initial_prediction({}, n - 1) is not None
    assert n - 1 not in svc.last_predicted_mask  # popped once used for training


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


def test_ops_open_without_ol_token(settings):
    s = dataclasses.replace(settings, ol_token="")
    c = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert c.get("/telemetry/live").status_code == 200


def test_webhook_not_trusted_without_ls_token(settings, monkeypatch):
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="")
    svc = app_module.OnlineService(s)
    monkeypatch.setattr(app_module.requests, "request", lambda *a, **k: pytest.fail("must not call LS without a token"))
    assert svc.enqueue(_event(quality=3)) == "queued"
    deadline = time.time() + 30
    while time.time() < deadline and "last_error" not in svc.telemetry:
        time.sleep(0.1)
    assert "LS_TOKEN" in svc.telemetry["last_error"]
    assert svc.telemetry["episodes"] == []
