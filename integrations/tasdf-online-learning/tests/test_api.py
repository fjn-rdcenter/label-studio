import dataclasses
import time

import numpy as np
import pytest
import torch
from conftest import blastocyst_mask

from tasdf_online import app as app_module
from tasdf_online import trainer
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
    # N=0: list the project's backends first so LS runs /setup and accepts the new version on PATCH
    expected_calls = ["POST", "PATCH"] if next_n > 0 else ["GET", "PATCH"]
    assert [m for m, _ in calls] == expected_calls


def test_refresh_ls_reports_failed_patch(settings, monkeypatch):
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="tok", ls_project_id=7, refresh_next_n=0)
    svc = app_module.OnlineService(s)

    def fake_req(method, url, **kw):
        class R:
            def raise_for_status(self):
                if method == "PATCH":
                    raise app_module.requests.HTTPError("400 model_version is not valid")
        return R()

    monkeypatch.setattr(app_module.requests, "request", fake_req)
    stats = svc.refresh_ls()
    assert "400" in stats["patch_failed"]
    assert svc.telemetry["last_refresh"]["patch_failed"] == stats["patch_failed"]


def test_refresh_ls_skips_when_already_running(settings):
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="tok")
    svc = app_module.OnlineService(s)
    with svc.refresh_lock:
        assert svc.refresh_ls() == {"skipped": "a refresh is already running"}


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
    assert svc.label[:3] == ("q", "x", "pic")


def test_rollback_validates_version(client):
    assert client.post("/rollback", json={"version": "../../etc"}, headers=AUTH).status_code == 400
    assert client.post("/rollback", json={"version": "v5"}, headers=AUTH).status_code == 404


def test_ops_token_not_accepted_in_query(client):
    assert client.get("/telemetry/live?token=secret").status_code == 401
    assert client.get("/telemetry/live", headers=AUTH).status_code == 200


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


def test_ops_closed_without_ol_token_when_dashboard_off(settings):
    # product: DASHBOARD_ENABLED=0 and no OL_TOKEN -> nobody may roll back or refresh
    s = dataclasses.replace(settings, ol_token="", dashboard_enabled=False)
    c = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert c.post("/rollback", json={"version": "v0"}).status_code == 401
    assert c.post("/refresh_ls_predictions").status_code == 401
    assert c.get("/telemetry/live").status_code == 401


def test_dashboard_not_mounted_when_disabled(settings):
    # even with a valid OL_TOKEN the product cannot switch the dev dashboard on
    s = dataclasses.replace(settings, dashboard_enabled=False)
    c = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert c.get("/dashboard").status_code == 404
    assert c.post("/dashboard/toggle", json={"enabled": True}, headers=AUTH).status_code == 404


def test_missing_anchor_file_is_optional(settings, tmp_path, caplog):
    # compose always sets ANCHOR_NPZ; an environment without anchors.npz must still boot
    s = dataclasses.replace(settings, anchor_npz=str(tmp_path / "missing.npz"))
    with caplog.at_level("WARNING", logger="tasdf_online"):
        svc = app_module.OnlineService(s)
    assert svc.anchors is None
    assert "missing.npz" in caplog.text


def test_startup_logs_settings_without_secrets(settings, caplog):
    s = dataclasses.replace(settings, ls_token="ls-secret-value", ol_token="ol-secret-value")
    with caplog.at_level("INFO", logger="tasdf_online"):
        app_module.create_app(s, app_module.OnlineService(s))
    assert "settings:" in caplog.text and "holdout_limit" in caplog.text
    assert "ls-secret-value" not in caplog.text and "ol-secret-value" not in caplog.text


def test_output_labels_follow_the_project_aliases():
    alias = ('<View><Image name="timelapse_3" value="$image"/><PolygonLabels name="pl-3" toName="timelapse_3">'
             '<Label value="ZP"/><Label value="TE"/><Label value="ICM"/><Label value="Blastocoel"/></PolygonLabels></View>')
    label = app_module.parse_label_config(alias)
    assert label[:3] == ("pl-3", "timelapse_3", "image")
    names = {r["value"]["polygonlabels"][0] for r in mask_to_results(blastocyst_mask(), 256, 256, *label[:2], label[3])}
    assert "Blastocoel" in names and "BC" not in names
    # a config that defines BC keeps it
    assert app_module.parse_label_config(alias.replace("Blastocoel", "BC"))[3] == app_module.CLASS_NAMES


# ---- batch training ------------------------------------------------------------------------------
TRAIN_IDS = [t for t in range(1, 200) if trainer.split_of(7, t) == "train"][:3]
VAL_IDS = [t for t in range(1, 200) if trainer.split_of(7, t) == "val"][:2]


def _ls_tasks(ids, result=None, quality=2):
    result = result or mask_to_results(blastocyst_mask(), 256, 256, "seg", "image")
    return [{"id": t, "data": {"image": f"/data/{t}.png"}, "predictions": [],
             "annotations": [{"id": 100 + t, "result": result, "quality_level": quality, "was_cancelled": False,
                              "updated_at": "2026-11-12T10:00"}]} for t in ids]


@pytest.fixture
def ls_svc(settings, monkeypatch):
    img = np.random.default_rng(2).random((256, 256)).astype(np.float32)
    monkeypatch.setattr(app_module, "fetch_image", lambda *a, **k: (img, 256, 256))
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="tok", ls_project_id=7)
    tasks = {"list": _ls_tasks(TRAIN_IDS + VAL_IDS)}

    def fake_get(url, **kw):
        page = kw.get("params", {}).get("page")

        class R:
            status_code = 200 if page == 1 else 404
            def raise_for_status(self): pass
            def json(self): return tasks["list"] if page == 1 else []
        return R()

    monkeypatch.setattr(app_module.requests, "get", fake_get)
    monkeypatch.setattr(app_module.requests, "request", lambda *a, **k: type("R", (), {"raise_for_status": lambda self: None})())
    svc = app_module.OnlineService(s)
    svc.fake_tasks = tasks
    return svc


def test_job_trains_on_reviewed_annotations_and_records_both_metrics(ls_svc):
    rec = ls_svc.run_job(force=True, job_id="t1")
    assert "last_error" not in ls_svc.telemetry, ls_svc.telemetry.get("last_error")
    assert rec["version"] == "v1"
    assert sorted(rec["train_tasks"]) == sorted(TRAIN_IDS) and sorted(rec["val_tasks"]) == sorted(VAL_IDS)
    info = ls_svc.registry.info("v1")
    assert "metrics" in info and info["project_val_dice"] is not None and info["kind"] == "batch"
    assert "insufficient_project_val" in info["note"]  # 2 val tasks < PROJECT_VAL_MIN
    assert info["layer_shifts"]["encoder_frozen"]["l2_shift"] == 0.0
    # nothing new since the last job -> no retrain unless forced, and this survives a restart
    assert ls_svc.run_job(force=False) is None and ls_svc.telemetry["pending"]["new_train"] == 0
    again = app_module.OnlineService(ls_svc.s)
    assert set(again.registry.data["last_job"]["digests"]) == {str(t) for t in TRAIN_IDS}


def test_job_waits_for_n_new_annotations(ls_svc):
    ls_svc.s = dataclasses.replace(ls_svc.s, train_every_n=10)
    assert ls_svc.run_job(force=False) is None
    assert ls_svc.telemetry["pending"] == {"new_train": len(TRAIN_IDS), "train": len(TRAIN_IDS), "val": len(VAL_IDS), "threshold": 10}
    assert ls_svc.registry.active == "v0" and "last_job" not in ls_svc.registry.data


def test_job_skips_low_quality_annotations(ls_svc):
    ls_svc.fake_tasks["list"] = _ls_tasks(TRAIN_IDS, quality=1)
    assert ls_svc.run_job(force=True) is None
    assert ls_svc.telemetry["pending"]["train"] == 0


def test_job_needs_ls_credentials(settings):
    svc = app_module.OnlineService(dataclasses.replace(settings, ls_url="http://ls.test", ls_token=""))
    assert svc.run_job(force=True) is None
    assert "LS_TOKEN" in svc.telemetry["last_error"]


def test_webhook_filters_and_train_route(ls_svc):
    client = app_module.create_app(ls_svc.s, ls_svc).test_client()
    assert client.post("/webhook", json={"action": "TASKS_CREATED"}).get_json()["status"] == "ignored-action"
    assert client.post("/webhook", json={"action": "ANNOTATION_CREATED", "project": {"id": 99}}).get_json()["status"] == "ignored-project"
    with ls_svc.train_lock:  # a job is running
        assert client.post("/webhook", json={"action": "ANNOTATION_CREATED", "project": {"id": 7}}).get_json()["status"] == "running"
        assert client.post("/train").status_code == 409
    body = client.post("/train").get_json()
    assert body["status"] == "started"
    deadline = time.time() + 120
    while time.time() < deadline and ls_svc.train_lock.locked():
        time.sleep(0.2)
    assert ls_svc.telemetry["jobs"][-1]["job_id"] == body["job_id"]
