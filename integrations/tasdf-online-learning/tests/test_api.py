import dataclasses
import hashlib

import numpy as np
import pytest
import torch
from conftest import blastocyst_mask

from tasdf_online import app as app_module
from tasdf_online import trainer
from tasdf_online.bundle_registry import BASE
from tasdf_online.ls_format import mask_to_results

AUTH = {"X-OL-Token": "secret"}
CFG = '<View><Image name="img" value="$picture"/><PolygonLabels name="p" toName="img"><Label value="ZP"/></PolygonLabels></View>'


@pytest.fixture
def client(settings, monkeypatch):
    img = np.random.default_rng(1).random((256, 256)).astype(np.float32)
    monkeypatch.setattr(app_module, "fetch_image", lambda *a, **k: (img, 800, 600))
    application = app_module.create_app(settings, app_module.OnlineService(settings))
    return application.test_client()


def test_health_setup_predict_per_stage(client):
    assert client.get("/health").get_json()["model_version"] == BASE
    assert client.post("/setup", json={"project": "7.1700000000", "schema": CFG}).get_json()["model_version"] == f"{BASE}.blastocyst"
    body = client.post("/predict", json={"project": "7.1700000000", "label_config": CFG,
                                         "tasks": [{"id": 1, "data": {"picture": "/data/x.png"}}]}).get_json()
    (res,) = body["results"]
    assert res["model_version"] == f"{BASE}.blastocyst" and isinstance(res["result"], list) and 0.0 <= res["score"] <= 1.0
    assert all(r["from_name"] == "p" and r["to_name"] == "img" for r in res["result"])
    assert client.post("/predict", json={"project": "7.1", "tasks": []}).status_code == 400


def test_unconfigured_project_is_refused(client):
    r = client.post("/setup", json={"project": "99.1700000000", "schema": CFG})
    assert r.status_code == 400 and "stages.json" in r.get_json()["error"]
    assert client.post("/predict", json={"project": "99.1", "tasks": [{"id": 1, "data": {"image": "/x"}}]}).status_code == 400


def test_dashboard_toggle_and_standalone(client):
    assert client.get("/dashboard").status_code == 200
    assert client.post("/dashboard/toggle", json={"enabled": False}).status_code == 401
    assert client.post("/dashboard/toggle", json={"enabled": False}, headers=AUTH).get_json()["dashboard_enabled"] is False
    assert client.get("/dashboard").status_code == 404
    assert client.post("/dashboard/toggle", json={"enabled": True}, headers=AUTH).get_json()["dashboard_enabled"] is True
    assert client.get("/dashboard").status_code == 200


def test_webhook_only_counts(client):
    assert client.post("/webhook", json={"action": "TASKS_CREATED"}).get_json()["status"] == "ignored-action"
    assert client.post("/webhook", json={"action": "ANNOTATION_CREATED"}).get_json()["status"] == "ignored-project"
    assert client.post("/webhook", json={"action": "ANNOTATION_CREATED", "project": {"id": 99}}).get_json()["status"] == "ignored-project"
    assert client.post("/webhook", json={"action": "ANNOTATION_UPDATED", "project": {"id": 7}}).get_json()["status"] == "counted"
    tel = client.get("/telemetry/live", headers=AUTH).get_json()
    assert tel["webhook_events"] == 1 and not tel["training"] and tel["production"] == BASE


def test_predict_survives_one_bad_task(client, monkeypatch):
    good = np.zeros((256, 256), np.float32)

    def fake_fetch(ls_url, token, source, timeout=30):
        if "bad" in source:
            raise ValueError("broken image")
        return good, 256, 256

    monkeypatch.setattr(app_module, "fetch_image", fake_fetch)
    tasks = [{"id": 1, "data": {"image": "/data/ok.png"}}, {"id": 2, "data": {"image": "/data/bad.png"}}]
    body = client.post("/predict", json={"project": 7, "tasks": tasks}).get_json()
    assert len(body["results"]) == 2 and body["results"][1]["result"] == []


def test_ops_token_not_accepted_in_query(client):
    assert client.get("/telemetry/live?token=secret").status_code == 401
    assert client.get("/telemetry/live", headers=AUTH).status_code == 200


def test_rollback_requires_token(client):
    assert client.post("/rollback").status_code == 401
    body = client.post("/rollback", headers=AUTH).get_json()
    assert body["production"] == BASE and "skipped" in body["refresh"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predict_on_cuda(settings, monkeypatch):
    img = np.zeros((256, 256), np.float32)
    monkeypatch.setattr(app_module, "fetch_image", lambda *a, **k: (img, 256, 256))
    s = dataclasses.replace(settings, device="cuda", amp="auto")
    client = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert client.get("/health").get_json()["device"].startswith("cuda")
    assert client.post("/predict", json={"project": 7, "tasks": [{"id": 1, "data": {"image": "/data/x.png"}}]}).status_code == 200


def test_ops_open_without_ol_token(settings):
    s = dataclasses.replace(settings, ol_token="")
    c = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert c.get("/telemetry/live").status_code == 200


def test_ops_closed_without_ol_token_when_dashboard_off(settings):
    # product: DASHBOARD_ENABLED=0 and no OL_TOKEN -> nobody may roll back or refresh
    s = dataclasses.replace(settings, ol_token="", dashboard_enabled=False)
    c = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert c.post("/rollback").status_code == 401
    assert c.post("/refresh_ls_predictions").status_code == 401
    assert c.get("/telemetry/live").status_code == 401


def test_dashboard_not_mounted_when_disabled(settings):
    s = dataclasses.replace(settings, dashboard_enabled=False)
    c = app_module.create_app(s, app_module.OnlineService(s)).test_client()
    assert c.get("/dashboard").status_code == 404
    assert c.post("/dashboard/toggle", json={"enabled": True}, headers=AUTH).status_code == 404


def test_missing_anchor_file_is_optional(settings, tmp_path):
    svc = app_module.OnlineService(dataclasses.replace(settings, anchor_npz=str(tmp_path / "missing.npz")))
    assert svc.anchors["blastocyst"] is None


def test_startup_logs_settings_without_secrets(settings, caplog):
    s = dataclasses.replace(settings, ls_token="ls-secret-value", ol_token="ol-secret-value")
    with caplog.at_level("INFO", logger="tasdf_online"):
        app_module.create_app(s, app_module.OnlineService(s))
    assert "settings:" in caplog.text and "train_epochs" in caplog.text
    assert "ls-secret-value" not in caplog.text and "ol-secret-value" not in caplog.text


def test_baseline_metrics_are_measured_once_per_holdout(settings):
    svc = app_module.OnlineService(settings)
    m = svc.registry.info(BASE)["metrics"]["blastocyst"]
    assert "pooled_mDice" in m["d10"] and m["holdout_sha"]


def test_output_labels_follow_the_project_aliases():
    alias = ('<View><Image name="timelapse_3" value="$image"/><PolygonLabels name="pl-3" toName="timelapse_3">'
             '<Label value="ZP"/><Label value="TE"/><Label value="ICM"/><Label value="Blastocoel"/></PolygonLabels></View>')
    label = app_module.parse_label_config(alias)
    assert label[:3] == ("pl-3", "timelapse_3", "image")
    names = {r["value"]["polygonlabels"][0] for r in mask_to_results(blastocyst_mask(), 256, 256, *label[:2], label[3])}
    assert "Blastocoel" in names and "BC" not in names
    assert app_module.parse_label_config(alias.replace("Blastocoel", "BC"))[3] == app_module.CLASS_NAMES


# ---- long-term training with a fake Label Studio -----------------------------------------------
def _img(source: str) -> np.ndarray:
    seed = int(hashlib.sha1(source.encode()).hexdigest(), 16) % 2 ** 32
    return np.random.default_rng(seed).random((256, 256)).astype(np.float32)


def _ls_tasks(ids, quality=2):
    result = mask_to_results(blastocyst_mask(), 256, 256, "seg", "image")
    return [{"id": t, "data": {"image": f"/data/{t}.png"}, "predictions": [],
             "annotations": [{"id": 100 + t, "result": result, "quality_level": quality, "was_cancelled": False,
                              "updated_at": "2026-11-12T10:00", "completed_by": 1}]} for t in ids]


@pytest.fixture
def ls_svc(settings, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_image", lambda url, tok, src, timeout=30: (_img(src), 256, 256))
    monkeypatch.setattr(app_module, "_on_battery", lambda: False)  # host power state must not decide tests
    s = dataclasses.replace(settings, ls_url="http://ls.test", ls_token="tok")
    tasks = {"list": _ls_tasks(range(1, 13))}
    calls = []

    def fake_get(url, **kw):
        page = kw.get("params", {}).get("page")

        class R:
            status_code = 200 if page == 1 else 404
            def raise_for_status(self): pass
            def json(self): return tasks["list"] if page == 1 else []
        return R()

    def fake_request(method, url, **kw):
        calls.append((method, url))

        class R:
            def raise_for_status(self): pass
            def json(self): return {"label_config": None}  # default label: from_name "seg", key "image"
        return R()

    monkeypatch.setattr(app_module.requests, "get", fake_get)
    monkeypatch.setattr(app_module.requests, "request", fake_request)
    svc = app_module.OnlineService(s)
    svc.fake_tasks, svc.calls = tasks, calls
    return svc


def test_job_saves_a_bundle_and_gates_it(ls_svc):
    rec = ls_svc.run_job(force=True, job_id="t1")
    assert "last_error" not in ls_svc.telemetry, ls_svc.telemetry.get("last_error")
    n_val = sum(trainer.image_split(_img(f"/data/{t}.png")) == "val" for t in range(1, 13))
    assert rec["train_images"] == 12 - n_val and rec["val_images"] == n_val
    info = ls_svc.registry.info(rec["bundle"])
    assert info["status"] in ("production", "rejected") and info["metrics"]["blastocyst"]["d10"]
    assert (ls_svc.registry.production == rec["bundle"]) == rec["accepted"]
    assert info["layer_shifts"]["encoder_frozen"]["l2_shift"] == 0.0
    # snapshot persists: nothing new -> no job unless forced, also after a restart
    assert ls_svc.run_job(force=False) is None and ls_svc.telemetry["pending"]["new"] == 0
    again = app_module.OnlineService(ls_svc.s)
    assert again.registry.data["last_data_snapshot"]["7"].keys() == {str(t) for t in range(1, 13)}


def test_promoted_bundle_is_served_and_synced(ls_svc, monkeypatch):
    monkeypatch.setattr(trainer, "promotion_decision", lambda **kw: (True, [], None))
    rec = ls_svc.run_job(force=True, job_id="t2")
    assert rec["accepted"] and ls_svc.registry.production == rec["bundle"] == ls_svc.model_version
    assert ls_svc.registry.previous == BASE and ls_svc.registry.data["history"][-1]["by"] == "gate"
    assert ("PATCH", "http://ls.test/api/projects/7") in ls_svc.calls  # LS points the project at the new version
    assert ls_svc.version_for("blastocyst") == f"{rec['bundle']}.blastocyst"
    assert ls_svc.rollback() == BASE and ls_svc.model_version == BASE


def test_job_waits_for_enough_new_labels(ls_svc):
    ls_svc.s = dataclasses.replace(ls_svc.s, train_min_new=50)
    assert ls_svc.run_job(force=False) is None
    assert ls_svc.telemetry["pending"]["new"] == 12 and ls_svc.registry.production == BASE


def test_job_is_postponed_when_memory_is_short(ls_svc):
    ls_svc.s = dataclasses.replace(ls_svc.s, min_free_ram_gb=10 ** 6)
    assert ls_svc.run_job(force=True) is None
    assert "free RAM" in ls_svc.telemetry["last_skip"]["reason"]


def test_job_is_postponed_on_battery(ls_svc, monkeypatch):
    monkeypatch.setattr(app_module, "_on_battery", lambda: True)
    assert ls_svc.run_job(force=True) is None
    assert "battery" in ls_svc.telemetry["last_skip"]["reason"]


def test_dry_run_measures_without_saving(ls_svc):
    rec = ls_svc.run_job(force=True, job_id="d1", dry_run=True)
    assert rec["dry_run"] and rec["ms_per_image"] > 0
    assert set(ls_svc.registry.data["bundles"]) == {BASE}


def test_job_needs_ls_credentials(settings):
    svc = app_module.OnlineService(dataclasses.replace(settings, ls_url="http://ls.test", ls_token=""))
    assert svc.run_job(force=True) is None and "LS_TOKEN" in svc.telemetry["last_error"]


def test_train_route_conflicts_while_running(ls_svc):
    client = app_module.create_app(ls_svc.s, ls_svc).test_client()
    with ls_svc.train_lock:
        assert client.post("/train").status_code == 409


def test_registry_and_promote_endpoints(ls_svc, monkeypatch):
    client = app_module.create_app(ls_svc.s, ls_svc).test_client()
    assert client.get("/registry").status_code == 401
    rec = ls_svc.run_job(force=True, job_id="p1")
    reg = client.get("/registry", headers=AUTH).get_json()
    assert rec["bundle"] in reg["bundles"] and reg["pointers"]["production"] in reg["bundles"]
    assert reg["stages"]["blastocyst"]["projects"] == [7] and reg["data_snapshot"]["7"] == 12
    assert reg["schedule"]["cron"] == "0 1 * * 0" and reg["schedule"]["next_run"]
    assert client.post("/promote", json={"bundle": "nope"}, headers=AUTH).status_code == 400
    ls_svc.rollback()  # production back to the baseline, then promote the job's bundle by hand
    monkeypatch.setattr(trainer, "promotion_decision", lambda **kw: (False, ["val gain too small"], None))
    body = client.post("/promote", json={"bundle": rec["bundle"]}, headers=AUTH).get_json()
    assert body == {"promoted": False, "reasons": ["val gain too small"]}
    body = client.post("/promote", json={"bundle": rec["bundle"], "override": True, "reason": "doctor review"},
                       headers=AUTH).get_json()
    assert body["promoted"] and ls_svc.registry.production == rec["bundle"] == ls_svc.model_version
    assert ls_svc.registry.data["history"][-1]["reason"] == "doctor review"
    with ls_svc.train_lock:
        assert client.post("/promote", json={"bundle": BASE}, headers=AUTH).status_code == 409
