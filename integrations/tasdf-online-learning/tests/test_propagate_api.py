import dataclasses
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import tasdf_online.app as app_module
from tasdf_online.bundle_registry import BASE

ZP_RING = {"id": "zp1", "type": "polygonlabels", "from_name": "pl-0", "to_name": "timelapse_0",
           "value": {"points": [[30, 30], [70, 30], [70, 70], [30, 70]],
                     "holes": [[[40, 40], [60, 40], [60, 60], [40, 60]]], "ring": True, "polygonlabels": ["ZP"]}}
PN_RECT = {"id": "pn1", "type": "rectanglelabels", "from_name": "rect-0", "to_name": "timelapse_0",
           "value": {"x": 45, "y": 45, "width": 5, "height": 5, "rectanglelabels": ["PN"]}}
CONFIG = ('<View><Image name="timelapse_0" value="$image"/><PolygonLabels name="pl-0" toName="timelapse_0">'
          '<Label value="ZP"/><Label value="TE"/></PolygonLabels></View>')


@pytest.fixture
def client(settings, monkeypatch, tmp_path):
    s = dataclasses.replace(settings, journal_dir=str(tmp_path / "journal"))
    monkeypatch.setattr(app_module, "fetch_gray", lambda *a, **k: np.full((1288, 964), 128, np.uint8))
    svc = app_module.OnlineService(s)
    return app_module.create_app(s, svc).test_client(), svc, s


def call(c, url, ctx):
    task = {"id": 5, "data": {"image": url}}
    return c.post("/predict", json={"project": 7, "label_config": CONFIG, "tasks": [task],
                                    "context": {"action": "propagate", "threshold": 0.0, **ctx}})


def test_unknown_labels_copied_and_no_new_regions(client):
    c, _, _ = client
    body = call(c, "/data/b.jpg", {"carried": [ZP_RING, PN_RECT], "distance": 1}).get_json()
    res = body["results"][0]["result"]
    assert {r["id"] for r in res} <= {"zp1", "pn1"} and PN_RECT in res


def test_gold_scores_three_sources_after_a_proposal(client):
    c, _, _ = client
    first = call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1,
                                     "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}}).get_json()
    assert first["results"][0]["propagation"]["scores"] is None
    assert first["results"][0]["propagation"]["adapt"]["samples"] == 1
    second = call(c, "/data/c.jpg", {"carried": [ZP_RING], "distance": 1,
                                      "gold": {"url": "/data/b.jpg", "results": [ZP_RING]}}).get_json()
    assert set(second["results"][0]["propagation"]["scores"]) == {"copy", "production", "adapter"}


def test_empty_source_stops_without_learning(client):
    c, _, _ = client
    p = call(c, "/data/b.jpg", {"carried": [], "distance": 1, "gold": {"url": "/data/a.jpg", "results": []}}).get_json()
    prop = p["results"][0]["propagation"]
    assert p["results"][0]["result"] == [] and prop["stop"] == "no_objects" and prop["adapt"] is None


def test_model_seeing_nothing_stops_and_keeps_carried_copy(client, monkeypatch):
    c, _, _ = client
    bg = torch.zeros(1, 5, 256, 256)
    bg[:, 0] = 1
    monkeypatch.setattr(app_module.learner, "predict_probs", lambda m, x, batch=8: (bg.repeat(len(x), 1, 1, 1), None))
    body = call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1}).get_json()["results"][0]
    # rule 2 keeps the eroded core, so an empty prediction shrinks the object: it stops on area, not vanished
    assert body["propagation"]["stop"] == "area" and body["result"][0]["id"] == "zp1"


def test_restart_or_eviction_does_not_crash(client):
    c, svc, _ = client
    svc.wells.clear()
    r = call(c, "/data/c.jpg", {"carried": [ZP_RING], "distance": 2,
                                "gold": {"url": "/data/b.jpg", "results": [ZP_RING]}})
    assert r.status_code == 200 and r.get_json()["results"][0]["propagation"]["scores"] is None


def test_new_production_resets_wells(client):
    c, svc, _ = client
    call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1, "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}})
    assert len(svc.wells) == 1
    svc._serve(BASE)
    assert len(svc.wells) == 0


def test_bad_context_is_400(client):
    c, _, _ = client
    assert call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": "x"}).status_code == 400
    two = c.post("/predict", json={"project": 7, "label_config": CONFIG,
                                   "tasks": [{"id": 1, "data": {"image": "/data/a.jpg"}}] * 2,
                                   "context": {"action": "propagate", "carried": [], "distance": 1, "threshold": 0.8}})
    assert two.status_code == 400


def test_context_sent_under_params_like_label_studio_does(client):
    c, _, _ = client
    ctx = {"action": "propagate", "carried": [PN_RECT], "distance": 1, "threshold": 0.0}
    body = c.post("/predict", json={"project": 7, "label_config": CONFIG,
                                    "tasks": [{"id": 5, "data": {"image": "/data/b.jpg"}}],
                                    "params": {"login": None, "password": None, "context": ctx}}).get_json()
    assert body["results"][0]["result"] == [PN_RECT] and "propagation" in body["results"][0]


def test_reconfirming_an_image_neither_duplicates_its_sample_nor_its_score(client):
    c, svc, _ = client
    gold_b = {"url": "/data/b.jpg", "results": [ZP_RING]}
    call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1, "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}})
    call(c, "/data/c.jpg", {"carried": [ZP_RING], "distance": 1, "gold": gold_b})
    call(c, "/data/c.jpg", {"carried": [ZP_RING], "distance": 1, "gold": gold_b})
    state = svc.wells.get((7, 5), dict)
    assert len(state.samples) == 2 and len(state.scores) == 1


def test_well_memory_keeps_the_32_newest_samples(client, monkeypatch):
    c, svc, _ = client
    monkeypatch.setattr(app_module, "adapt", lambda *a, **k: {"samples": 0})
    for i in range(40):
        call(c, "/data/x.jpg", {"carried": [], "distance": 1, "gold": {"url": f"/data/{i}.jpg", "results": [ZP_RING]}})
    state = svc.wells.get((7, 5), dict)
    assert len(state.samples) == 32 and state.samples[-1].url == "/data/39.jpg"
    assert state.samples[0].y.dtype == torch.uint8


def test_copy_source_returns_the_carried_object_untouched(client):
    c, svc, _ = client
    svc.wells.get((7, 5), dict).scores = [{"copy": 1.0, "production": 0.0, "adapter": 0.0}] * 3
    body = call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1}).get_json()["results"][0]
    assert body["propagation"]["source"] == "copy" and body["result"] == [ZP_RING]


def test_refined_polygons_are_traced_at_the_crop_resolution(client, monkeypatch):
    c, _, _ = client
    shapes = []
    real = app_module.object_result
    monkeypatch.setattr(app_module, "object_result", lambda m, *a: shapes.append(m.shape) or real(m, *a))
    call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1})
    assert shapes == [(600, 600)]


def test_adapter_does_not_train_while_a_long_term_job_holds_the_gpu(client):
    c, svc, _ = client
    with svc.train_lock:
        body = call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1,
                                        "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}}).get_json()
    assert body["results"][0]["propagation"]["adapt"] == {"skipped": "training"}


def test_journal_written(client):
    c, _, s = client
    call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1, "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}})
    lines = (Path(s.journal_dir) / "7" / "5.jsonl").read_text().splitlines()
    assert any('"gold"' in line for line in lines) and any('"propose"' in line for line in lines)


def test_long_term_training_waits_while_adapter_runs():
    from tasdf_online import learner, trainer
    pause = threading.Event()
    pause.set()
    done = []
    m = learner.build_model(8)
    x = torch.rand(2, 1, 256, 256)
    y = torch.zeros(2, 256, 256, dtype=torch.int64)
    t = threading.Thread(target=lambda: done.append(
        trainer.train_epochs(m, x, y, None, mode="light", epochs=1, lr=1e-4, amp=False, pause=pause)))
    t.start()
    time.sleep(0.3)
    assert not done
    pause.clear()
    t.join(timeout=60)
    assert done
