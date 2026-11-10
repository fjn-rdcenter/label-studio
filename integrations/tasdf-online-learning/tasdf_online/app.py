"""TASDF online-learning ML backend for Label Studio.

LS protocol: /health, /setup, /validate, /predict, /webhook (see label_studio/ml/api_connector.py).
Ops (header X-OL-Token): /telemetry/live, /rollback, /refresh_ls_predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import queue
import random
import threading
import time
from collections import deque

import numpy as np
import requests
import torch
from defusedxml import ElementTree
from flask import Flask, jsonify, request

from . import __version__, learner
from .config import Settings, resolve_device
from .dashboard import create_dashboard_blueprint, render_visuals
from .ls_format import bio_valid, fetch_image, mask_to_results, results_to_mask
from .registry import Registry

log = logging.getLogger("tasdf_online")
TRAIN_ACTIONS = {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"}


def parse_label_config(xml: str | None) -> tuple[str, str, str]:
    """-> (from_name, to_name, image data key) of the first PolygonLabels and its Image."""
    if not xml:
        return "seg", "image", "image"
    root = ElementTree.fromstring(xml)
    poly = next((e for e in root.iter() if e.tag == "PolygonLabels"), None)
    if poly is None:
        raise ValueError("label config needs a PolygonLabels tag")
    to_name = poly.get("toName", "")
    image = next((e for e in root.iter() if e.tag == "Image" and e.get("name") == to_name), None)
    if image is None or not (image.get("value") or "").startswith("$"):
        raise ValueError(f"PolygonLabels toName={to_name!r} must point to an <Image value=\"$key\">")
    return poly.get("name"), to_name, image.get("value")[1:]


def _load_npz(path: str, limit: int = 0):
    if not path:
        return None
    data = np.load(path)
    images, masks = data["images"], data["masks"]
    if limit:
        images, masks = images[:limit], masks[:limit]
    return torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1), masks.astype(np.int64)


class OnlineService:
    def __init__(self, s: Settings):
        self.s = s
        self.device = resolve_device(s.device)
        if s.torch_threads:
            torch.set_num_threads(s.torch_threads)
        self.amp = self.device.type == "cuda" if s.amp == "auto" else s.amp.lower() in {"1", "true", "on"}
        if s.baseline_ckpt:
            self.base_state = learner.load_state_dict(s.baseline_ckpt)
        else:
            log.warning("BASELINE_CKPT not set: using a randomly initialised model (tests/smoke only)")
            self.base_state = learner.build_model(s.base_channels, s.erosion).state_dict()
        self.registry = Registry(s.state_dir, s.ls_project_id)
        self.teacher = self._load(self.registry.active)
        self.holdout = _load_npz(s.holdout_npz, s.holdout_limit)
        self.anchors = _load_npz(s.anchor_npz)
        self.replay: deque = deque(maxlen=64)
        self.label = parse_label_config(None)
        self.lock = threading.RLock()
        self.qlock = threading.Lock()
        self.queue: queue.Queue = queue.Queue()
        self.pending: dict = {}
        self.seen: dict = {}
        self.last_predicted_mask: dict[int, np.ndarray] = {}
        self.dashboard_enabled: bool = s.dashboard_enabled
        self.telemetry: dict = {"status": "idle", "episodes": [], "next_task_dice": []}
        if self.holdout is not None and "metrics" not in self.registry.info("v0"):
            self.registry.record("v0", metrics=learner.evaluate(self._load("v0"), *self.holdout))
        threading.Thread(target=self._worker, daemon=True, name="tasdf-online-worker").start()

    # ---- model -------------------------------------------------------------------------------
    def _load(self, version: str):
        model = learner.build_model(self.s.base_channels, self.s.erosion)
        self.registry.load_into(model, self.base_state, version)
        return model.to(self.device).eval()

    @property
    def model_version(self) -> str:
        return f"tasdf-{self.registry.active}"

    def _image(self, task: dict):
        return fetch_image(self.s.ls_url, self.s.ls_token, task["data"][self.label[2]], self.s.request_timeout)

    def predict_task(self, task: dict) -> dict:
        img, w, h = self._image(task)
        with self.lock:
            probs, _ = learner.predict_probs(self.teacher, torch.from_numpy(img)[None, None])
            version = self.model_version
        conf, mask = probs[0].max(0)
        mask = mask.numpy().astype(np.uint8)
        if task.get("id") is not None:
            self.last_predicted_mask[int(task["id"])] = mask.copy()
        fg = mask > 0
        score = float(conf[torch.from_numpy(fg)].mean()) if fg.any() else 0.0
        result = mask_to_results(mask, w, h, self.label[0], self.label[1])
        return {"result": result, "score": score, "model_version": version}

    # ---- webhook / training ------------------------------------------------------------------
    def enqueue(self, payload: dict) -> str:
        if payload.get("action") not in TRAIN_ACTIONS:
            return "ignored-action"
        ann, task = payload.get("annotation") or {}, payload.get("task") or {}
        if self.s.ls_project_id and (payload.get("project") or {}).get("id", self.s.ls_project_id) != self.s.ls_project_id:
            return "ignored-project"
        if ann.get("was_cancelled") or not ann.get("result"):
            return "ignored-empty"
        if int(ann.get("quality_level") or 1) < self.s.min_quality_level:
            return "ignored-quality"
        if not bio_valid(results_to_mask(ann["result"], self.label[0])):
            return "ignored-invalid-mask"
        geom = [{k: v for k, v in r.items() if k != "id"} for r in ann["result"] if isinstance(r, dict)]
        digest = hashlib.sha1(json.dumps(geom, sort_keys=True).encode()).hexdigest()
        if self.seen.get(ann.get("id")) == digest:
            return "ignored-unchanged"
        with self.qlock:
            first = ann.get("id") not in self.pending
            self.pending[ann.get("id")] = (ann, task, digest)  # coalesce: keep the latest version only
            if first:
                self.queue.put(ann.get("id"))
        return "queued"

    def _worker(self):
        while True:
            ann_id = self.queue.get()
            with self.qlock:
                ann, task, digest = self.pending.pop(ann_id)
            self.telemetry["status"] = f"training annotation {ann_id}"
            try:
                self._episode(ann, task)
                self.seen[ann_id] = digest
            except Exception as exc:  # keep the worker alive; surface the error in telemetry
                log.exception("episode failed for annotation %s", ann_id)
                self.telemetry["last_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self.telemetry["status"] = "idle"
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _initial_prediction(self, ann: dict, task_id: int | None = None):
        pid = ann.get("parent_prediction")
        if pid and self.s.ls_url:
            try:
                resp = requests.get(f"{self.s.ls_url}/api/predictions/{pid}", headers=self._auth(), timeout=self.s.request_timeout)
                resp.raise_for_status()
                return results_to_mask(resp.json().get("result") or [], self.label[0])
            except requests.RequestException as exc:
                log.warning("could not fetch parent_prediction %s: %s", pid, exc)
        if task_id is not None and int(task_id) in self.last_predicted_mask:
            return self.last_predicted_mask[int(task_id)]
        return None

    def _episode(self, ann: dict, task: dict):
        t0 = time.time()
        img, _, _ = self._image(task)
        y = results_to_mask(ann["result"], self.label[0])
        p_init = self._initial_prediction(ann, task.get("id"))  # what the annotator actually saw, not today's model
        record = {"annotation_id": ann.get("id"), "task_id": task.get("id"), "at": t0}
        if p_init is not None:
            record["next_task_dice"] = learner.mean_dice(p_init, y)[0]
            self.telemetry["next_task_dice"].append({"task_id": task.get("id"), "dice": record["next_task_dice"]})

        teacher, parent = self.teacher, self.registry.active
        x = torch.from_numpy(img)[None, None]
        before, _ = learner.predict_probs(teacher, x)
        ref_mask = p_init if p_init is not None else before[0].argmax(0).numpy()
        w_new = 1.0 + self.s.corr_beta * torch.from_numpy((ref_mask != y).astype(np.float32))
        xs, ys, ws = [x], [torch.from_numpy(y.astype(np.int64))[None]], [w_new[None]]
        replay = random.sample(list(self.replay), min(self.s.replay_size, len(self.replay)))
        if self.anchors is not None and len(replay) < self.s.replay_size:
            idx = random.sample(range(len(self.anchors[0])), min(self.s.replay_size - len(replay), len(self.anchors[0])))
            replay += [(self.anchors[0][i:i + 1], torch.from_numpy(self.anchors[1][i:i + 1])) for i in idx]
        for rx, ry in replay:
            xs.append(rx); ys.append(ry); ws.append(torch.ones(ry.shape))

        cand, losses = learner.run_episode(
            teacher, torch.cat(xs), torch.cat(ys), torch.cat(ws), epochs=self.s.episode_epochs, lr=self.s.lr,
            ema_alpha=self.s.ema_alpha, topo_sup_weight=self.s.topo_sup_weight, erosion=self.s.erosion, amp=self.amp)
        after, topo_after = learner.predict_probs(cand, x)
        shifts = learner.layer_shifts(teacher, cand, self.base_state)
        frozen_ok = shifts["encoder_frozen"]["l2_shift"] == 0.0 and shifts["decoder_frozen"]["l2_shift"] == 0.0
        record.update(losses=losses, local_probe=learner.local_probe(before[0], after[0], y), layer_shifts=shifts,
                      frozen_unchanged=frozen_ok)
        if self.dashboard_enabled:
            self.telemetry["last_visuals"] = render_visuals(img, ref_mask, before[0], after[0], topo_after[0], y)

        accepted, reasons = False, ["no holdout configured"]
        if self.holdout is not None and frozen_ok:
            record["metrics"] = learner.evaluate(cand, *self.holdout)
            accepted, reasons = learner.gate(record["metrics"], self.registry.info(parent)["metrics"],
                                             self.registry.info("v0")["metrics"])
        elif not frozen_ok:
            reasons = ["frozen weights changed"]
        with self.lock:
            if self.registry.active != parent:  # a rollback happened meanwhile; candidate is stale
                accepted, reasons = False, [f"active changed {parent}->{self.registry.active} during episode"]
            record.update(gate_reasons=reasons, seconds=round(time.time() - t0, 2))
            version = self.registry.add(learner.trainable_state(cand), record, accepted)
            if accepted:
                self.teacher = cand
        if accepted:
            self.replay.append((x, ys[0]))
        record["version"], record["accepted"] = version, accepted
        self.telemetry["episodes"] = (self.telemetry["episodes"] + [record])[-20:]
        log.info("episode %s accepted=%s reasons=%s (%.1fs)", version, accepted, reasons, time.time() - t0)
        if accepted:
            self.refresh_ls()

    # ---- Label Studio sync ------------------------------------------------------------------
    def _auth(self) -> dict:
        return {"Authorization": f"Token {self.s.ls_token}"}

    def _ls(self, method: str, path: str, **kw):
        resp = requests.request(method, f"{self.s.ls_url}{path}", headers=self._auth(), timeout=self.s.request_timeout, **kw)
        resp.raise_for_status()
        return resp

    def _project_tasks(self):
        page = 1
        while True:
            resp = requests.get(f"{self.s.ls_url}/api/projects/{self.s.ls_project_id}/tasks/",
                                params={"page": page, "page_size": 100}, headers=self._auth(), timeout=self.s.request_timeout)
            if resp.status_code == 404 or not resp.json():
                return  # LS paginator raises 404 past the last page
            resp.raise_for_status()
            yield from resp.json()
            page += 1

    def refresh_ls(self) -> dict:
        """Drop stale predictions on unlabeled tasks, precompute the next N for the active version,
        then PATCH project.model_version (LS validate_model_version requires predictions with that
        version to exist first). Needs a project-Manager token."""
        if not (self.s.ls_url and self.s.ls_token and self.s.ls_project_id):
            return {"skipped": "LS_URL/LS_TOKEN/LS_PROJECT_ID not set"}
        if self.s.refresh_next_n <= 0:
            self.telemetry["last_refresh"] = {"version": self.model_version, "skipped": "on-demand mode (REFRESH_NEXT_N=0)"}
            return self.telemetry["last_refresh"]
        name, stats = self.model_version, {"deleted": 0, "created": 0, "errors": 0}
        try:
            tasks = list(self._project_tasks())
            unlabeled = [t for t in tasks if not any(not a.get("was_cancelled") for a in t.get("annotations", []))]
            for t in unlabeled:
                for p in t.get("predictions", []):
                    if p.get("model_version") != name:
                        self._ls("DELETE", f"/api/predictions/{p['id']}")
                        stats["deleted"] += 1
            for t in sorted(unlabeled, key=lambda t: t["id"])[: self.s.refresh_next_n]:
                if any(p.get("model_version") == name for p in t.get("predictions", [])):
                    continue
                try:
                    self._ls("POST", "/api/predictions/", json={"task": t["id"], **self.predict_task(t)})
                    stats["created"] += 1
                except Exception:
                    log.exception("prediction for task %s failed", t["id"])
                    stats["errors"] += 1
            has_ver = stats["created"] > 0 or any(
                p.get("model_version") == name for t in tasks for p in t.get("predictions", [])
            )
            if has_ver:
                self._ls("PATCH", f"/api/projects/{self.s.ls_project_id}", json={"model_version": name})
        except requests.RequestException as exc:
            log.warning("Label Studio sync failed: %s", exc)
            stats["failed"] = str(exc)
        self.telemetry["last_refresh"] = {"version": name, **stats}
        return stats

    def rollback(self, version: str) -> None:
        with self.lock:
            self.registry.rollback(version)
            self.teacher = self._load(version)


def create_app(settings: Settings | None = None, service: OnlineService | None = None) -> Flask:
    settings = settings or Settings.from_env()
    svc = service
    if svc is None:
        svc = OnlineService(settings)
        if settings.refresh_next_n > 0:
            threading.Thread(target=svc.refresh_ls, daemon=True).start()
    app = Flask(__name__)
    app.config["svc"] = svc

    def ops_allowed() -> bool:
        tok = request.headers.get("X-OL-Token") or request.args.get("token")
        return bool(settings.ol_token) and tok == settings.ol_token

    @app.get("/health")
    def health():
        return jsonify(status="UP", model_class="TASDFOnline", model_version=svc.model_version, device=str(svc.device))

    @app.post("/setup")
    def setup():
        schema = (request.get_json(silent=True) or {}).get("schema")
        if schema:
            svc.label = parse_label_config(schema)
        return jsonify(model_version=svc.model_version)

    @app.post("/validate")
    def validate():
        payload = request.get_json(silent=True) or {}
        parse_label_config(payload.get("config") or payload.get("label_config"))
        return jsonify(status="ok")

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        if payload.get("label_config"):
            svc.label = parse_label_config(payload["label_config"])
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return jsonify(error="tasks must be a non-empty list"), 400
        return jsonify(results=[svc.predict_task(t) for t in tasks], model_version=svc.model_version)

    @app.post("/webhook")
    def webhook():
        return jsonify(status=svc.enqueue(request.get_json(silent=True) or {}))

    @app.get("/versions")
    def versions():
        return jsonify(versions=[f"tasdf-{v}" for v, i in svc.registry.data["versions"].items() if i.get("status") != "rejected"])

    @app.get("/telemetry/live")
    def telemetry():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        versions = {v: {k: i.get(k) for k in ("status", "parent", "metrics", "gate_reasons", "next_task_dice", "created")}
                    for v, i in svc.registry.data["versions"].items()}
        return jsonify(active=svc.model_version, device=str(svc.device), queue=svc.queue.qsize(),
                       dashboard_enabled=svc.dashboard_enabled, version_history=versions, **svc.telemetry)

    @app.post("/rollback")
    def rollback():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        version = (request.get_json(silent=True) or {}).get("version", "v0")
        svc.rollback(version)
        return jsonify(active=svc.model_version, refresh=svc.refresh_ls())

    @app.post("/refresh_ls_predictions")
    def refresh():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        return jsonify(svc.refresh_ls())

    app.register_blueprint(create_dashboard_blueprint(svc, ops_allowed))

    @app.errorhandler(ValueError)
    def bad_input(err):
        return jsonify(error=str(err)), 400

    return app


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", "9090")), threaded=True)
