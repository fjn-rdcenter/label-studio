"""TASDF online-learning ML backend for Label Studio.

LS protocol: /health, /setup, /validate, /predict, /webhook, /train (see label_studio/ml/api_connector.py).
Ops (header X-OL-Token): /telemetry/live, /rollback, /refresh_ls_predictions.
Training is batch retraining on all reviewed annotations (trainer.py), run on N new ones or on /train.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
import threading
import time
import uuid

import numpy as np
import requests
import torch
from defusedxml import ElementTree
from flask import Flask, jsonify, request

from . import __version__, learner, trainer
from .config import Settings, resolve_device
from .dashboard import create_dashboard_blueprint
from .ls_format import CLASS_NAMES, NAME_TO_CLASS, fetch_image, mask_to_results
from .registry import Registry

log = logging.getLogger("tasdf_online")
TRAIN_ACTIONS = {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"}
SMALL_VAL_TOLERANCE = 0.01  # project-val too small to demand a gain: only refuse a clear drop


def parse_label_config(xml: str | None) -> tuple[str, str, str, dict]:
    """-> (from_name, to_name, image data key, {class: output label}) of the first PolygonLabels and its Image."""
    if not xml:
        return "seg", "image", "image", CLASS_NAMES
    root = ElementTree.fromstring(xml)
    poly = next((e for e in root.iter() if e.tag == "PolygonLabels"), None)
    if poly is None:
        raise ValueError("label config needs a PolygonLabels tag")
    to_name = poly.get("toName", "")
    image = next((e for e in root.iter() if e.tag == "Image" and e.get("name") == to_name), None)
    if image is None or not (image.get("value") or "").startswith("$"):
        raise ValueError(f"PolygonLabels toName={to_name!r} must point to an <Image value=\"$key\">")
    labels = {e.get("value") for e in poly.iter("Label")}
    # emit each class under the alias the project defines (e.g. "Blastocoel" instead of "BC")
    names = {cls: next((n for n, c in NAME_TO_CLASS.items() if c == cls and n in labels), name)
             for cls, name in CLASS_NAMES.items()}
    return poly.get("name"), to_name, image.get("value")[1:], names


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        holdout_changed = self.registry.sync_fingerprint({
            "baseline": _sha256(s.baseline_ckpt) if s.baseline_ckpt else "random-init",
            "holdout": _sha256(s.holdout_npz) if s.holdout_npz else None,
            "holdout_limit": s.holdout_limit,
        })
        self.teacher = self._load(self.registry.active)
        self.holdout = _load_npz(s.holdout_npz, s.holdout_limit)
        if s.anchor_npz and not os.path.exists(s.anchor_npz):
            # anchors are optional (compose always sets the path): training then uses project samples only
            log.warning("ANCHOR_NPZ=%s not found: training without anchors", s.anchor_npz)
            self.anchors = None
        else:
            self.anchors = _load_npz(s.anchor_npz)
            if self.anchors is not None:
                self.anchors = (self.anchors[0], self.anchors[1].astype(np.int64))
        self.label = parse_label_config(None)
        self.lock = threading.RLock()
        self.train_lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.dashboard_enabled: bool = s.dashboard_enabled
        self.telemetry: dict = {"status": "idle", "jobs": []}
        if self.holdout is not None and (holdout_changed or "metrics" not in self.registry.info("v0")):
            # the gate compares against v0 and the active version: both must be measured on this holdout
            self.registry.record("v0", metrics=learner.evaluate(self._load("v0"), *self.holdout))
            if self.registry.active != "v0":
                self.registry.record(self.registry.active, metrics=learner.evaluate(self.teacher, *self.holdout))

    # ---- model -------------------------------------------------------------------------------
    def _load(self, version: str):
        model = learner.build_model(self.s.base_channels, self.s.erosion)
        self.registry.load_into(model, self.base_state, version)
        return model.to(self.device).eval()

    @property
    def model_version(self) -> str:
        return f"tasdf-{self.registry.active}"

    def _image(self, task: dict, label: tuple[str, str, str] | None = None):
        key = (label or self.label)[2]
        return fetch_image(self.s.ls_url, self.s.ls_token, task["data"][key], self.s.request_timeout)

    def predict_task(self, task: dict, label: tuple[str, str, str] | None = None) -> dict:
        label = label or self.label
        img, w, h = self._image(task, label)
        with self.lock:
            probs, _ = learner.predict_probs(self.teacher, torch.from_numpy(img)[None, None])
            version = self.model_version
        conf, mask = probs[0].max(0)
        fg = mask.numpy() > 0
        score = float(conf[torch.from_numpy(fg)].mean()) if fg.any() else 0.0
        # polygons are traced at the original resolution: tracing the 256 map and scaling it up gives staircase edges
        full = torch.nn.functional.interpolate(probs[:1].float(), size=(h, w), mode="bilinear", align_corners=False)
        result = mask_to_results(full[0].argmax(0).numpy().astype(np.uint8), w, h, label[0], label[1], label[3])
        return {"result": result, "score": score, "model_version": version}

    # ---- training ------------------------------------------------------------------------------
    def on_webhook(self, payload: dict) -> str:
        """A webhook is only a hint: the job re-reads every annotation from LS before using it."""
        if payload.get("action") not in TRAIN_ACTIONS:
            return "ignored-action"
        if self.s.ls_project_id and (payload.get("project") or {}).get("id") != self.s.ls_project_id:
            return "ignored-project"
        return self.start_job(force=False)

    def start_job(self, force: bool) -> str:
        if not self.train_lock.acquire(blocking=False):
            return "running"
        job_id = uuid.uuid4().hex[:12]

        def run():
            try:
                self.run_job(force, job_id)
            finally:
                self.train_lock.release()

        threading.Thread(target=run, daemon=True, name=f"tasdf-train-{job_id}").start()
        return job_id

    def _samples(self) -> list[dict]:
        if not (self.s.ls_url and self.s.ls_token and self.s.ls_project_id):
            # fail closed: never train on anything that was not re-read from LS with our own token
            raise RuntimeError("LS_URL/LS_TOKEN/LS_PROJECT_ID must be set to train")
        return trainer.select_samples(list(self._project_tasks()), self.s.ls_project_id, self.label[0],
                                      self.s.min_quality_level)

    def _tensors(self, samples: list[dict]) -> tuple[torch.Tensor, list[np.ndarray]]:
        imgs = [self._image(d["task"])[0] for d in samples]
        x = torch.from_numpy(np.stack(imgs)).unsqueeze(1) if imgs else torch.zeros((0, 1, 256, 256))
        return x, [d["y"] for d in samples]

    def run_job(self, force: bool, job_id: str = "") -> dict | None:
        """Retrain from v0 on every reviewed train-split annotation; promote only if measured not worse."""
        t0 = time.time()
        try:
            self.telemetry["status"] = f"job {job_id}: collecting"
            samples = self._samples()
            train = [d for d in samples if d["split"] == "train"]
            val = [d for d in samples if d["split"] == "val"]
            last = (self.registry.data.get("last_job") or {}).get("digests", {})
            new = [d for d in train if last.get(str(d["task_id"])) != d["digest"]]
            self.telemetry["pending"] = {"new_train": len(new), "train": len(train), "val": len(val),
                                         "threshold": self.s.train_every_n}
            if not train or (not force and len(new) < self.s.train_every_n):
                return None
            self.telemetry["status"] = f"job {job_id}: training on {len(train)} (+{len(val)} val)"
            x, ys = self._tensors(train)
            vx, vys = self._tensors(val)
            y = torch.from_numpy(np.stack(ys).astype(np.int64))
            parent, active = self.registry.active, self.teacher
            base = self._load("v0")
            use_project = len(val) >= self.s.project_val_min

            def score(model):
                if use_project or self.holdout is None:
                    return trainer.project_dice(model, vx, vys) or 0.0
                return learner.evaluate(model, *self.holdout)["pooled_mDice"]

            cand, history = trainer.train(base, x, y, self.anchors, steps=self.s.train_steps, lr=self.s.lr,
                                          eval_every=self.s.eval_every, score_fn=score, amp=self.amp)
            shifts = learner.layer_shifts(base, cand, self.base_state)
            reasons = []
            if shifts["encoder_frozen"]["l2_shift"] or shifts["decoder_frozen"]["l2_shift"]:
                reasons.append("frozen weights changed")
            record = {"kind": "batch", "job_id": job_id, "train_tasks": [d["task_id"] for d in train],
                      "val_tasks": [d["task_id"] for d in val], "new_train": len(new), "history": history[-50:],
                      "layer_shifts": shifts}
            if self.holdout is not None:
                record["metrics"] = learner.evaluate(cand, *self.holdout)
                ok, gate_reasons = learner.gate(record["metrics"], learner.evaluate(active, *self.holdout),
                                                self.registry.info("v0")["metrics"])
                reasons += gate_reasons
            pv_cand, pv_active = trainer.project_dice(cand, vx, vys), trainer.project_dice(active, vx, vys)
            record.update(project_val_dice=pv_cand, active_project_val_dice=pv_active)
            if pv_cand is not None and pv_active is not None:
                if use_project and pv_cand < pv_active + self.s.promote_min_gain:
                    reasons.append(f"project-val {pv_cand:.4f} < active {pv_active:.4f} + {self.s.promote_min_gain}")
                elif not use_project and pv_cand < pv_active - SMALL_VAL_TOLERANCE:
                    reasons.append(f"project-val {pv_cand:.4f} dropped vs active {pv_active:.4f}")
            if not use_project:
                record["note"] = f"insufficient_project_val ({len(val)} < {self.s.project_val_min})"
            with self.lock:
                if self.registry.active != parent:  # a rollback happened meanwhile; candidate is stale
                    reasons.append(f"active changed {parent}->{self.registry.active} during job")
                accepted = not reasons
                record.update(gate_reasons=reasons, seconds=round(time.time() - t0, 1))
                version = self.registry.add(learner.trainable_state(cand), record, accepted)
                if accepted:
                    self.teacher = cand
            # completed (accepted or not): these samples no longer count as new
            self.registry.set_meta("last_job", {"digests": {str(d["task_id"]): d["digest"] for d in train},
                                                "version": version, "at": t0})
            record.update(version=version, accepted=accepted)
            self.telemetry["jobs"] = (self.telemetry["jobs"] + [record])[-20:]
            log.info("job %s -> %s accepted=%s reasons=%s (%.1fs)", job_id, version, accepted, reasons, time.time() - t0)
            if accepted:
                self.refresh_ls()
            return record
        except Exception as exc:  # keep the service alive; surface the error in telemetry
            log.exception("training job %s failed", job_id)
            self.telemetry["last_error"] = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            self.telemetry["status"] = "idle"
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

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
        """Precompute the next N unlabeled tasks for the active version, then PATCH project.model_version
        (LS validate_model_version requires a prediction of that version first). Old predictions are kept:
        an annotator may have one open (parent_prediction), and LS only shows the project's version.
        Needs a project-Manager token."""
        if not (self.s.ls_url and self.s.ls_token and self.s.ls_project_id):
            return {"skipped": "LS_URL/LS_TOKEN/LS_PROJECT_ID not set"}
        if not self.refresh_lock.acquire(blocking=False):
            return {"skipped": "a refresh is already running"}
        try:
            return self._refresh_ls()
        finally:
            self.refresh_lock.release()

    def _refresh_ls(self) -> dict:
        # On-demand (N=0) does not precompute static predictions into tasks
        precompute_n = max(0, self.s.refresh_next_n)
        name, stats = self.model_version, {"created": 0, "errors": 0}
        try:
            if precompute_n > 0:
                unlabeled, has_ver = [], False
                for t in self._project_tasks():
                    if any(p.get("model_version") == name for p in t.get("predictions", [])):
                        has_ver = True
                    if not any(not a.get("was_cancelled") for a in t.get("annotations", [])):
                        unlabeled.append(t)
                        if len(unlabeled) >= precompute_n:
                            break
                for t in sorted(unlabeled, key=lambda t: t["id"])[:precompute_n]:
                    if any(p.get("model_version") == name for p in t.get("predictions", [])):
                        continue
                    try:
                        self._ls("POST", "/api/predictions/", json={"task": t["id"], **self.predict_task(t)})
                        stats["created"] += 1
                    except Exception:
                        log.exception("prediction for task %s failed", t["id"])
                        stats["errors"] += 1
            else:
                has_ver = False
            if stats["created"] > 0 or has_ver or precompute_n == 0:
                try:
                    if precompute_n == 0:
                        # no prediction of the new version yet: LS accepts it only once the backend reports it, and
                        # listing the project's backends runs update_state (/setup) which stores ml_backend.model_version
                        self._ls("GET", "/api/ml/", params={"project": self.s.ls_project_id})
                    self._ls("PATCH", f"/api/projects/{self.s.ls_project_id}", json={"model_version": name})
                except requests.RequestException as exc:
                    log.warning("could not set project.model_version=%s: %s", name, exc)
                    stats["patch_failed"] = str(exc)
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
    # gunicorn configures only its own loggers: without this, INFO logs of tasdf_online are dropped in Docker
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("settings: %s", {k: v for k, v in dataclasses.asdict(settings).items() if k not in ("ls_token", "ol_token")})
    svc = service
    if svc is None:
        svc = OnlineService(settings)
        # sync LS with the active version after every restart (needs LS reachable)
        threading.Thread(target=svc.refresh_ls, daemon=True).start()
    app = Flask(__name__)
    app.config["svc"] = svc

    def ops_allowed() -> bool:
        # header only: a ?token= query would end up in the gunicorn access log
        if settings.ol_token:
            return request.headers.get("X-OL-Token") == settings.ol_token
        # no OL_TOKEN: open only while the dev dashboard is configured on; the product runs with DASHBOARD_ENABLED=0
        return settings.dashboard_enabled

    @app.get("/health")
    def health():
        return jsonify(status="UP", model_class="TASDFOnline", model_version=svc.model_version, device=str(svc.device))

    @app.post("/setup")
    def setup():
        payload = request.get_json(silent=True) or {}
        # LS sends project as "<id>.<created_ts>"; only the configured project may set the training label config
        project_id = str(payload.get("project", "")).split(".")[0]
        if payload.get("schema") and (not settings.ls_project_id or project_id == str(settings.ls_project_id)):
            svc.label = parse_label_config(payload["schema"])
        return jsonify(model_version=svc.model_version)

    @app.post("/validate")
    def validate():
        payload = request.get_json(silent=True) or {}
        parse_label_config(payload.get("config") or payload.get("label_config"))
        return jsonify(status="ok")

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        # per-request config: never overwrite the shared one (another project may call this backend)
        label = parse_label_config(payload["label_config"]) if payload.get("label_config") else None
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return jsonify(error="tasks must be a non-empty list"), 400
        results = []
        for t in tasks:
            try:
                results.append(svc.predict_task(t, label))
            except Exception:  # one bad task must not fail the whole batch
                log.exception("predict failed for task %s", t.get("id") if isinstance(t, dict) else t)
                results.append({"result": [], "score": 0.0, "model_version": svc.model_version})
        return jsonify(results=results, model_version=svc.model_version)

    @app.post("/webhook")
    def webhook():
        return jsonify(status=svc.on_webhook(request.get_json(silent=True) or {}))

    @app.post("/train")
    def train():
        # same trust level as /webhook (LS "Start Training" sends no X-OL-Token); data is re-read from LS anyway
        job = svc.start_job(force=True)
        if job == "running":
            return jsonify(status="running"), 409
        return jsonify(job_id=job, status="started")

    @app.get("/versions")
    def versions():
        return jsonify(versions=[f"tasdf-{v}" for v, i in svc.registry.data["versions"].items() if i.get("status") != "rejected"])

    @app.get("/telemetry/live")
    def telemetry():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        versions = {v: {k: i.get(k) for k in ("status", "parent", "metrics", "gate_reasons", "project_val_dice",
                                              "note", "created")}
                    for v, i in svc.registry.data["versions"].items()}
        return jsonify(active=svc.model_version, device=str(svc.device), training=svc.train_lock.locked(),
                       dashboard_enabled=svc.dashboard_enabled, version_history=versions, **svc.telemetry)

    @app.post("/rollback")
    def rollback():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        version = str((request.get_json(silent=True) or {}).get("version", "v0"))
        if not re.fullmatch(r"v\d+", version):
            return jsonify(error="version must look like v<N>"), 400
        try:
            svc.rollback(version)
        except FileNotFoundError:
            return jsonify(error=f"no stored weights for {version} (rejected or unknown)"), 404
        return jsonify(active=svc.model_version, refresh=svc.refresh_ls())

    @app.post("/refresh_ls_predictions")
    def refresh():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        return jsonify(svc.refresh_ls())

    if settings.dashboard_enabled:  # dev-only tool: with DASHBOARD_ENABLED=0 it is not mounted, so nothing can switch it on
        app.register_blueprint(create_dashboard_blueprint(svc, ops_allowed))

    @app.errorhandler(ValueError)
    def bad_input(err):
        return jsonify(error=str(err)), 400

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", "9090")), threaded=True)
