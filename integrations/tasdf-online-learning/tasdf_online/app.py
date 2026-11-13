"""TASDF ML backend for Label Studio: stage adapters on a shared backbone, long-term retraining.

LS protocol: /health, /setup, /validate, /predict, /webhook, /train (see label_studio/ml/api_connector.py).
Ops (header X-OL-Token): /telemetry/live, /rollback, /refresh_ls_predictions.
Models live in one bundle registry (bundle_registry.py); each LS project maps to an embryo stage (stages.json)
and is served `<production bundle>.<stage>`. Only a long-term job (trainer.py) can change production.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import gc
import hashlib
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import torch
from defusedxml import ElementTree
from flask import Flask, jsonify, request

from . import learner, trainer
from .bundle_registry import BASE, BundleRegistry, adapter_state
from .config import Settings, load_stages, resolve_device, stage_of
from .dashboard import create_dashboard_blueprint
from .ls_format import CLASS_NAMES, NAME_TO_CLASS, fetch_image, mask_to_results
from .schedule import Cron

log = logging.getLogger("tasdf_online")
TRAIN_ACTIONS = {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"}


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
    if not path or not os.path.exists(path):
        return None
    data = np.load(path)
    images, masks = data["images"], data["masks"]
    if limit:
        images, masks = images[:limit], masks[:limit]
    return torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1), masks.astype(np.int64)


def _free_ram_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    return float("inf")


def _on_battery() -> bool:
    """True only when a battery reports Discharging (a desktop or a host without sysfs counts as mains).
    USB-C source entries also say "Discharging" while on AC, so only `type == Battery` counts."""
    for supply in Path("/sys/class/power_supply").glob("*"):
        try:
            if (supply / "type").read_text().strip() == "Battery" and \
                    (supply / "status").read_text().strip() == "Discharging":
                return True
        except OSError:
            pass
    return False


class OnlineService:
    def __init__(self, s: Settings):
        self.s = s
        self.device = resolve_device(s.device)
        if s.torch_threads:
            torch.set_num_threads(s.torch_threads)
        self.amp = self.device.type == "cuda" if s.amp == "auto" else s.amp.lower() in {"1", "true", "on"}
        self.stages = load_stages(s.stages_file)
        if s.baseline_ckpt:
            self.base_state = learner.load_state_dict(s.baseline_ckpt)
            base_sha = _sha256(s.baseline_ckpt)
        else:
            log.warning("BASELINE_CKPT not set: using a randomly initialised model (tests/smoke only)")
            self.base_state = learner.build_model(s.base_channels, s.erosion).state_dict()
            base_sha = "random-init"
        gc.collect()  # the checkpoint's optimizer state is garbage now: give the RAM back
        self.registry = BundleRegistry(s.registry_dir, self.base_state, base_sha, list(self.stages))
        moved = self.registry.migrate_legacy(s.state_dir)
        if moved:
            log.info("moved legacy per-project registries %s to %s/legacy (not served)", moved, s.registry_dir)
        self.holdouts = {st: _load_npz(cfg.get("holdout_npz") or s.holdout_npz, s.holdout_limit)
                         for st, cfg in self.stages.items()}
        self.anchors = {st: _load_npz(cfg.get("anchors_npz") or s.anchor_npz) for st, cfg in self.stages.items()}
        self.lock = threading.RLock()
        self.train_lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.dashboard_enabled: bool = s.dashboard_enabled
        self.telemetry: dict = {"status": "idle", "jobs": [], "webhook_events": 0}
        self.cron = Cron(s.train_cron) if s.train_cron.strip() else None  # bad expression: fail at startup
        self.telemetry["schedule"] = {"cron": s.train_cron or None, "next_run": self.next_run()}
        self._serve(self.registry.production)
        self._baseline_metrics()

    # ---- serving -----------------------------------------------------------------------------
    def model_for(self, bundle: str, stage: str):
        model = learner.build_model(self.s.base_channels, self.s.erosion)
        model.load_state_dict(self.registry.state_for(bundle, stage))
        return model.to(self.device).eval()

    def _serve(self, bundle: str) -> None:
        """Load `bundle` once; switching stage later only swaps its small adapter (BN + heads)."""
        stages = list(self.stages)
        teacher = self.model_for(bundle, stages[0])
        adapters = {st: {k: v.to(self.device) for k, v in adapter_state(self.registry.state_for(bundle, st)).items()}
                    for st in stages}
        with self.lock:
            self.teacher, self.adapters, self.serving_bundle, self.serving_stage = teacher, adapters, bundle, stages[0]
        gc.collect()

    def _use_stage(self, stage: str) -> None:
        if stage != self.serving_stage:
            self.teacher.load_state_dict(self.adapters[stage], strict=False)
            self.serving_stage = stage

    def _baseline_metrics(self) -> None:
        """D10 metrics of the baseline per stage, measured once per holdout file (the forgetting gate's anchor)."""
        metrics = self.registry.info(BASE).get("metrics") or {}
        changed = False
        for st, hold in self.holdouts.items():
            sha = _sha256(self.stages[st].get("holdout_npz") or self.s.holdout_npz) if hold is not None else None
            if hold is not None and (metrics.get(st) or {}).get("holdout_sha") != sha:
                metrics[st] = {"d10": learner.evaluate(self.model_for(BASE, st), *hold), "holdout_sha": sha}
                changed = True
        if changed:
            self.registry.record(BASE, metrics=metrics)

    @property
    def model_version(self) -> str:
        return self.serving_bundle

    def version_for(self, stage: str) -> str:
        return self.registry.model_version(self.serving_bundle, stage)

    def stage_for(self, project) -> str | None:
        return stage_of(self.stages, project)

    def _image(self, task: dict, key: str):
        return fetch_image(self.s.ls_url, self.s.ls_token, task["data"][key], self.s.request_timeout)

    def predict_task(self, task: dict, label: tuple | None, stage: str) -> dict:
        label = label or parse_label_config(None)
        img, w, h = self._image(task, label[2])
        with self.lock:
            self._use_stage(stage)
            probs, _ = learner.predict_probs(self.teacher, torch.from_numpy(img)[None, None])
            version = self.version_for(stage)
        conf, mask = probs[0].max(0)
        fg = mask.numpy() > 0
        score = float(conf[torch.from_numpy(fg)].mean()) if fg.any() else 0.0
        # polygons are traced at the original resolution: tracing the 256 map and scaling it up gives staircase edges
        full = torch.nn.functional.interpolate(probs[:1].float(), size=(h, w), mode="bilinear", align_corners=False)
        result = mask_to_results(full[0].argmax(0).numpy().astype(np.uint8), w, h, label[0], label[1], label[3])
        return {"result": result, "score": score, "model_version": version}

    # ---- long-term training --------------------------------------------------------------------
    def next_run(self, now: dt.datetime | None = None) -> str | None:
        if self.cron is None:
            return None
        return self.cron.next_after(now or dt.datetime.now()).isoformat(timespec="minutes")

    def schedule_loop(self) -> None:
        """Fire the long-term job at every cron time; the job itself skips if there are too few new labels."""
        while self.cron is not None:
            due = self.cron.next_after(dt.datetime.now())
            self.telemetry["schedule"]["next_run"] = due.isoformat(timespec="minutes")
            while (left := (due - dt.datetime.now()).total_seconds()) > 0:
                time.sleep(min(left, 60))  # short sleeps: clock changes (suspend, NTP) are picked up
            log.info("scheduled long-term training (%s)", self.cron.expr)
            self.start_job(force=False)

    def on_webhook(self, payload: dict) -> str:
        """Webhooks only count new labels: training runs on the schedule or on /train."""
        if payload.get("action") not in TRAIN_ACTIONS:
            return "ignored-action"
        if self.stage_for((payload.get("project") or {}).get("id")) is None:
            return "ignored-project"
        self.telemetry["webhook_events"] += 1
        return "counted"

    def start_job(self, force: bool, dry_run: bool = False) -> str:
        if not self.train_lock.acquire(blocking=False):
            return "running"
        job_id = uuid.uuid4().hex[:12]

        def run():
            try:
                self.run_job(force, job_id, dry_run=dry_run)
            finally:
                self.train_lock.release()

        threading.Thread(target=run, daemon=True, name=f"tasdf-train-{job_id}").start()
        return job_id

    def _resources_ok(self) -> tuple[bool, str]:
        ram = _free_ram_gb()
        if ram < self.s.min_free_ram_gb:
            return False, f"free RAM {ram:.1f} GB < {self.s.min_free_ram_gb} GB"
        if _on_battery():
            return False, "running on battery: plug in AC power"
        if self.device.type == "cuda":
            temp = trainer.gpu_temp()
            if temp is not None and temp >= trainer.GPU_START_MAX:
                return False, f"GPU at {temp} C >= {trainer.GPU_START_MAX} C"
            free, _ = torch.cuda.mem_get_info(self.device)
            if free / 2 ** 30 < self.s.min_free_vram_gb:
                return False, f"free VRAM {free / 2 ** 30:.1f} GB < {self.s.min_free_vram_gb} GB"
        return True, ""

    def _project_label(self, pid: int) -> tuple:
        cfg = self._ls("GET", f"/api/projects/{pid}").json().get("label_config")
        return parse_label_config(cfg)

    def collect(self, stage: str) -> tuple[list[dict], list[dict], dict]:
        """Reviewed annotations of every project of `stage`, re-read from LS, split by image content."""
        if not (self.s.ls_url and self.s.ls_token):
            # fail closed: never train on anything that was not re-read from LS with our own token
            raise RuntimeError("LS_URL/LS_TOKEN must be set to train")
        train, val, snapshot = [], [], {}
        for pid in self.stages[stage]["projects"]:
            label = self._project_label(pid)
            for d in trainer.select_samples(list(self._project_tasks(pid)), pid, label[0], self.s.min_quality_level):
                img = self._image(d["task"], label[2])[0]
                d["x"] = torch.from_numpy(img)[None]
                d.pop("task")
                (val if trainer.image_split(img) == "val" else train).append(d)
                snapshot.setdefault(str(pid), {})[str(d["task_id"])] = d["digest"]
        return train, val, snapshot

    def run_job(self, force: bool, job_id: str = "", dry_run: bool = False) -> dict | None:
        """Warm-start from production on every reviewed annotation; promote only on a real held-out gain."""
        t0 = time.time()
        stage = list(self.stages)[0]  # shortcut: phase A trains one stage; joint multi-stage training is phase C
        try:
            ok, why = self._resources_ok()
            if not ok:
                self.telemetry["last_skip"] = {"at": t0, "reason": why}
                log.warning("training job %s postponed: %s", job_id, why)
                return None
            self.telemetry["status"] = f"job {job_id}: collecting {stage}"
            train, val, snapshot = self.collect(stage)
            last = self.registry.data.get("last_data_snapshot") or {}
            new = sum(1 for pid, tasks in snapshot.items() for tid, dg in tasks.items()
                      if (last.get(pid) or {}).get(tid) != dg)
            self.telemetry["pending"] = {"stage": stage, "new": new, "train": len(train), "val": len(val),
                                         "threshold": self.s.train_min_new}
            if not train or (not force and new < self.s.train_min_new):
                return None
            x = torch.stack([d["x"] for d in train])
            y = torch.from_numpy(np.stack([d["y"] for d in train]).astype(np.int64))
            vx = torch.stack([d["x"] for d in val]) if val else torch.zeros((0, 1, 256, 256))
            vys = [d["y"] for d in val]
            hold, anchors = self.holdouts[stage], self.anchors[stage]
            parent = self.registry.production
            prod = self.model_for(parent, stage)
            use_val = len(val) >= self.s.project_val_min

            def score(model):
                if use_val or hold is None:
                    return trainer.project_dice(model, vx, vys) or 0.0
                return learner.evaluate(model, *hold)["pooled_mDice"]

            def fit(start, epochs):
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                cand, history = trainer.train_epochs(
                    start, x, y, anchors, mode="heavy", epochs=epochs, lr=self.s.lr, amp=self.amp, score_fn=score,
                    max_seconds=self.s.train_max_hours * 3600)
                if len(train) >= self.s.bn_recal_min:
                    trainer.recalibrate_bn(cand, x)
                return cand, history

            self.telemetry["status"] = f"job {job_id}: training {stage} on {len(train)} (+{len(val)} val)"
            cand, history = fit(prod, 1 if dry_run else self.s.train_epochs)
            if dry_run:
                peak = torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self.device.type == "cuda" else None
                rec = {"job_id": job_id, "dry_run": True, "images": len(train), "epoch_seconds": history[-1]["seconds"],
                       "ms_per_image": round(1000 * history[-1]["seconds"] / max(1, len(train)), 1),
                       "peak_vram_mib": peak}
                self.telemetry["last_dry_run"] = rec
                return rec
            job_count = int(self.registry.data.get("job_count", 0)) + 1
            reason_prefix = "gate"
            if self.s.audit_every and job_count % self.s.audit_every == 0:
                # audit: does the warm-start chain still beat a fresh fit from the baseline on the same data?
                base_cand, _ = fit(self.model_for(BASE, stage), self.s.train_epochs)
                if score(base_cand) > score(cand) + self.s.audit_margin:
                    cand, reason_prefix = base_cand, "audit"
                del base_cand
            cand_val = trainer.mean_dice_per_image(cand, vx, vys)
            prod_val = trainer.mean_dice_per_image(prod, vx, vys)
            cand_train = float(np.mean(trainer.mean_dice_per_image(cand, x, [d["y"] for d in train])))
            d10_ok, d10_reasons, cand_d10 = True, [], None
            if hold is not None:
                cand_d10 = learner.evaluate(cand, *hold)
                base_d10 = ((self.registry.info(BASE).get("metrics") or {}).get(stage) or {}).get("d10")
                d10_ok, d10_reasons = learner.gate(cand_d10, learner.evaluate(prod, *hold), base_d10 or cand_d10)
            shifts = learner.layer_shifts(prod, cand, self.base_state)
            accepted, reasons, note = trainer.promotion_decision(
                cand_val=cand_val, prod_val=prod_val, cand_train=cand_train, d10_ok=d10_ok, d10_reasons=d10_reasons,
                encoder_moved=shifts["encoder_frozen"]["l2_shift"] > 0, val_min=self.s.project_val_min,
                min_gain=self.s.promote_min_gain, max_gap=self.s.max_train_val_gap)
            record = {"job_id": job_id, "stage": stage, "parent": parent, "kind": reason_prefix,
                      "train_images": len(train), "val_images": len(val), "new": new, "history": history[-60:],
                      "metrics": {stage: {"d10": cand_d10, "val": float(np.mean(cand_val)) if cand_val else None,
                                          "production_val": float(np.mean(prod_val)) if prod_val else None,
                                          "train": cand_train}},
                      "layer_shifts": shifts, "note": note, "seconds": round(time.time() - t0, 1)}
            bundle = self.registry.save_bundle(cand.state_dict(), [stage], record)
            with self.lock:
                if self.registry.production != parent:  # a rollback/promote happened meanwhile; candidate is stale
                    accepted, reasons = False, reasons + [f"production changed {parent}->{self.registry.production}"]
                self.registry.record(bundle, status="pending" if accepted else "rejected", gate_reasons=reasons)
                if accepted:
                    self.registry.promote(bundle, reason=reason_prefix, by="gate")
            if accepted:
                self._serve(bundle)
            self.registry.set_meta("last_data_snapshot", snapshot)
            self.registry.set_meta("job_count", job_count)
            trend = (self.registry.data.get("val_trend") or []) + [{
                "at": t0, "bundle": bundle, "accepted": accepted, "val": record["metrics"][stage]["val"],
                "production_val": record["metrics"][stage]["production_val"], "val_images": len(val)}]
            self.registry.set_meta("val_trend", trend[-200:])
            record.update(bundle=bundle, accepted=accepted, gate_reasons=reasons)
            self.telemetry["jobs"] = (self.telemetry["jobs"] + [record])[-20:]
            log.info("job %s -> %s accepted=%s reasons=%s (%.1fs)", job_id, bundle, accepted, reasons, time.time() - t0)
            if accepted:
                self.refresh_ls()
            return record
        except Exception as exc:  # keep the service alive; surface the error in telemetry
            log.exception("training job %s failed", job_id)
            self.telemetry["last_error"] = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            self.telemetry["status"] = "idle"
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- Label Studio sync ------------------------------------------------------------------
    def _auth(self) -> dict:
        return {"Authorization": f"Token {self.s.ls_token}"}

    def _ls(self, method: str, path: str, **kw):
        resp = requests.request(method, f"{self.s.ls_url}{path}", headers=self._auth(), timeout=self.s.request_timeout, **kw)
        resp.raise_for_status()
        return resp

    def _project_tasks(self, pid: int):
        page = 1
        while True:
            resp = requests.get(f"{self.s.ls_url}/api/projects/{pid}/tasks/",
                                params={"page": page, "page_size": 100}, headers=self._auth(), timeout=self.s.request_timeout)
            if resp.status_code == 404 or not resp.json():
                return  # LS paginator raises 404 past the last page
            resp.raise_for_status()
            yield from resp.json()
            page += 1

    def refresh_ls(self) -> dict:
        """Point every configured project at `<production>.<stage>` (optionally precomputing N unlabeled tasks).
        LS validate_model_version accepts it once the backend reports it (/setup) or a prediction has it.
        Old predictions are kept: an annotator may have one open. Needs a project-Manager token."""
        if not (self.s.ls_url and self.s.ls_token):
            return {"skipped": "LS_URL/LS_TOKEN not set"}
        if not self.refresh_lock.acquire(blocking=False):
            return {"skipped": "a refresh is already running"}
        try:
            return {str(pid): self._refresh_project(pid, st) for st, cfg in self.stages.items() for pid in cfg["projects"]}
        finally:
            self.refresh_lock.release()

    def _refresh_project(self, pid: int, stage: str) -> dict:
        precompute_n = max(0, self.s.refresh_next_n)
        name, stats = self.version_for(stage), {"created": 0, "errors": 0}
        try:
            if precompute_n > 0:
                label = self._project_label(pid)
                unlabeled = []
                for t in self._project_tasks(pid):
                    if not any(not a.get("was_cancelled") for a in t.get("annotations", [])) and not any(
                            p.get("model_version") == name for p in t.get("predictions", [])):
                        unlabeled.append(t)
                        if len(unlabeled) >= precompute_n:
                            break
                for t in unlabeled:
                    try:
                        self._ls("POST", "/api/predictions/", json={"task": t["id"], **self.predict_task(t, label, stage)})
                        stats["created"] += 1
                    except Exception:
                        log.exception("prediction for task %s failed", t["id"])
                        stats["errors"] += 1
            try:
                # listing the project's backends runs /setup, which stores the new version on the ML backend in LS
                self._ls("GET", "/api/ml/", params={"project": pid})
                self._ls("PATCH", f"/api/projects/{pid}", json={"model_version": name})
            except requests.RequestException as exc:
                log.warning("could not set project %s model_version=%s: %s", pid, name, exc)
                stats["patch_failed"] = str(exc)
        except requests.RequestException as exc:
            log.warning("Label Studio sync of project %s failed: %s", pid, exc)
            stats["failed"] = str(exc)
        self.telemetry.setdefault("last_refresh", {})[str(pid)] = {"version": name, **stats}
        return stats

    def rollback(self) -> str:
        with self.lock:
            target = self.registry.rollback(by="operator")
        self._serve(target)
        return target

    def promote(self, bundle: str, override: bool = False, reason: str = "") -> dict:
        """Manual promotion. The gate still runs on today's held-out data; `override` skips it but is recorded."""
        if bundle not in self.registry.data["bundles"]:
            raise ValueError(f"unknown bundle {bundle!r}")
        stage = list(self.stages)[0]  # shortcut: phase A has one stage
        reasons: list[str] = []
        if not override:
            _, val, _ = self.collect(stage)
            vx = torch.stack([d["x"] for d in val]) if val else torch.zeros((0, 1, 256, 256))
            vys = [d["y"] for d in val]
            cand, prod, hold = self.model_for(bundle, stage), self.model_for(self.registry.production, stage), self.holdouts[stage]
            d10_ok, d10_reasons = True, []
            if hold is not None:
                base_d10 = ((self.registry.info(BASE).get("metrics") or {}).get(stage) or {}).get("d10")
                cand_d10 = learner.evaluate(cand, *hold)
                d10_ok, d10_reasons = learner.gate(cand_d10, learner.evaluate(prod, *hold), base_d10 or cand_d10)
            ok, reasons, _ = trainer.promotion_decision(
                cand_val=trainer.mean_dice_per_image(cand, vx, vys), prod_val=trainer.mean_dice_per_image(prod, vx, vys),
                cand_train=None, d10_ok=d10_ok, d10_reasons=d10_reasons, encoder_moved=False,
                val_min=self.s.project_val_min, min_gain=self.s.promote_min_gain, max_gap=self.s.max_train_val_gap)
            del cand, prod
            gc.collect()
            if not ok:
                return {"promoted": False, "reasons": reasons}
        with self.lock:
            self.registry.promote(bundle, reason=reason or ("override" if override else "manual"), by="operator")
        self._serve(bundle)
        return {"promoted": True, "reasons": reasons, "refresh": self.refresh_ls()}


def create_app(settings: Settings | None = None, service: OnlineService | None = None) -> Flask:
    settings = settings or Settings.from_env()
    # gunicorn configures only its own loggers: without this, INFO logs of tasdf_online are dropped in Docker
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("settings: %s", {k: v for k, v in dataclasses.asdict(settings).items() if k not in ("ls_token", "ol_token")})
    svc = service
    if svc is None:
        svc = OnlineService(settings)
        # sync LS with the production bundle after every restart (needs LS reachable)
        threading.Thread(target=svc.refresh_ls, daemon=True).start()
        threading.Thread(target=svc.schedule_loop, daemon=True, name="tasdf-schedule").start()
    app = Flask(__name__)
    app.config["svc"] = svc

    def ops_allowed() -> bool:
        # header only: a ?token= query would end up in the gunicorn access log
        if settings.ol_token:
            return request.headers.get("X-OL-Token") == settings.ol_token
        # no OL_TOKEN: open only while the dev dashboard is configured on; the product runs with DASHBOARD_ENABLED=0
        return settings.dashboard_enabled

    def stage_or_400(project):
        stage = svc.stage_for(project)
        if stage is None:
            raise ValueError(f"project {project!r} is not configured in {settings.stages_file}")
        return stage

    @app.get("/health")
    def health():
        return jsonify(status="UP", model_class="TASDFOnline", model_version=svc.model_version, device=str(svc.device))

    @app.post("/setup")
    def setup():
        payload = request.get_json(silent=True) or {}
        return jsonify(model_version=svc.version_for(stage_or_400(payload.get("project"))))

    @app.post("/validate")
    def validate():
        payload = request.get_json(silent=True) or {}
        parse_label_config(payload.get("config") or payload.get("label_config"))
        return jsonify(status="ok")

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        stage = stage_or_400(payload.get("project"))
        # per-request config: another project of the same stage may label differently
        label = parse_label_config(payload["label_config"]) if payload.get("label_config") else None
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return jsonify(error="tasks must be a non-empty list"), 400
        results = []
        for t in tasks:
            try:
                results.append(svc.predict_task(t, label, stage))
            except Exception:  # one bad task must not fail the whole batch
                log.exception("predict failed for task %s", t.get("id") if isinstance(t, dict) else t)
                results.append({"result": [], "score": 0.0, "model_version": svc.version_for(stage)})
        return jsonify(results=results, model_version=svc.version_for(stage))

    @app.post("/webhook")
    def webhook():
        return jsonify(status=svc.on_webhook(request.get_json(silent=True) or {}))

    @app.post("/train")
    def train():
        # same trust level as /webhook (LS "Start Training" sends no X-OL-Token); data is re-read from LS anyway
        body = request.get_json(silent=True) or {}
        job = svc.start_job(force=bool(body.get("force")), dry_run=bool(body.get("dry_run")))
        if job == "running":
            return jsonify(status="running"), 409
        return jsonify(job_id=job, status="started")

    @app.get("/versions")
    def versions():
        return jsonify(versions=[b for b, i in svc.registry.data["bundles"].items() if i.get("status") != "rejected"])

    @app.get("/telemetry/live")
    def telemetry():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        bundles = {b: {k: i.get(k) for k in ("status", "parent", "metrics", "gate_reasons", "note", "created", "kind")}
                   for b, i in svc.registry.data["bundles"].items()}
        return jsonify(production=svc.registry.production, previous=svc.registry.previous, device=str(svc.device),
                       training=svc.train_lock.locked(), dashboard_enabled=svc.dashboard_enabled, bundles=bundles,
                       val_trend=svc.registry.data.get("val_trend", []), **svc.telemetry)

    @app.get("/registry")
    def registry():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        d = svc.registry.data
        snapshot = d.get("last_data_snapshot") or {}
        return jsonify(pointers=d["pointers"], bundles=d["bundles"], history=d["history"], legacy=d.get("legacy", {}),
                       stages=svc.stages, data_snapshot={p: len(t) for p, t in snapshot.items()},
                       job_count=d.get("job_count", 0), val_trend=d.get("val_trend", []),
                       schedule=svc.telemetry.get("schedule"), pending=svc.telemetry.get("pending"))

    @app.post("/promote")
    def promote():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        body = request.get_json(silent=True) or {}
        if not svc.train_lock.acquire(blocking=False):  # never swap production under a running job
            return jsonify(status="training job running"), 409
        try:
            return jsonify(svc.promote(str(body.get("bundle", "")), bool(body.get("override")), str(body.get("reason", ""))))
        finally:
            svc.train_lock.release()

    @app.post("/rollback")
    def rollback():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        return jsonify(production=svc.rollback(), refresh=svc.refresh_ls())

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
