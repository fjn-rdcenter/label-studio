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

from . import journal, learner, trainer
from .adapter import SOURCES, Sample, WellAdapters, adapt, choose_source
from .bundle_registry import BASE, BundleRegistry, adapter_state
from .config import Settings, load_stages, resolve_device, stage_of
from .crop import crop_box, crop_input, from_box, to_box
from .dashboard import create_dashboard_blueprint
from .ls_format import CLASS_NAMES, IGNORE, NAME_TO_CLASS, annotation_target, fetch_gray, fetch_image, mask_to_results
from .propagate import RING_CLASSES, object_class, object_mask, object_result, refine_object, stop_reason
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


def ls_project_label(url: str, token: str, timeout: float, pid: int) -> tuple:
    """parse_label_config of LS project `pid` (shared with scripts/learning_curve.py)."""
    resp = requests.request("GET", f"{url}/api/projects/{pid}", headers={"Authorization": f"Token {token}"},
                            timeout=timeout)
    resp.raise_for_status()
    return parse_label_config(resp.json().get("label_config"))


def ls_project_tasks(url: str, token: str, timeout: float, pid: int):
    """Every task of LS project `pid`, page by page (shared with scripts/learning_curve.py)."""
    page = 1
    while True:
        resp = requests.get(f"{url}/api/projects/{pid}/tasks/", params={"page": page, "page_size": 100},
                            headers={"Authorization": f"Token {token}"}, timeout=timeout)
        if resp.status_code == 404 or not resp.json():
            return  # LS paginator raises 404 past the last page
        resp.raise_for_status()
        yield from resp.json()
        page += 1


def _propagation_context(ctx: dict) -> tuple[list[dict], int, float, dict | None]:
    """Validate the editor's propagate context -> (carried, distance, threshold, gold). /predict has no auth."""
    carried, distance, threshold, gold = ctx.get("carried"), ctx.get("distance"), ctx.get("threshold", 0.8), ctx.get("gold")
    if not isinstance(carried, list) or not all(isinstance(r, dict) for r in carried):
        raise ValueError("context.carried must be a list of results")
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("context.distance must be a non-negative integer")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("context.threshold must be a number")
    if gold is not None and not (isinstance(gold, dict) and isinstance(gold.get("url"), str)
                                 and isinstance(gold.get("results", []), list)):
        raise ValueError("context.gold must be {url, results}")
    if gold is not None:
        gold = {"url": gold["url"], "results": [r for r in gold.get("results", []) if isinstance(r, dict)]}
    return carried, distance, min(max(float(threshold), 0.0), 1.0), gold


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_npz(path: str, limit: int = 0):
    if not path:
        return None
    if not os.path.exists(path):
        log.warning("%s does not exist: not loaded", path)
        return None
    data = np.load(path)
    images, masks = data["images"], data["masks"]
    if limit:
        images, masks = images[:limit], masks[:limit]
    return torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1), masks.astype(np.int64)


def _free_ram_gb(meminfo: str = "/proc/meminfo", cgroup: str = "/sys/fs/cgroup") -> float:
    """Free RAM for this process: the host's MemAvailable, capped by what is left under the container's
    cgroup v2 limit (compose mem_limit). Page cache we could drop (inactive_file) does not count as used."""
    host = float("inf")
    with open(meminfo) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) / 2 ** 20
                break
    try:
        limit = Path(cgroup, "memory.max").read_text().strip()
        if limit == "max":
            return host
        used = int(Path(cgroup, "memory.current").read_text())
        stat = dict(line.split() for line in Path(cgroup, "memory.stat").read_text().splitlines() if line.strip())
        used -= int(stat.get("inactive_file", 0))
        return min(host, max(0, int(limit) - used) / 2 ** 30)
    except (OSError, ValueError):  # no cgroup v2 files (host run, cgroup v1): the host figure is all we have
        return host


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
        uses_global = any(not cfg.get("holdout_npz") for cfg in self.stages.values())
        if uses_global and s.holdout_npz and not os.path.exists(s.holdout_npz):
            # fail closed: without the holdout the D10 forgetting gate would be skipped silently
            raise RuntimeError(f"HOLDOUT_NPZ={s.holdout_npz} does not exist (unset HOLDOUT_NPZ, or TASDF_HOLDOUT_NPZ= in compose, to run without D10)")
        self.holdouts = {st: _load_npz(cfg.get("holdout_npz") or s.holdout_npz, s.holdout_limit)
                         for st, cfg in self.stages.items()}
        self.anchors = {st: _load_npz(cfg.get("anchors_npz") or s.anchor_npz) if s.anchor_frac > 0 else None
                        for st, cfg in self.stages.items()}  # not loaded at all when unused (RAM)
        self.lock = threading.RLock()
        self.train_lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending_delay = 5.0  # seconds after a webhook burst before labels are counted again
        self._pending_timer: threading.Timer | None = None
        self._pending_sched = threading.Lock()  # guards _pending_timer / _pending_dirty
        self._pending_dirty = False  # a webhook arrived since the running count started
        self._split_cache: dict[tuple, str] = {}  # (project, task, image) -> "train" | "val": fetch each image once
        self.dashboard_enabled: bool = s.dashboard_enabled
        self.telemetry: dict = {"status": "idle", "jobs": [], "webhook_events": 0}
        self.cron = Cron(s.train_cron) if s.train_cron.strip() else None  # bad expression: fail at startup
        self.telemetry["schedule"] = {"cron": s.train_cron or None, "next_run": self.next_run()}
        # short-term tier: one adapter per open well; a long-term job yields the GPU while `interactive` is set
        self.wells = WellAdapters(s.adapt_max_wells)
        self.adapt_lock = threading.Lock()
        self.interactive = threading.Event()
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
        adapt_model = self.model_for(bundle, stages[0])  # well adapters are loaded into this copy, never into teacher
        with self.lock:
            self.teacher, self.adapters, self.serving_bundle, self.serving_stage = teacher, adapters, bundle, stages[0]
        with self.adapt_lock:
            self.adapt_model = adapt_model
            self.wells.clear()  # every well adapter started from the old production
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

    # ---- short-term tier: "Done image" -> learn the well, propagate to the next frames ----------
    def _probs(self, x: np.ndarray, stage: str, adapter: dict | None) -> np.ndarray:
        """(C, 256, 256) class probabilities of production, or of production with a well adapter."""
        xt = torch.from_numpy(x)[None, None]
        if adapter is None:
            with self.lock:
                self._use_stage(stage)
                probs, _ = learner.predict_probs(self.teacher, xt)
        else:
            self.adapt_model.load_state_dict(adapter, strict=False)
            probs, _ = learner.predict_probs(self.adapt_model, xt)
        return probs[0].numpy()

    def _learn_gold(self, state, gold: dict, label: tuple, stage: str, key: tuple[int, int]):
        """Score the proposals once shown on the confirmed image, store it as a sample, adapt the well."""
        results, url = gold["results"], gold["url"]
        gray = fetch_gray(self.s.ls_url, self.s.ls_token, url, self.s.request_timeout)
        h, w = gray.shape
        box = crop_box(results, w, h, self.s.crop_size)
        x = crop_input(gray, box)
        y = annotation_target(to_box(results, w, h, box))
        prior = self._probs(x, stage, None).argmax(0)
        scores, changed = None, np.zeros(y.shape, bool)
        shown = state.proposals.get(url)
        if shown:
            used = choose_source(state.scores)  # the source the proposal on this image was taken from
            scores = {src: trainer.mean_dice_ignore(m, y) for src, m in shown.items()}
            state.scores.append(scores)
            changed = (shown[used] != y) & (y != IGNORE)
        stats = None
        if any(object_class(r) is not None for r in results):  # nothing the model knows: nothing to learn
            state.samples.append(Sample(torch.from_numpy(x)[None], torch.from_numpy(y.astype(np.int64)),
                                        torch.from_numpy(changed), torch.from_numpy(prior.astype(np.int64))))
            stats = adapt(self.adapt_model, state, self.adapters[stage], steps=self.s.adapt_steps, lr=self.s.adapt_lr)
        journal.append(self.s.journal_dir, key[0], key[1], {"event": "gold", "url": url, "scores": scores, "adapt": stats})
        journal.save_arrays(self.s.journal_dir, key[0], key[1], url, box=np.asarray(box), gold=y,
                            **{src: m for src, m in (shown or {}).items()})
        return scores, stats

    def propagate(self, task: dict, label: tuple, stage: str, ctx: dict) -> dict:
        label = label or parse_label_config(None)
        carried, distance, threshold, gold = _propagation_context(ctx)
        key = (int(str(task.get("project") or 0).split(".")[0]), int(task["id"]))
        version = self.version_for(stage)
        with self.adapt_lock:
            self.interactive.set()
            try:
                prod_adapter = self.adapters[stage]
                state = self.wells.get(key, lambda: {k: v.clone() for k, v in prod_adapter.items()})
                scores = stats = None
                if gold and gold["results"]:
                    scores, stats = self._learn_gold(state, gold, label, stage, key)
                meta = {"distance": distance, "adapt": stats, "scores": scores, "source": choose_source(state.scores)}
                if not carried:
                    return {"result": [], "score": 0.0, "model_version": version,
                            "propagation": {"stop": "no_objects", **meta}}
                url = task["data"][label[2]]
                gray = fetch_gray(self.s.ls_url, self.s.ls_token, url, self.s.request_timeout)
                h, w = gray.shape
                box = crop_box(carried, w, h, self.s.crop_size)
                x = crop_input(gray, box)
                maps = {"production": self._probs(x, stage, None), "adapter": self._probs(x, stage, state.adapter)}
                source = meta["source"]
                free_map = maps["adapter" if source == "adapter" else "production"].argmax(0)
                proposals = {src: np.zeros((x.shape[0], x.shape[1]), np.uint8) for src in SOURCES}
                out, stop = [], None
                for orig, r in zip(carried, to_box(carried, w, h, box)):
                    cls = object_class(r)
                    if cls is None:  # a label the model does not know: carried over unchanged
                        out.append(orig)
                        continue
                    ring, cm = cls in RING_CLASSES, object_mask(r, x.shape[0])
                    refined = {"copy": cm}
                    for src in ("production", "adapter"):
                        refined[src] = refine_object(cm, maps[src].argmax(0) == cls, self.s.band_px, ring)
                    for src, m in refined.items():
                        proposals[src][m] = cls
                    reason = stop_reason(cm, refined[source], free_map == cls, ring, threshold)
                    res = object_result(refined[source], r, cls)
                    stop = stop or reason or (None if res else "vanished")
                    out.append(from_box([res], w, h, box)[0] if res else orig)
                state.proposals[url] = proposals
                if stop:  # keep the carried copy on this image and let the physician look
                    out = list(carried)
                journal.append(self.s.journal_dir, key[0], key[1],
                               {"event": "propose", "url": url, "distance": distance, "source": source, "stop": stop})
                return {"result": out, "score": 0.0 if stop else 1.0, "model_version": version,
                        "propagation": {"stop": stop, **meta}}
            finally:
                self.interactive.clear()

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
        self.schedule_pending_count()
        return "counted"

    def schedule_pending_count(self) -> None:
        """Count labels again shortly after a burst of webhooks: one count per burst, never two at once.
        A webhook that arrives while a count runs gets one more count after it."""
        with self._pending_sched:
            self._pending_dirty = True
            if self._pending_timer is not None:
                return
            self._pending_timer = threading.Timer(self.pending_delay, self._pending_loop)
            self._pending_timer.daemon = True
            self._pending_timer.start()

    def _pending_loop(self) -> None:
        while True:
            with self._pending_sched:
                self._pending_dirty = False
            self.count_pending(wait=True)
            with self._pending_sched:
                if not self._pending_dirty:
                    self._pending_timer = None
                    return
            time.sleep(self.pending_delay)

    def _new_labels(self, snapshot: dict) -> int:
        """Reviewed annotations added or changed since the last training job."""
        last = self.registry.data.get("last_data_snapshot") or {}
        return sum(1 for pid, tasks in snapshot.items() for tid, dg in tasks.items()
                   if (last.get(pid) or {}).get(tid) != dg)

    def count_pending(self, stage: str | None = None, wait: bool = False) -> dict:
        """Dashboard counts without training: reviewed samples per side, new since the last job, and annotations
        still waiting for review (below MIN_QUALITY_LEVEL). Only images never split before are fetched.
        `wait` queues behind a running count instead of skipping (webhook recounts, end of a job)."""
        stage = stage or list(self.stages)[0]
        if not (self.s.ls_url and self.s.ls_token):
            return {"skipped": "LS_URL/LS_TOKEN not set"}
        if not self.pending_lock.acquire(blocking=wait):
            return {"skipped": "a count is already running"}
        try:
            counts, snapshot = {"train": 0, "val": 0, "unreviewed": 0}, {}
            for pid in self.stages[stage]["projects"]:
                label = self._project_label(pid)
                tasks = list(self._project_tasks(pid))
                counts["unreviewed"] += sum(
                    1 for t in tasks for a in t.get("annotations") or []
                    if a.get("result") and not a.get("was_cancelled")
                    and int(a.get("quality_level") or 1) < self.s.min_quality_level)
                for d in trainer.select_samples(tasks, pid, label[0], self.s.min_quality_level):
                    key = (pid, d["task_id"], d["task"]["data"][label[2]])
                    if key not in self._split_cache:
                        self._split_cache[key] = trainer.image_split(self._image(d["task"], label[2])[0])
                    counts[self._split_cache[key]] += 1
                    snapshot.setdefault(str(pid), {})[str(d["task_id"])] = d["digest"]
            pending = {"stage": stage, "new": self._new_labels(snapshot), **counts,
                       "threshold": self.s.train_min_new, "at": time.time()}
        except Exception as exc:  # LS down or a bad image: keep the last counts, say why
            log.warning("counting labels failed: %s", exc)
            pending = {**(self.telemetry.get("pending") or {}), "error": f"{type(exc).__name__}: {exc}",
                       "at": time.time()}
        finally:
            self.pending_lock.release()
        self.telemetry["pending"] = pending
        return pending

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
        return ls_project_label(self.s.ls_url, self.s.ls_token, self.s.request_timeout, pid)

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
                side = self._split_cache[(pid, d["task_id"], d.pop("task")["data"][label[2]])] = trainer.image_split(img)
                (val if side == "val" else train).append(d)
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
            new = self._new_labels(snapshot)
            pending = {**(self.telemetry.get("pending") or {}), "stage": stage, "new": new, "train": len(train),
                       "val": len(val), "threshold": self.s.train_min_new, "at": time.time()}
            pending.pop("error", None)  # LS answered: an older count failure no longer applies
            self.telemetry["pending"] = pending
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
                # fixed epochs, no early stopping: picking the best epoch on val and then gating on the same val
                # would inflate the gain the gate sees; val stays untouched until the gate
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                cand, history = trainer.train_epochs(
                    start, x, y, anchors, mode="heavy", epochs=epochs, lr=self.s.lr, amp=self.amp,
                    anchor_frac=self.s.anchor_frac,
                    max_seconds=self.s.train_max_hours * 3600, pause=self.interactive)
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
                # shortcut: picking the better of two candidates on val, then gating on that val, is a small
                # best-of-2 bias; it is well inside the gate's 2x bootstrap margin
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
            self.count_pending(wait=True)  # against the new snapshot: labels saved during the job stay counted
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
        return ls_project_tasks(self.s.ls_url, self.s.ls_token, self.s.request_timeout, pid)

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
        threading.Thread(target=svc.count_pending, daemon=True, name="tasdf-count").start()
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
        ctx = payload.get("context")
        if isinstance(ctx, dict) and ctx.get("action") == "propagate":
            if len(tasks) != 1 or not isinstance(tasks[0], dict):
                raise ValueError("propagate needs exactly one task")
            task = {**tasks[0], "project": tasks[0].get("project") or payload.get("project")}
            result = svc.propagate(task, label, stage, ctx)
            return jsonify(results=[result], model_version=result["model_version"])
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
        # LS offers these in the project's model-version list and saves only a version the backend reports (/setup)
        # or a prediction carries: list exactly what is served. Switching versions is /promote or /rollback.
        return jsonify(versions=[svc.version_for(st) for st in svc.stages])

    @app.get("/telemetry/live")
    def telemetry():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        d = svc.registry.snapshot()
        bundles = {b: {k: i.get(k) for k in ("status", "parent", "metrics", "gate_reasons", "note", "created", "kind")}
                   for b, i in d["bundles"].items()}
        return jsonify(production=d["pointers"]["production"], previous=d["pointers"]["previous"], device=str(svc.device),
                       training=svc.train_lock.locked(), dashboard_enabled=svc.dashboard_enabled, bundles=bundles,
                       val_trend=d.get("val_trend", []), **svc.telemetry)

    @app.get("/registry")
    def registry():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        d = svc.registry.snapshot()
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
        if not svc.train_lock.acquire(blocking=False):  # a finishing job would serve its bundle over the rollback
            return jsonify(status="training job running"), 409
        try:
            target = svc.rollback()
        finally:
            svc.train_lock.release()
        return jsonify(production=target, refresh=svc.refresh_ls())

    @app.post("/pending/refresh")
    def pending_refresh():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        return jsonify(svc.count_pending())

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
