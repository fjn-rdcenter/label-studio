"""Single model registry: bundles = shared decoder + one adapter per embryo stage, on top of a fixed baseline.

    registry/registry.json          bundles, pointers {production, previous}, history, legacy notes
    registry/decoders/<dec-id>.pt   conv weights of up1..up4 (both branches), fp16
    registry/adapters/<stage>/<ad-id>.pt   BN affine + BN running stats + pixel/topo heads of one stage

Names are content hashes (`tasdf-<yyyymmdd>-<hash6>`), never reused. `tasdf-base` is the baseline itself.
LS sees `<bundle>.<stage>` as model_version.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from .learner import is_bn, is_decoder

BASE = "tasdf-base"
KEEP_RECENT = 3


def is_adapter_key(name: str) -> bool:
    return is_bn(name) or ".heads." in name


def is_decoder_key(name: str) -> bool:
    return is_decoder(name) and not is_bn(name)


def decoder_state(state: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in state.items() if is_decoder_key(k)}


def adapter_state(state: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in state.items() if is_adapter_key(k)}


def _digest(tensors: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        h.update(k.encode())
        h.update(tensors[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:6]


def _today() -> str:
    return time.strftime("%Y%m%d")


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class BundleRegistry:
    def __init__(self, root: str | Path, base_state: dict, base_sha: str, stages: list[str]):
        self.root = Path(root)
        self.base_state = base_state
        self.path = self.root / "registry.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data["base_sha"] != base_sha:
                raise RuntimeError(
                    f"BASELINE_CKPT changed ({self.data['base_sha'][:12]} -> {base_sha[:12]}): bundles in {self.root} "
                    "were trained on the old baseline. Restore it or use a new registry directory.")
        else:
            self.data = {"base_sha": base_sha, "bundles": {}, "pointers": {"production": BASE, "previous": BASE},
                         "history": [], "legacy": {}}
        self.data["bundles"].setdefault(BASE, {"decoder": "dec-base", "adapters": {}, "status": "baseline",
                                               "created": time.time(), "metrics": {}})
        for stage in stages:  # a new stage starts from the baseline adapter
            self.data["bundles"][BASE]["adapters"].setdefault(stage, f"ad-{stage}-base")
        self._save()

    # ---- persistence -------------------------------------------------------------------------
    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name("registry.json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=float), encoding="utf-8")
        os.replace(tmp, self.path)

    @property
    def production(self) -> str:
        return self.data["pointers"]["production"]

    @property
    def previous(self) -> str:
        return self.data["pointers"]["previous"]

    def info(self, bundle: str) -> dict:
        return self.data["bundles"][bundle]

    def model_version(self, bundle: str, stage: str) -> str:
        return f"{bundle}.{stage}"

    # ---- bundles -----------------------------------------------------------------------------
    def save_bundle(self, state: dict, stages: list[str], meta: dict, fp16: bool = True) -> str:
        """Store the decoder of `state` and the adapter of each stage (one trained state per stage list).

        Phase A trains one model for its single stage; with several stages pass one call per training
        result and the same decoder (identical decoder tensors dedupe to the same id)."""
        dec = decoder_state(state)
        dec_id = f"dec-{_today()}-{_digest(dec)}"
        dec_path = self.root / "decoders" / f"{dec_id}.pt"
        if not dec_path.exists():
            _atomic_save({k: (v.half() if fp16 and v.is_floating_point() else v) for k, v in dec.items()}, dec_path)
        adapters = {}
        ad = adapter_state(state)
        for stage in stages:
            ad_id = f"ad-{stage}-{_today()}-{_digest(ad)}"
            ad_path = self.root / "adapters" / stage / f"{ad_id}.pt"
            if not ad_path.exists():
                _atomic_save(ad, ad_path)
            adapters[stage] = ad_id
        bundle = f"tasdf-{_today()}-{hashlib.sha256((dec_id + json.dumps(adapters, sort_keys=True)).encode()).hexdigest()[:6]}"
        self.data["bundles"][bundle] = {"decoder": dec_id, "adapters": adapters, "status": "candidate",
                                        "created": time.time(), **meta}
        self._save()
        return bundle

    def state_for(self, bundle: str, stage: str) -> dict:
        """Full state dict (baseline + bundle decoder + stage adapter), float32."""
        info = self.info(bundle)
        state = {k: v.clone() for k, v in self.base_state.items()}
        if info["decoder"] != "dec-base":
            dec = torch.load(self.root / "decoders" / f"{info['decoder']}.pt", map_location="cpu", weights_only=True)
            state.update({k: v.float() if v.is_floating_point() else v for k, v in dec.items()})
        ad_id = info["adapters"].get(stage)
        if ad_id is None:
            raise KeyError(f"bundle {bundle} has no adapter for stage {stage!r}")
        if not ad_id.endswith("-base"):
            state.update(torch.load(self.root / "adapters" / stage / f"{ad_id}.pt", map_location="cpu", weights_only=True))
        return state

    def set_meta(self, key: str, value) -> None:
        """Top-level bookkeeping that must survive restarts (last_data_snapshot, job_count, val_trend)."""
        self.data[key] = value
        self._save()

    def record(self, bundle: str, **fields) -> None:
        self.data["bundles"][bundle].update(fields)
        self._save()

    # ---- pointers ----------------------------------------------------------------------------
    def promote(self, bundle: str, reason: str, by: str = "gate") -> None:
        if bundle not in self.data["bundles"]:
            raise KeyError(bundle)
        old = self.production
        if bundle == old:
            return
        self.data["pointers"] = {"production": bundle, "previous": old}
        self.data["bundles"][bundle]["status"] = "production"
        if self.data["bundles"][old]["status"] == "production":
            self.data["bundles"][old]["status"] = "retired"
        self.data["history"].append({"at": time.time(), "from": old, "to": bundle, "by": by, "reason": reason})
        self._gc()
        self._save()

    def rollback(self, by: str = "operator") -> str:
        target = self.previous
        self.promote(target, reason="rollback", by=by)
        return target

    def _gc(self) -> None:
        """Keep baseline, production, previous and the KEEP_RECENT newest bundles; drop unreferenced files."""
        bundles = self.data["bundles"]
        recent = sorted((b for b in bundles if b != BASE), key=lambda b: bundles[b]["created"])[-KEEP_RECENT:]
        keep = {BASE, self.production, self.previous, *recent}
        for b in [b for b in bundles if b not in keep]:
            del bundles[b]
        used_dec = {i["decoder"] for i in bundles.values()}
        used_ad = {(s, a) for i in bundles.values() for s, a in i["adapters"].items()}
        for f in (self.root / "decoders").glob("*.pt"):
            if f.stem not in used_dec:
                f.unlink()
        for f in (self.root / "adapters").glob("*/*.pt"):
            if (f.parent.name, f.stem) not in used_ad:
                f.unlink()

    # ---- migration ---------------------------------------------------------------------------
    def migrate_legacy(self, state_dir: str | Path) -> list[str]:
        """Move the old per-project registries (state/<project>/) under registry/legacy/ for reference only."""
        moved = []
        state_dir = Path(state_dir)
        for d in sorted(p for p in state_dir.glob("*") if p.is_dir() and (p / "registry.json").exists()):
            dest = self.root / "legacy" / d.name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            old = json.loads((d / "registry.json").read_text(encoding="utf-8"))
            shutil.move(str(d), str(dest))
            self.data["legacy"][d.name] = {"active_was": old.get("active"), "versions": sorted(old.get("versions", {})),
                                           "note": f"old tasdf-vN names of project {d.name}; not served"}
            moved.append(d.name)
        if moved:
            self._save()
        return moved
