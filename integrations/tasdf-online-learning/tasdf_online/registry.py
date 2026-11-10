"""Model versions as v0 (baseline checkpoint) + per-version trainable-parameter state.

Frozen weights and BN running stats never change in norm_and_heads mode, so a version is fully
described by its ~311k trainable tensors (~1.25 MB) on top of v0. Each file holds the absolute
values of those tensors (not a diff), so loading any version is a single overlay.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def _atomic_write_bytes(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class Registry:
    def __init__(self, state_dir: str, project_id: int):
        self.root = Path(state_dir) / str(project_id)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"active": "v0", "next": 1, "versions": {"v0": {"status": "baseline", "created": time.time()}}}
            self._save()

    def _save(self) -> None:
        payload = json.dumps(self.data, indent=1, default=float).encode("utf-8")
        _atomic_write_bytes(self.path, lambda p: p.write_bytes(payload))

    @property
    def active(self) -> str:
        return self.data["active"]

    def info(self, version: str) -> dict:
        return self.data["versions"][version]

    def sync_fingerprint(self, fp: dict) -> bool:
        """Remember which baseline/holdout the versions and metrics were built from.
        Refuse a different baseline (deltas would be applied to the wrong weights); return True if the holdout changed."""
        old = self.data.get("fingerprint")
        if old and old.get("baseline") != fp["baseline"]:
            raise RuntimeError(
                f"BASELINE_CKPT changed ({old['baseline'][:12]} -> {fp['baseline'][:12]}): versions in {self.root} "
                "were trained on the old baseline. Restore it or use a new STATE_DIR."
            )
        changed = old != fp
        if changed:
            self.data["fingerprint"] = fp
            self._save()
        return changed

    def record(self, version: str, **fields) -> None:
        self.data["versions"].setdefault(version, {}).update(fields)
        self._save()

    def add(self, trainable: dict, meta: dict, accepted: bool) -> str:
        """Store a candidate; accepted ones become active atomically. Rejected ones keep metrics only."""
        version = f"v{self.data['next']}"
        self.data["next"] += 1
        if accepted:
            _atomic_write_bytes(self.root / "versions" / f"{version}.pt", lambda p: torch.save(trainable, p))
        self.data["versions"][version] = {"status": "accepted" if accepted else "rejected", "parent": self.active,
                                          "created": time.time(), **meta}
        if accepted:
            self.data["active"] = version
        self._save()
        return version

    def load_into(self, model: torch.nn.Module, base_state: dict, version: str) -> None:
        model.load_state_dict(base_state)
        if version != "v0":
            overlay = torch.load(self.root / "versions" / f"{version}.pt", map_location="cpu", weights_only=True)
            missing = set(overlay) - set(dict(model.named_parameters()))
            if missing:
                raise KeyError(f"{version} has unknown tensors: {sorted(missing)[:3]}")
            model.load_state_dict(overlay, strict=False)

    def rollback(self, version: str) -> None:
        if version != "v0" and not (self.root / "versions" / f"{version}.pt").exists():
            raise FileNotFoundError(f"no stored weights for {version}")
        self.data["active"] = version
        self.data.setdefault("rollbacks", []).append({"to": version, "at": time.time()})
        self._save()
