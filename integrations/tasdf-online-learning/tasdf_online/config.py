import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _fraction(name: str, default: float) -> float:
    value = _float(name, default)
    if not 0 <= value < 1:
        raise ValueError(f"{name}={value} must be in [0, 1)")
    return value


def resolve_device(value: str):
    """'auto' | 'cpu' | 'cuda' | 'cuda:N' -> torch.device; fail fast if CUDA was asked for but is missing."""
    import torch

    value = (value or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={value} but CUDA is not available")
    return device


@dataclass(frozen=True)
class Settings:
    device: str
    torch_threads: int
    amp: str
    baseline_ckpt: str
    holdout_npz: str
    anchor_npz: str
    state_dir: str
    registry_dir: str
    stages_file: str
    ls_url: str
    ls_token: str
    ol_token: str
    min_quality_level: int
    refresh_next_n: int
    train_cron: str
    train_min_new: int
    train_epochs: int
    train_max_hours: float
    project_val_min: int
    promote_min_gain: float
    max_train_val_gap: float
    bn_recal_min: int
    audit_every: int
    audit_margin: float
    min_free_ram_gb: float
    min_free_vram_gb: float
    holdout_limit: int
    lr: float
    anchor_frac: float
    base_channels: int
    erosion: str
    request_timeout: float
    dashboard_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            device=os.getenv("DEVICE", "auto"),
            torch_threads=_int("TORCH_THREADS", 0),
            amp=os.getenv("AMP", "auto"),
            baseline_ckpt=os.getenv("BASELINE_CKPT", ""),
            holdout_npz=os.getenv("HOLDOUT_NPZ", ""),
            anchor_npz=os.getenv("ANCHOR_NPZ", ""),
            state_dir=os.getenv("STATE_DIR", "/data/tasdf-online/state"),  # legacy per-project registries (migrated)
            registry_dir=os.getenv("REGISTRY_DIR", "/data/tasdf-online/registry"),
            stages_file=os.getenv("STAGES_FILE", "/data/tasdf-online/stages.json"),
            ls_url=os.getenv("LS_URL", "").rstrip("/"),
            ls_token=os.getenv("LS_TOKEN", ""),
            ol_token=os.getenv("OL_TOKEN", ""),
            min_quality_level=_int("MIN_QUALITY_LEVEL", 2),
            refresh_next_n=_int("REFRESH_NEXT_N", 0),
            # long-term training (weekly or POST /train): measured 49 ms/image/epoch, 2.4 GB VRAM on an RTX 3060
            # local time of the container (TZ); empty disables the schedule (POST /train still works)
            train_cron=os.getenv("TRAIN_CRON", "0 1 * * 0"),
            train_min_new=_int("TRAIN_MIN_NEW", 10),
            train_epochs=_int("TRAIN_EPOCHS", 60),
            train_max_hours=_float("TRAIN_MAX_HOURS", 2),
            project_val_min=_int("PROJECT_VAL_MIN", 10),
            promote_min_gain=_float("PROMOTE_MIN_GAIN", 0.005),
            max_train_val_gap=_float("MAX_TRAIN_VAL_GAP", 0.15),
            bn_recal_min=_int("BN_RECAL_MIN", 32),
            audit_every=_int("AUDIT_EVERY", 4),
            audit_margin=_float("AUDIT_MARGIN", 0.01),
            # a job waits (and says why) instead of starting when the machine is short of memory
            min_free_ram_gb=_float("MIN_FREE_RAM_GB", 3),
            min_free_vram_gb=_float("MIN_FREE_VRAM_GB", 3),
            holdout_limit=_int("HOLDOUT_LIMIT", 0),
            lr=_float("LR", 3e-5),  # 3e-4 wrecked the pretrained decoder in its first steps
            # share of each batch taken from ANCHOR_NPZ; 0 by default: D10 anchors pulled the model off the project
            # labels (learning curve 2026-11-14: light -0.018 with anchors, -0.002 without; D10 did not drop)
            anchor_frac=_fraction("ANCHOR_FRAC", 0.0),
            # must match the baseline checkpoint (M2-B: 64 / cross)
            base_channels=_int("BASE_CHANNELS", 64),
            erosion=os.getenv("EROSION", "cross"),
            request_timeout=_float("LS_TIMEOUT", 30),
            dashboard_enabled=os.getenv("DASHBOARD_ENABLED", "1").strip().lower() in {"1", "true", "on", "yes"},
        )


def load_stages(path: str) -> dict:
    """stages.json: {stage: {"projects": [ids], "holdout_npz": file, "anchors_npz": file, ...}}.
    Fails fast on a project mapped to two stages or a missing npz (relative paths are next to the file)."""
    import json
    from pathlib import Path

    p = Path(path)
    stages = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(stages, dict) or not stages:
        raise ValueError(f"{path}: expected a non-empty {{stage: config}} object")
    seen: dict[int, str] = {}
    for stage, cfg in stages.items():
        projects = cfg.get("projects")
        if not isinstance(projects, list) or not all(isinstance(x, int) for x in projects):
            raise ValueError(f"{path}: stage {stage!r} needs an integer 'projects' list")
        for pid in projects:
            if pid in seen:
                raise ValueError(f"{path}: project {pid} is in both {seen[pid]!r} and {stage!r}")
            seen[pid] = stage
        for key in ("holdout_npz", "anchors_npz"):
            if cfg.get(key):
                f = Path(cfg[key]) if Path(cfg[key]).is_absolute() else p.parent / cfg[key]
                if not f.exists():
                    raise ValueError(f"{path}: {key} {f} of stage {stage!r} does not exist")
                cfg[key] = str(f)
    return stages


def stage_of(stages: dict, project_id) -> str | None:
    """LS sends the project as "<id>.<created_ts>" or an int."""
    try:
        pid = int(str(project_id).split(".")[0])
    except ValueError:
        return None
    return next((s for s, cfg in stages.items() if pid in cfg["projects"]), None)
