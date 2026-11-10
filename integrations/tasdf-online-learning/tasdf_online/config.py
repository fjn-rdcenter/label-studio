import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


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
    ls_url: str
    ls_token: str
    ls_project_id: int
    ol_token: str
    min_quality_level: int
    refresh_next_n: int
    episode_epochs: int
    replay_size: int
    holdout_limit: int
    lr: float
    ema_alpha: float
    corr_beta: float
    topo_sup_weight: float
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
            state_dir=os.getenv("STATE_DIR", "/data/tasdf-online"),
            ls_url=os.getenv("LS_URL", "").rstrip("/"),
            ls_token=os.getenv("LS_TOKEN", ""),
            ls_project_id=_int("LS_PROJECT_ID", 0),
            ol_token=os.getenv("OL_TOKEN", ""),
            min_quality_level=_int("MIN_QUALITY_LEVEL", 2),
            refresh_next_n=_int("REFRESH_NEXT_N", 0),
            episode_epochs=_int("EPISODE_EPOCHS", 3),
            replay_size=_int("REPLAY_SIZE", 4),
            holdout_limit=_int("HOLDOUT_LIMIT", 0),
            lr=_float("LR", 1e-4),
            ema_alpha=_float("EMA_ALPHA", 0.65),
            corr_beta=_float("CORR_BETA", 0.5),
            topo_sup_weight=_float("TOPO_SUP_WEIGHT", 0.1),
            # must match the baseline checkpoint (M2-B: 64 / cross)
            base_channels=_int("BASE_CHANNELS", 64),
            erosion=os.getenv("EROSION", "cross"),
            request_timeout=_float("LS_TIMEOUT", 30),
            dashboard_enabled=os.getenv("DASHBOARD_ENABLED", "1").strip().lower() in {"1", "true", "on", "yes"},
        )
