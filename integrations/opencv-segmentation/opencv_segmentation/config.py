import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    label_studio_url: str
    label_studio_api_key: str
    allowed_image_hosts: frozenset[str]
    request_timeout: float
    max_image_bytes: int
    max_image_pixels: int
    max_tasks: int
    max_request_bytes: int
    default_ring: bool
    ring_width: int

    @classmethod
    def from_env(cls) -> "Settings":
        label_studio_url = os.getenv("LABEL_STUDIO_URL", "").rstrip("/")
        hosts = {
            host.strip().lower().rstrip(".")
            for host in os.getenv("ALLOWED_IMAGE_HOSTS", "").split(",")
            if host.strip()
        }
        ls_host = urlparse(label_studio_url).hostname
        if ls_host:
            hosts.add(ls_host.lower().rstrip("."))
        return cls(
            label_studio_url=label_studio_url,
            label_studio_api_key=os.getenv("LABEL_STUDIO_API_KEY", ""),
            allowed_image_hosts=frozenset(hosts),
            request_timeout=float(os.getenv("IMAGE_FETCH_TIMEOUT", "10")),
            max_image_bytes=_positive_int("MAX_IMAGE_BYTES", 20 * 1024 * 1024),
            max_image_pixels=_positive_int("MAX_IMAGE_PIXELS", 40_000_000),
            max_tasks=_positive_int("MAX_TASKS_PER_REQUEST", 16),
            max_request_bytes=_positive_int("MAX_REQUEST_BYTES", 2 * 1024 * 1024),
            default_ring=_boolean("DEFAULT_RING"),
            ring_width=_positive_int("DEFAULT_RING_WIDTH", 3),
        )
