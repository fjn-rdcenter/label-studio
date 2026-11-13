import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from tasdf_online.config import Settings  # noqa: E402
from tasdf_online.ls_format import BC, ICM, TE, ZP  # noqa: E402


def blastocyst_mask(size: int = 256) -> np.ndarray:
    """Synthetic class map: ZP ring > TE ring > BC disk with an ICM blob."""
    m = np.zeros((size, size), np.uint8)
    c = (size // 2, size // 2)
    cv2.circle(m, c, int(size * 0.40), ZP, -1)
    cv2.circle(m, c, int(size * 0.33), TE, -1)
    cv2.circle(m, c, int(size * 0.27), BC, -1)
    cv2.ellipse(m, (c[0] + int(size * 0.12), c[1]), (int(size * 0.08), int(size * 0.12)), 0, 0, 360, ICM, -1)
    return m


@pytest.fixture
def settings(tmp_path):
    rng = np.random.default_rng(0)
    masks = np.stack([blastocyst_mask() for _ in range(2)])
    images = (rng.random((2, 256, 256)) * 255).astype(np.uint8)
    np.savez(tmp_path / "holdout.npz", images=images, masks=masks)
    (tmp_path / "stages.json").write_text(json.dumps({"blastocyst": {"projects": [7], "holdout_npz": "holdout.npz"}}))
    return dataclasses.replace(
        Settings.from_env(), device="cpu", base_channels=8, state_dir=str(tmp_path / "state"), baseline_ckpt="",
        registry_dir=str(tmp_path / "registry"), stages_file=str(tmp_path / "stages.json"),
        holdout_npz="", anchor_npz="", ls_url="", ls_token="", ol_token="secret", min_quality_level=2,
        train_epochs=2, train_min_new=1, min_free_ram_gb=0, min_free_vram_gb=0, amp="0",
    )
