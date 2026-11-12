import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_npz  # noqa: E402


def _write(root, split, name, mask):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / split / "images" / f"{name}.png"), np.full((64, 64), 128, np.uint8))
    cv2.imwrite(str(root / split / "masks" / f"{name}.png"), mask)


def test_export_resizes_and_keeps_all_classes(tmp_path):
    m = np.zeros((64, 64), np.uint8)
    for c in range(1, 5):
        m[c * 10:c * 10 + 8, :] = c
    _write(tmp_path, "valid", "a", m)
    out = tmp_path / "h.npz"
    export_npz.main(["--root", str(tmp_path), "--split", "valid", "--out", str(out)])
    d = np.load(out)
    assert d["images"].shape == (1, 256, 256) and set(np.unique(d["masks"])) == {0, 1, 2, 3, 4}


def test_export_refuses_masks_without_bc(tmp_path):
    m = np.zeros((64, 64), np.uint8)
    m[10:20] = 1; m[20:30] = 2; m[30:40] = 3  # the old broken holdout: no class 4
    _write(tmp_path, "valid", "a", m)
    with pytest.raises(SystemExit, match="exactly classes"):
        export_npz.main(["--root", str(tmp_path), "--split", "valid", "--out", str(tmp_path / "h.npz")])
