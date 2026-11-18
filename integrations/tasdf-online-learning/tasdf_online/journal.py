"""Append-only record of every confirmed image and proposal, so a labeling session can be replayed offline."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def append(root: str, project: int, task: int, record: dict) -> None:
    path = Path(root) / str(project) / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.time(), **record}, ensure_ascii=False) + "\n")


def save_arrays(root: str, project: int, task: int, name: str, **arrays: np.ndarray) -> str:
    folder = Path(root) / str(project) / str(task)
    folder.mkdir(parents=True, exist_ok=True)
    # image URLs carry "/" and "?": hash them into a safe file name
    path = folder / f"{hashlib.sha1(name.encode()).hexdigest()[:12]}-{int(time.time() * 1000)}.npz"
    np.savez_compressed(path, **arrays)
    return str(path)
