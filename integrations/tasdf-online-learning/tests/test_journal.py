import json

import numpy as np

from tasdf_online import journal
from tasdf_online.config import Settings


def test_append_writes_one_json_line_each(tmp_path):
    journal.append(str(tmp_path), 38, 10911, {"event": "gold"})
    journal.append(str(tmp_path), 38, 10911, {"event": "propose", "distance": 1})
    lines = (tmp_path / "38" / "10911.jsonl").read_text().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["gold", "propose"] and "at" in json.loads(lines[0])


def test_save_arrays_is_loadable_with_url_names(tmp_path):
    p = journal.save_arrays(str(tmp_path), 38, 1, "/data/local-files/?d=a/b.jpg", gold=np.ones((4, 4), np.uint8))
    assert np.load(p)["gold"].sum() == 16 and "?" not in p


def test_settings_defaults():
    s = Settings.from_env()
    assert (s.crop_size, s.band_px, s.adapt_steps, s.adapt_lr, s.adapt_max_wells) == (600, 8, 3, 1e-4, 16)
