# TASDF Well Adapter + Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bác sĩ bấm "Xong ảnh" (Shift+Enter) trên một ảnh timelapse; adapter của well học từ ảnh đó dưới 1 giây và đề xuất đã nắn biên được lan sang N frame kế tiếp, không bao giờ có vùng rác hay ghi đè ảnh bác sĩ đã gán.

**Architecture:** Ba module thuần mới trong `tasdf_online` (crop, luật lan truyền, adapter) cộng một nhật ký; `OnlineService.propagate` ghép chúng sau một nhánh `context.action == "propagate"` của `/predict`. Label Studio chỉ chuyển `context` qua `predict/test` và trả lại `propagation`. Frontend thêm nút "Xong ảnh" vào DataManager, dùng lại đường gọi của Smart Prediction và các helper thuần có test.

**Tech Stack:** Python 3 / PyTorch / OpenCV / Flask (TASDF backend), Django REST (Label Studio), React + MobX-State-Tree (editor), pytest, jest.

**Spec:** `docs/superpowers/specs/2026-11-19-tasdf-well-adapter-design.md`

## Global Constraints

- Không thêm dependency mới (Python hay npm).
- Model nhận ảnh 256×256; crop từ ảnh gốc là **600×600**; `BAND_PX=8` (ở 256); dung sai biên 2 px; ngưỡng diện tích [0.67, 1.5].
- Adapter = các khoá `bundle_registry.is_adapter_key`; thống kê chạy BN không đổi (model ở `eval()` khi học).
- Học: `ADAPT_STEPS=3`, `ADAPT_LR=1e-4`, bộ nhớ 32 mẫu/well, trọng số loss 2 (pixel đã sửa) / 0.1 (giữ dự đoán) / 1e-3 (L2), rollback khi Dice trung bình giảm > 0.01.
- Cổng: `production` cho tới khi có 3 ảnh được chấm; sau đó nguồn có Dice trung bình cao nhất trên 5 ảnh chấm gần nhất trong {copy, production, adapter}.
- Kho well: tối đa 16 well, bỏ well không dùng quá 7200 s, xoá hết khi production đổi.
- UI: nút "Xong ảnh", phím Shift+Enter, mặc định N=3 (0–10), ngưỡng 0.8 (0–1), tự lan bật.
- Máy là laptop dễ tự tắt khi chạy nặng: mọi lệnh pytest có torch chạy qua **bản sao** `integrations/tasdf-online-learning/scripts/safe_run.sh` đặt ngoài repo (gọi là `$SAFE`), với `CUDA_VISIBLE_DEVICES=`; mỗi lúc chỉ một tiến trình torch. Build image Docker cũng qua `$SAFE`, khi CPU < 60°C.
- Test backend dùng model nhỏ `base_channels=8` trên CPU (fixture `settings` trong `tests/conftest.py`).
- Làm trên worktree tách từ nhánh `ML-backend-design`. Commit theo Conventional Commits (`feat(tasdf): …`, `feat(editor): …`), tác giả `ldhv-04`, cuối message có `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Không push. Không commit `docs/superpowers/specs/` hay `docs/superpowers/plans/`.
- Comment trong code backend bằng tiếng Anh như code hiện có; chỉ ghi lý do mà code không tự nói.

## Review Focus

1. Bác sĩ bấm "Xong ảnh" trên ảnh chưa có vùng nào → không học, không lan, báo rõ; backend trả kết quả rỗng có `stop="no_objects"` thay vì lỗi. (Task 5, Task 7)
2. Frame kế tiếp đã có vùng của bác sĩ (`manual` hoặc `prediction-changed`) → không gọi model, không ghi đè, dừng chuỗi. (Task 7)
3. Lan tới frame cuối của well → dừng êm, không gọi request thừa. (Task 7)
4. Backend khởi động lại hoặc well bị dọn giữa hai lần bấm → không lỗi; adapter khởi tạo lại từ production; lần đó không chấm điểm. (Task 5)
5. Ảnh nguồn có vùng nhãn model không biết (PN, rect, ellipse, keypoint) → được mang sang nguyên vẹn, không bị model xoá hay đổi. (Task 5)

---

### Task 1: Crop 600×600 và ảnh gốc

**Files:**
- Create: `integrations/tasdf-online-learning/tasdf_online/crop.py`
- Modify: `integrations/tasdf-online-learning/tasdf_online/ls_format.py:222-240` (`fetch_image`)
- Test: `integrations/tasdf-online-learning/tests/test_crop.py`

**Interfaces:**
- Produces:
  - `ls_format.fetch_gray(ls_url: str, token: str, source: str, timeout: float = 30) -> np.ndarray` — uint8 H×W, cùng kiểm tra an toàn như hiện tại; `fetch_image` gọi lại nó (hành vi không đổi).
  - `crop.Box = tuple[int, int, int, int]` (x0, y0, x1, y1, pixel ảnh gốc).
  - `crop.crop_box(results: list[dict], w: int, h: int, size: int = 600) -> Box` — tâm là tâm hộp bao mọi điểm `value.points` của các vùng; kẹp trong ảnh; ảnh nhỏ hơn `size` thì lấy cả chiều đó; không có vùng nào thì lấy giữa ảnh.
  - `crop.to_box(results, w, h, box) -> list[dict]` và `crop.from_box(results, w, h, box) -> list[dict]` — đổi `value.points` và `value.holes` giữa % ảnh gốc và % hộp; vùng không có `points` trả nguyên (cùng object); `from_box` đặt `original_width=w`, `original_height=h`. Không sửa dict đầu vào.
  - `crop.crop_input(gray: np.ndarray, box: Box) -> np.ndarray` — `ls_format.preprocess(gray[y0:y1, x0:x1])`.

- [ ] **Step 1: Viết test hỏng**

```python
from tasdf_online.crop import crop_box, crop_input, from_box, to_box

def poly(cx, cy, r=2.0, label="ZP"):
    pts = [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]
    return {"id": "a", "type": "polygonlabels", "from_name": "pl-0", "to_name": "timelapse_0",
            "value": {"points": pts, "polygonlabels": [label]}}

def test_crop_box_centres_on_regions_and_clamps():
    assert crop_box([poly(90, 90)], 964, 1288) == (364, 688, 964, 1288)

def test_crop_box_small_image_and_no_regions():
    assert crop_box([poly(50, 50)], 400, 300) == (0, 0, 400, 300)
    assert crop_box([], 964, 1288) == (182, 344, 782, 944)

def test_box_roundtrip_and_passthrough():
    rect = {"id": "r", "type": "rectanglelabels", "value": {"x": 1, "y": 2, "width": 3, "height": 4}}
    ring = poly(50, 50, 10)
    ring["value"]["holes"] = [[[48, 48], [52, 48], [52, 52]]]
    box = crop_box([ring], 964, 1288)
    back = from_box(to_box([ring, rect], 964, 1288, box), 964, 1288, box)
    assert back[1] is rect
    for p, q in zip(back[0]["value"]["points"] + back[0]["value"]["holes"][0],
                    ring["value"]["points"] + ring["value"]["holes"][0]):
        assert abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6
    assert back[0]["original_width"] == 964 and ring["value"]["points"][0] == [40.0, 40.0]

def test_crop_input_is_256_float():
    import numpy as np
    x = crop_input(np.full((1288, 964), 255, np.uint8), (364, 688, 964, 1288))
    assert x.shape == (256, 256) and x.dtype == np.float32 and float(x.max()) == 1.0
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `cd integrations/tasdf-online-learning && CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_crop.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tasdf_online.crop'`

- [ ] **Step 3: Viết `fetch_gray` (tách phần tải/giải mã khỏi `fetch_image`) và `crop.py` theo Interfaces**

- [ ] **Step 4: Chạy test, phải qua; chạy lại `tests/test_ls_format.py` để chắc `fetch_image` không đổi**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_crop.py tests/test_ls_format.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tasdf_online/crop.py tasdf_online/ls_format.py tests/test_crop.py
git commit -m "feat(tasdf): crop a 600x600 window around the drawn regions"
```

---

### Task 2: Bốn luật lan truyền (hàm thuần)

**Files:**
- Create: `integrations/tasdf-online-learning/tasdf_online/propagate.py`
- Test: `integrations/tasdf-online-learning/tests/test_propagate.py`

**Interfaces:**
- Consumes: `ls_format.results_to_mask`, `ls_format.mask_to_results`, `ls_format.NAME_TO_CLASS`, `ZP`, `TE`.
- Produces:
  - `RING_CLASSES = {ZP, TE}`
  - `object_class(result: dict) -> int | None` — lớp TASDF nếu `type == "polygonlabels"` và nhãn có trong `NAME_TO_CLASS`, ngược lại `None`.
  - `object_mask(result: dict, size: int = 256) -> np.ndarray` — bool, chỉ vùng này.
  - `refine_object(carried: np.ndarray, model: np.ndarray, band_px: int, ring: bool) -> np.ndarray` — bool.
  - `boundary_f(a: np.ndarray, b: np.ndarray, tol_px: int = 2) -> float` — 1.0 khi cả hai rỗng, 0.0 khi chỉ một bên rỗng.
  - `stop_reason(carried, refined, free, ring: bool, threshold: float) -> str | None` — trả theo thứ tự ưu tiên `"vanished"`, `"area"`, `"ring"`, `"disagree"`, hoặc `None`.
  - `object_result(mask: np.ndarray, template: dict, cls: int) -> dict | None` — một result polygon (vòng có một lỗ nếu là ring) với `id`, `from_name`, `to_name`, `type`, nhãn lấy từ `template`; toạ độ % theo kích thước `mask`; `None` khi mask rỗng.

`refine_object` (thuật toán spec chốt):
```
core = erode(carried, band_px); zone = dilate(carried, band_px)
out = core | (model & zone)
keep the connected component of out with the largest overlap with carried (drop the rest)
fill every hole; if ring: re-open only the largest hole that out had before filling
```
`stop_reason`: `vanished` nếu `refined` rỗng; `area` nếu `refined.sum()/carried.sum()` ngoài [0.67, 1.5]; `ring` nếu `ring` và `refined` không có lỗ; `disagree` nếu `boundary_f(carried, free) < threshold`, trong đó `free` là thành phần của dự đoán tự do chạm `carried`.

- [ ] **Step 1: Viết test hỏng**

```python
import cv2, numpy as np
from tasdf_online.propagate import boundary_f, object_result, refine_object, stop_reason

def disk(r, c=(128, 128)):
    m = np.zeros((256, 256), np.uint8); cv2.circle(m, c, r, 1, -1); return m.astype(bool)

def holes(m):
    _, h = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return 0 if h is None else int((h[0][:, 3] != -1).sum())

def test_refine_never_leaves_the_band():
    out = refine_object(disk(60), np.ones((256, 256), bool), 8, ring=False)
    assert not (out & ~disk(68)).any()

def test_refine_keeps_core_when_model_sees_nothing():
    out = refine_object(disk(60), np.zeros((256, 256), bool), 8, ring=False)
    assert out.sum() > 0 and not (out & ~disk(60)).any()

def test_refine_drops_far_junk():
    junk = disk(60) | disk(10, (20, 20))
    assert not refine_object(disk(60), junk, 8, ring=False)[20, 20]

def test_ring_keeps_exactly_one_hole():
    ring = disk(80) & ~disk(60)
    model = ring.copy(); model[128, 128] = False
    assert holes(refine_object(ring, model, 8, ring=True)) == 1

def test_boundary_f():
    assert boundary_f(disk(60), disk(60)) == 1.0
    assert boundary_f(disk(60), disk(60, (138, 128))) < 0.5
    assert boundary_f(np.zeros((256, 256), bool), np.zeros((256, 256), bool)) == 1.0

def test_stop_reasons_in_priority_order():
    c = disk(60)
    assert stop_reason(c, np.zeros_like(c), c, False, 0.8) == "vanished"
    assert stop_reason(c, disk(90), c, False, 0.8) == "area"
    assert stop_reason(c, c, c, True, 0.8) == "ring"
    assert stop_reason(c, c, disk(60, (150, 128)), False, 0.8) == "disagree"
    assert stop_reason(c, c, c, False, 0.8) is None

def test_object_result_keeps_identity_and_ring():
    tpl = {"id": "zp1", "type": "polygonlabels", "from_name": "pl-3", "to_name": "timelapse_3",
           "value": {"points": [], "polygonlabels": ["ZP"], "ring": True}}
    r = object_result(disk(80) & ~disk(60), tpl, 1)
    assert r["id"] == "zp1" and r["from_name"] == "pl-3" and r["value"]["polygonlabels"] == ["ZP"]
    assert len(r["value"]["holes"]) == 1
    assert object_result(np.zeros((256, 256), bool), tpl, 1) is None
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_propagate.py -q`
Expected: FAIL — `No module named 'tasdf_online.propagate'`

- [ ] **Step 3: Viết `propagate.py` theo Interfaces; `object_result` dựng class map chỉ chứa vật thể này rồi gọi lại `mask_to_results`, lấy result có polygon lớn nhất**

- [ ] **Step 4: Chạy test, phải qua**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_propagate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tasdf_online/propagate.py tests/test_propagate.py
git commit -m "feat(tasdf): refine only the drawn objects inside a band and say when to stop"
```

---

### Task 3: Lõi adapter theo well

**Files:**
- Create: `integrations/tasdf-online-learning/tasdf_online/adapter.py`
- Test: `integrations/tasdf-online-learning/tests/test_adapter.py`

**Interfaces:**
- Consumes: `bundle_registry.is_adapter_key`, `trainer.augment`, `trainer.mean_dice_ignore`, `learner.predict_probs`, `learner._soft_dice_loss`, `ls_format.IGNORE`.
- Produces:
  - `@dataclass Sample: x: torch.Tensor  # (1,256,256) float; y: torch.Tensor  # (256,256) int64, IGNORE=255; changed: torch.Tensor  # (256,256) bool; prior: torch.Tensor  # (256,256) int64, argmax production`
  - `@dataclass WellState: adapter: dict[str, torch.Tensor]; samples: list[Sample]; proposals: dict[str, dict[str, np.ndarray]]  # url -> {"copy"|"production"|"adapter": uint8 class map 256}; scores: list[dict[str, float]]; last_used: float`
  - `class WellAdapters(max_wells: int = 16, ttl_s: float = 7200.0)` với `get(key: tuple[int, int], init: Callable[[], dict[str, torch.Tensor]]) -> WellState` (dọn well quá hạn và well dùng lâu nhất khi vượt `max_wells`), `clear() -> None`, `__len__`.
  - `adapter_loss(model: nn.Module, prod_adapter: dict, batch: list[Sample], w_changed: float = 2.0, w_keep: float = 0.1, w_l2: float = 1e-3) -> torch.Tensor` — CE (`reduction="none"`, trọng số `1 + w_changed*changed`, bỏ IGNORE) + soft-Dice trên cả hai nhánh (`branch="both"`) + `w_keep` × CE tới `prior` chỉ trên pixel IGNORE + `w_l2` × tổng bình phương (adapter − prod_adapter).
  - `adapt(model: nn.Module, state: WellState, prod_adapter: dict, steps: int = 3, lr: float = 1e-4, max_samples: int = 32, rollback_margin: float = 0.01, seed: int = 0) -> dict` — `{"loss": float, "seconds": float, "reverted": bool, "samples": int}`; nạp `state.adapter` (`strict=False`), chỉ bật grad cho khoá adapter, `model.eval()` suốt quá trình, AdamW `steps` bước trên `samples[-max_samples:]` có `augment`; so Dice trung bình (`mean_dice_ignore` trên argmax ensemble) trước/sau trên cùng mẫu (không augment), giảm quá `rollback_margin` thì trả adapter cũ; cuối cùng `state.adapter` là bản clone detach.
  - `choose_source(scores: list[dict[str, float]], min_scored: int = 3, window: int = 5) -> str` — `"production"` khi chưa đủ `min_scored`; ngược lại khoá có trung bình cao nhất trên `scores[-window:]`, hoà thì ưu tiên `production` > `adapter` > `copy`.

- [ ] **Step 1: Viết test hỏng**

```python
import time, torch
from conftest import blastocyst_mask
from tasdf_online import learner
from tasdf_online.adapter import Sample, WellAdapters, WellState, adapt, adapter_loss, choose_source
from tasdf_online.bundle_registry import adapter_state, is_adapter_key

def setup():
    torch.manual_seed(0)
    m = learner.build_model(8).eval()
    prod = {k: v.clone() for k, v in adapter_state(m.state_dict()).items()}
    y = torch.from_numpy(blastocyst_mask().astype("int64"))
    s = Sample(torch.rand(1, 256, 256), y, torch.zeros(256, 256, dtype=torch.bool), y.clone())
    return m, prod, WellState(dict(prod), [s], {}, [], time.time())

def test_adapt_changes_only_adapter_keys_and_freezes_bn_stats():
    m, prod, st = setup()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    out = adapt(m, st, prod, steps=3, rollback_margin=1.0)
    assert not out["reverted"] and out["samples"] == 1
    for k, v in m.state_dict().items():
        if "running_" in k or "num_batches" in k or not is_adapter_key(k):
            assert torch.equal(v, before[k]), k
    assert any(not torch.equal(st.adapter[k], prod[k]) for k in prod)

def test_loss_goes_down_on_the_well():
    m, prod, st = setup()
    l0 = float(adapter_loss(m, prod, st.samples))
    adapt(m, st, prod, steps=10, rollback_margin=1.0)
    m.load_state_dict(st.adapter, strict=False)
    assert float(adapter_loss(m, prod, st.samples)) < l0

def test_rollback_keeps_old_adapter(monkeypatch):
    import tasdf_online.adapter as A
    m, prod, st = setup()
    calls = iter([0.9, 0.1])
    monkeypatch.setattr(A, "_mean_dice", lambda *a, **k: next(calls))
    out = adapt(m, st, prod, steps=1)
    assert out["reverted"] and all(torch.equal(st.adapter[k], prod[k]) for k in prod)

def test_well_store_evicts_lru_and_idle():
    w = WellAdapters(max_wells=2, ttl_s=3600)
    for key in [(1, 1), (1, 2), (1, 3)]:
        w.get(key, dict)
    assert len(w) == 2
    w2 = WellAdapters(max_wells=8, ttl_s=0)
    w2.get((1, 1), dict); time.sleep(0.01); w2.get((1, 2), dict)
    assert len(w2) == 1

def test_choose_source():
    assert choose_source([{"copy": 1, "production": 0, "adapter": 0}] * 2) == "production"
    s = [{"copy": .5, "production": .6, "adapter": .8}] * 3
    assert choose_source(s) == "adapter"
    assert choose_source([{"copy": .9, "production": .1, "adapter": .1}] * 5 + [{"copy": 0, "production": 1, "adapter": 0}] * 5) == "production"
```

(`_mean_dice(model, samples) -> float` là helper riêng trong `adapter.py` mà `adapt` gọi trước và sau khi học; test thay nó để ép rollback.)

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_adapter.py -q`
Expected: FAIL — `No module named 'tasdf_online.adapter'`

- [ ] **Step 3: Viết `adapter.py` theo Interfaces**

- [ ] **Step 4: Chạy test, phải qua**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_adapter.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tasdf_online/adapter.py tests/test_adapter.py
git commit -m "feat(tasdf): per-well adapter that learns BN affine and heads in a few steps"
```

---

### Task 4: Cấu hình và nhật ký lan truyền

**Files:**
- Modify: `integrations/tasdf-online-learning/tasdf_online/config.py:34-112` (`Settings`, `from_env`)
- Create: `integrations/tasdf-online-learning/tasdf_online/journal.py`
- Test: `integrations/tasdf-online-learning/tests/test_journal.py`

**Interfaces:**
- Produces:
  - Trường mới của `Settings` (có mặc định, đặt sau `dashboard_enabled` để mọi chỗ dựng `Settings` hiện có vẫn chạy): `journal_dir: str = "/data/tasdf-online/propagation"` (`JOURNAL_DIR`), `crop_size: int = 600` (`ADAPT_CROP`), `band_px: int = 8` (`ADAPT_BAND_PX`), `adapt_steps: int = 3` (`ADAPT_STEPS`), `adapt_lr: float = 1e-4` (`ADAPT_LR`), `adapt_max_wells: int = 16` (`ADAPT_MAX_WELLS`).
  - `journal.append(root: str, project: int, task: int, record: dict) -> None` — một dòng JSON vào `<root>/<project>/<task>.jsonl`, tự thêm `"at"` (epoch giây), tạo thư mục nếu thiếu.
  - `journal.save_arrays(root: str, project: int, task: int, name: str, **arrays: np.ndarray) -> str` — `np.savez_compressed` vào `<root>/<project>/<task>/<sha1(name)[:12]>-<epoch_ms>.npz`, trả đường dẫn.

- [ ] **Step 1: Viết test hỏng**

```python
import json, numpy as np
from tasdf_online import journal
from tasdf_online.config import Settings

def test_append_writes_one_json_line_each(tmp_path):
    journal.append(str(tmp_path), 38, 10911, {"event": "gold"})
    journal.append(str(tmp_path), 38, 10911, {"event": "propose", "distance": 1})
    lines = (tmp_path / "38" / "10911.jsonl").read_text().splitlines()
    assert [json.loads(l)["event"] for l in lines] == ["gold", "propose"] and "at" in json.loads(lines[0])

def test_save_arrays_is_loadable_with_url_names(tmp_path):
    p = journal.save_arrays(str(tmp_path), 38, 1, "/data/local-files/?d=a/b.jpg", gold=np.ones((4, 4), np.uint8))
    assert np.load(p)["gold"].sum() == 16 and "?" not in p

def test_settings_defaults():
    s = Settings.from_env()
    assert (s.crop_size, s.band_px, s.adapt_steps, s.adapt_lr, s.adapt_max_wells) == (600, 8, 3, 1e-4, 16)
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_journal.py -q`
Expected: FAIL — `No module named 'tasdf_online.journal'`

- [ ] **Step 3: Thêm trường `Settings` + đọc env trong `from_env`; viết `journal.py`**

- [ ] **Step 4: Chạy test, phải qua; chạy cả `tests/test_api.py` để chắc `Settings` cũ không vỡ**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_journal.py tests/test_api.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tasdf_online/config.py tasdf_online/journal.py tests/test_journal.py
git commit -m "feat(tasdf): settings and a replayable journal for propagation"
```

---

### Task 5: Ghép vào service: `/predict` với `context.action == "propagate"`

**Files:**
- Modify: `integrations/tasdf-online-learning/tasdf_online/app.py` (`OnlineService.__init__`, `_serve`, thêm `propagate`/`_learn_gold`; route `predict` dòng ~628)
- Modify: `integrations/tasdf-online-learning/tasdf_online/trainer.py:118` (`train_epochs` thêm `pause`)
- Test: `integrations/tasdf-online-learning/tests/test_propagate_api.py`

**Interfaces:**
- Consumes: Task 1 (`fetch_gray`, `crop_box`, `to_box`, `from_box`, `crop_input`), Task 2 (tất cả), Task 3 (`WellAdapters`, `Sample`, `adapt`, `choose_source`), Task 4 (`Settings` mới, `journal`).
- Produces:
  - `trainer.train_epochs(..., pause: threading.Event | None = None)` — trước mỗi `opt.step` chờ (ngủ 0.05 s) khi `pause.is_set()`; `run_job` truyền `self.interactive`.
  - Thuộc tính `OnlineService`: `interactive: threading.Event`, `wells: WellAdapters` (max từ `adapt_max_wells`), `adapt_model: nn.Module` (bản model riêng của production, dựng trong `_serve`), `adapt_lock: threading.Lock`. `_serve` gọi `self.wells.clear()`.
  - `OnlineService.propagate(task: dict, label: tuple, stage: str, ctx: dict) -> dict` → `{"result": list[dict], "score": float, "model_version": str, "propagation": {"stop": str | None, "source": str, "distance": int, "adapt": dict | None, "scores": dict | None}}`.
  - Context nhận vào: `{"action": "propagate", "carried": list[dict], "distance": int ≥ 0, "threshold": float trong [0,1], "gold"?: {"url": str, "results": list[dict]}}`; sai kiểu → `ValueError` (route trả 400 qua `errorhandler` sẵn có). Nhánh này yêu cầu đúng một task.

Thứ tự trong `propagate` (khoá well = `(project, task["id"])`; ảnh đích = `task["data"][label[2]]`):
1. Có `gold` và trong đó có vùng: crop quanh vùng vàng, `y = annotation_target(to_box(gold.results))`; nếu well có `proposals[gold.url]` thì chấm `mean_dice_ignore` cho cả ba nguồn, thêm vào `scores`, `changed = proposals[nguồn đang dùng] != y` (bỏ IGNORE); thêm `Sample` (prior = argmax production trên crop đó); nếu gold có ít nhất một vật thể `object_class` khác `None` thì `adapt(...)` dưới `adapt_lock` với `interactive` bật; ghi `journal` (`"event": "gold"`) và `save_arrays` (box, y, ba đề xuất nếu có).
2. Không có vùng nào trong `carried` → `{"result": [], "propagation": {"stop": "no_objects", ...}}`.
3. Crop đích quanh `carried`; chạy production (`teacher`, có `_use_stage`) và `adapt_model` với `state.adapter` để lấy xác suất tự do; với từng vùng: không biết lớp → giữ nguyên; biết lớp → `refine_object` theo nguồn (copy = giữ mask mang sang), `stop_reason` với `threshold`, `object_result`, `from_box`. Lưu ba đề xuất (class map 256) vào `proposals[url đích]`.
4. Nguồn = `choose_source(state.scores)`. Có `stop` ở bất kỳ vật thể nào → trả `carried` nguyên cho ảnh này kèm lý do đầu tiên. Ghi `journal` (`"event": "propose"`).

- [ ] **Step 1: Viết test hỏng**

```python
import numpy as np, pytest, threading, time
import tasdf_online.app as app_module
from tasdf_online.bundle_registry import BASE

ZP_RING = {"id": "zp1", "type": "polygonlabels", "from_name": "pl-0", "to_name": "timelapse_0",
           "value": {"points": [[30, 30], [70, 30], [70, 70], [30, 70]], "holes": [[[40, 40], [60, 40], [60, 60], [40, 60]]],
                     "ring": True, "polygonlabels": ["ZP"]}}
PN_RECT = {"id": "pn1", "type": "rectanglelabels", "from_name": "rect-0", "to_name": "timelapse_0",
           "value": {"x": 45, "y": 45, "width": 5, "height": 5, "rectanglelabels": ["PN"]}}
CONFIG = ('<View><Image name="timelapse_0" value="$image"/><PolygonLabels name="pl-0" toName="timelapse_0">'
          '<Label value="ZP"/><Label value="TE"/></PolygonLabels></View>')

@pytest.fixture
def client(settings, monkeypatch, tmp_path):
    import dataclasses
    s = dataclasses.replace(settings, journal_dir=str(tmp_path / "journal"))
    monkeypatch.setattr(app_module, "fetch_gray", lambda *a, **k: np.full((1288, 964), 128, np.uint8))
    svc = app_module.OnlineService(s)
    return app_module.create_app(s, svc).test_client(), svc, s

def call(c, url, ctx):
    task = {"id": 5, "data": {"image": url}}
    return c.post("/predict", json={"project": 7, "label_config": CONFIG, "tasks": [task],
                                    "context": {"action": "propagate", "threshold": 0.0, **ctx}})

def test_unknown_labels_copied_and_no_new_regions(client):
    c, _, _ = client
    body = call(c, "/data/b.jpg", {"carried": [ZP_RING, PN_RECT], "distance": 1}).get_json()
    res = body["results"][0]["result"]
    assert {r["id"] for r in res} <= {"zp1", "pn1"} and PN_RECT in res

def test_gold_scores_three_sources_after_a_proposal(client):
    c, svc, _ = client
    first = call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1,
                                     "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}}).get_json()
    assert first["results"][0]["propagation"]["scores"] is None
    assert first["results"][0]["propagation"]["adapt"]["samples"] == 1
    second = call(c, "/data/c.jpg", {"carried": [ZP_RING], "distance": 1,
                                      "gold": {"url": "/data/b.jpg", "results": [ZP_RING]}}).get_json()
    assert set(second["results"][0]["propagation"]["scores"]) == {"copy", "production", "adapter"}

def test_empty_source_stops_without_learning(client):
    c, _, _ = client
    p = call(c, "/data/b.jpg", {"carried": [], "distance": 1, "gold": {"url": "/data/a.jpg", "results": []}}).get_json()
    prop = p["results"][0]["propagation"]
    assert p["results"][0]["result"] == [] and prop["stop"] == "no_objects" and prop["adapt"] is None

def test_vanished_object_keeps_carried_copy(client, monkeypatch):
    import torch
    c, _, _ = client
    bg = torch.zeros(1, 5, 256, 256); bg[:, 0] = 1
    monkeypatch.setattr(app_module.learner, "predict_probs", lambda m, x, batch=8: (bg.repeat(len(x), 1, 1, 1), None))
    body = call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1}).get_json()["results"][0]
    assert body["propagation"]["stop"] == "vanished" and body["result"][0]["id"] == "zp1"

def test_restart_or_eviction_does_not_crash(client):
    c, svc, _ = client
    svc.wells.clear()
    r = call(c, "/data/c.jpg", {"carried": [ZP_RING], "distance": 2, "gold": {"url": "/data/b.jpg", "results": [ZP_RING]}})
    assert r.status_code == 200 and r.get_json()["results"][0]["propagation"]["scores"] is None

def test_new_production_resets_wells(client):
    c, svc, _ = client
    call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1, "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}})
    assert len(svc.wells) == 1
    svc._serve(BASE)
    assert len(svc.wells) == 0

def test_bad_context_is_400(client):
    c, _, _ = client
    assert call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": "x"}).status_code == 400
    two = c.post("/predict", json={"project": 7, "label_config": CONFIG,
                                   "tasks": [{"id": 1, "data": {"image": "/data/a.jpg"}}] * 2,
                                   "context": {"action": "propagate", "carried": [], "distance": 1, "threshold": 0.8}})
    assert two.status_code == 400

def test_journal_written(client):
    c, _, s = client
    call(c, "/data/b.jpg", {"carried": [ZP_RING], "distance": 1, "gold": {"url": "/data/a.jpg", "results": [ZP_RING]}})
    lines = (__import__("pathlib").Path(s.journal_dir) / "7" / "5.jsonl").read_text().splitlines()
    assert any('"gold"' in l for l in lines) and any('"propose"' in l for l in lines)

def test_long_term_training_waits_while_adapter_runs():
    import torch
    from tasdf_online import learner, trainer
    pause = threading.Event(); pause.set(); done = []
    m = learner.build_model(8)
    x = torch.rand(2, 1, 256, 256); y = torch.zeros(2, 256, 256, dtype=torch.int64)
    t = threading.Thread(target=lambda: done.append(trainer.train_epochs(m, x, y, None, mode="light", epochs=1, lr=1e-4, amp=False, pause=pause)))
    t.start(); time.sleep(0.3)
    assert not done
    pause.clear(); t.join(timeout=60)
    assert done
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests/test_propagate_api.py -q`
Expected: FAIL — các test về `context` trả 200 với kết quả `predict_task` thường hoặc `AttributeError: 'OnlineService' object has no attribute 'wells'`

- [ ] **Step 3: Thêm `pause` vào `train_epochs` và truyền từ `run_job`**

- [ ] **Step 4: Thêm thuộc tính, `_serve`, `propagate`, `_learn_gold` vào `OnlineService` và nhánh context trong route `predict`, theo thứ tự ở trên**

- [ ] **Step 5: Chạy test mới và toàn bộ test cũ, phải qua**

Run: `CUDA_VISIBLE_DEVICES= $SAFE python -m pytest tests -q`
Expected: PASS (test CUDA tự bỏ qua)

- [ ] **Step 6: Commit**

```bash
git add tasdf_online/app.py tasdf_online/trainer.py tests/test_propagate_api.py
git commit -m "feat(tasdf): learn from the confirmed image and propagate refined objects to the next frames"
```

---

### Task 6: Label Studio chuyển `context` và trả `propagation`

**Files:**
- Modify: `label_studio/ml/api.py:309-312` (`MLBackendPredictTestAPI.post`)
- Modify: `label_studio/ml/models.py` (`predict_single_task`, `_get_predictions_from_ml_backend`)
- Test: `label_studio/tests/ml/test_api.py` (thêm cạnh `test_smart_prediction_*`, dùng `_timelapse_backend`)

**Interfaces:**
- Produces:
  - `MLBackend.predict_single_task(task, update_project_version=False, force=False, image=None, context=None) -> dict` — `context` phải là dict và chỉ đi kèm `image`, ngược lại `ValidationError({'context': ...})`; nếu có `context['gold']` thì `gold['url']` phải nằm trong `_strings(task.data)` (cùng kiểm tra như `image`); trả `{'data': ..., 'propagation': meta}` khi backend gửi `propagation`, giữ `{'data': ...}` khi không.
  - `_get_predictions_from_ml_backend(serialized_tasks, label_config=None, errors=None, context=None)` — chuyển `context` vào `self.api.make_predictions(..., context=context)`; bỏ khoá `propagation` khỏi từng prediction trước `PredictionSerializer` và đưa nó ra cho `predict_single_task`.
  - API đọc `request.data.get('context')`.

- [ ] **Step 1: Viết test hỏng**

```python
@pytest.mark.django_db
def test_smart_prediction_forwards_context_and_returns_propagation(business_client):
    backend, task = _timelapse_backend(business_client)
    url = '/data/local-files/?d=w/1/1.jpg'
    gold = {'url': '/data/local-files/?d=w/0/1.jpg', 'results': []}
    ctx = {'action': 'propagate', 'carried': [], 'distance': 1, 'threshold': 0.8, 'gold': gold}
    meta = {'stop': None, 'source': 'production', 'distance': 1, 'adapt': None, 'scores': None}
    with requests_mock.Mocker() as m:
        register_ml_backend_mock(m, url=backend.url,
                                 predictions={'results': [{'result': [], 'score': 0.5, 'propagation': meta}]})
        response = business_client.post(
            f'/api/ml/{backend.id}/predict/test',
            data=json.dumps({'task': task.id, 'image': {'url': url, 'to_name': 'timelapse_1', 'item_index': 0},
                             'context': ctx}),
            content_type='application/json',
        )
        sent = [r.json() for r in m.request_history if r.path == '/predict'][-1]
    assert response.status_code == 200, response.content
    assert sent['context'] == ctx
    assert response.json()['propagation'] == meta
    assert 'propagation' not in response.json()['data']


@pytest.mark.django_db
@pytest.mark.parametrize('body', [
    {'context': {'action': 'propagate', 'gold': {'url': 'http://evil.example/x.jpg', 'results': []}}},
    {'context': 'not-a-dict'},
    {'context': {'action': 'propagate'}, 'no_image': True},
])
def test_smart_prediction_rejects_bad_context(business_client, body):
    backend, task = _timelapse_backend(business_client)
    data = {'task': task.id, 'context': body['context']}
    if not body.get('no_image'):
        data['image'] = {'url': '/data/local-files/?d=w/1/1.jpg', 'to_name': 'timelapse_1', 'item_index': 0}
    with requests_mock.Mocker() as m:
        register_ml_backend_mock(m, url=backend.url)
        response = business_client.post(f'/api/ml/{backend.id}/predict/test', data=json.dumps(data),
                                         content_type='application/json')
        assert not [r for r in m.request_history if r.path == '/predict']
    assert response.status_code == 400
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `cd label_studio && python -m pytest tests/ml/test_api.py -k "smart_prediction" -q`
Expected: FAIL — `KeyError: 'context'` hoặc `KeyError: 'propagation'`; trường hợp context xấu trả 200 thay vì 400

- [ ] **Step 3: Sửa `api.py` và `models.py` theo Interfaces**

- [ ] **Step 4: Chạy test, phải qua (gồm cả hai test Smart Prediction cũ)**

Run: `cd label_studio && python -m pytest tests/ml/test_api.py -k "smart_prediction" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add label_studio/ml/api.py label_studio/ml/models.py label_studio/tests/ml/test_api.py
git commit -m "feat(ml): pass propagation context through Smart Prediction and return its result"
```

---

### Task 7: Nút "Xong ảnh" trong editor

**Files:**
- Create: `web/apps/labelstudio/src/pages/DataManager/propagation.js`
- Test: `web/apps/labelstudio/src/pages/DataManager/__tests__/propagation.test.js`
- Modify: `web/apps/labelstudio/src/pages/DataManager/DataManager.jsx:224-352, ~416` (tách đoạn nạp kết quả của `runSmartPrediction` thành `applyResults(annotation, results)` dùng chung; thêm `runFrameDone`, nút, ô tuỳ chọn, phím tắt)
- Modify: `web/apps/labelstudio/src/pages/DataManager/DataManager.styl` (ô tuỳ chọn)

**Interfaces:**
- Consumes: response của Task 6: `{data: {result, model_version}, propagation: {stop, source, distance, adapt, scores}}`.
- Produces (`propagation.js`, hàm thuần):
  - `DEFAULT_SETTINGS = { n: 3, threshold: 0.8, auto: true }`
  - `loadSettings(storage = globalThis.localStorage) -> {n, threshold, auto}` — đọc khoá `"tasdf.propagation"`, kẹp `n` vào [0, 10] (số nguyên), `threshold` vào [0, 1]; lỗi hoặc thiếu → `DEFAULT_SETTINGS`. `saveSettings(settings, storage = globalThis.localStorage) -> void` bỏ qua lỗi.
  - `onTarget(region, target) -> boolean` (cùng quy tắc trong `runSmartPrediction`).
  - `isDoctorRegion(region) -> boolean` — `region.origin !== "prediction"` (thiếu `origin` = bác sĩ).
  - `baseId(id) -> string` — bỏ hậu tố `@…`.
  - `acceptTarget(results, target) -> results` — vùng `origin: "prediction"` trên ảnh đích thành `"prediction-changed"`.
  - `frameTarget(frames, target, k) -> {url, to_name, item_index} | null` — `frames` là mảng `{to_name, urls}` theo thứ tự frame; frame của `target` + k, cùng `item_index`; ngoài phạm vi hoặc thiếu url → `null`.
  - `mergeProposal(results, target, proposal) -> results | null` — `null` khi ảnh đích có `isDoctorRegion`; ngược lại bỏ vùng `prediction` cũ của ảnh đích, thêm `proposal` với `id = \`${baseId(id)}@${to_name}:${item_index ?? 0}\``, `origin: "prediction"`, `item_index` của đích (khi khác `null`).

Luồng `runFrameDone` (trong `DataManager.jsx`):
1. Lấy `target` (`getTargetImage`) và các vùng trên đó. Không có vùng → toast "Ảnh này chưa có vùng nào để học" và dừng.
2. `applyResults(annotation, acceptTarget(...))`; `gold = {url: target.url, results: <vùng trên đích>}`; `carried` = các vùng đó.
3. `!settings.auto || settings.n === 0` → gọi `predictWithML` một lần với `image: target`, `context: {action: "propagate", carried, distance: 0, threshold, gold}`, bỏ kết quả, toast "Đã học từ ảnh này", dừng.
4. Với k = 1…n: `next = frameTarget(frames, target, k)`; `null` → dừng (toast "Đã tới frame cuối"); `merged = mergeProposal(<annotation hiện tại>, next, [])` là `null` → dừng (toast "Gặp ảnh đã gán, dừng lan") — kiểm tra **trước** khi gọi model; gọi `predictWithML` với `image: next`, `context: {action: "propagate", carried, distance: k, threshold, ...(k === 1 ? {gold} : {})}`; `applyResults(annotation, mergeProposal(..., resp.data.result))`; `carried = resp.data.result`; `resp.propagation?.stop` → toast "Dừng ở frame …: <lý do>" và dừng.
5. `frames` dựng từ các tag image của annotation: `{to_name: tag.name, urls: tag.parsedValueList ?? [tag.parsedValue]}`, sắp theo số cuối tên.

UI: nút "Xong ảnh" cạnh nút Smart Prediction (vô hiệu khi đang chạy); ô số `N` (`id="tasdf-prop-n"`, 0–10), ô số ngưỡng (`id="tasdf-prop-threshold"`, 0–1, bước 0.05), ô chọn "Tự lan" (`id="tasdf-prop-auto"`), mỗi ô có `<label htmlFor>`; đổi giá trị thì `saveSettings`. Phím Shift+Enter chạy `runFrameDone` khi focus không nằm trong `input`, `textarea`, `select` hay `[contenteditable]`.

- [ ] **Step 1: Viết test hỏng**

```javascript
import { acceptTarget, frameTarget, isDoctorRegion, loadSettings, mergeProposal, DEFAULT_SETTINGS } from "../propagation";

const t0 = { url: "u0", to_name: "timelapse_0", item_index: 5 };
const frames = [0, 1, 2].map((i) => ({ to_name: `timelapse_${i}`, urls: Array.from({ length: 11 }, (_, s) => `f${i}s${s}`) }));
const pred = (id, to_name, origin = "prediction") => ({ id, to_name, item_index: 5, origin, value: {} });

test("loadSettings falls back and clamps", () => {
  expect(loadSettings({ getItem: () => { throw new Error("blocked"); } })).toEqual(DEFAULT_SETTINGS);
  expect(loadSettings({ getItem: () => '{"n":42,"threshold":3,"auto":false}' })).toEqual({ n: 10, threshold: 1, auto: false });
});

test("frameTarget moves along frames on the same slice and stops at the end", () => {
  expect(frameTarget(frames, t0, 1)).toEqual({ url: "f1s5", to_name: "timelapse_1", item_index: 5 });
  expect(frameTarget(frames, t0, 3)).toBeNull();
});

test("doctor regions are anchors", () => {
  expect(isDoctorRegion({ origin: "manual" })).toBe(true);
  expect(isDoctorRegion({})).toBe(true);
  expect(isDoctorRegion({ origin: "prediction" })).toBe(false);
  const t1 = frameTarget(frames, t0, 1);
  expect(mergeProposal([pred("a", "timelapse_1", "prediction-changed")], t1, [])).toBeNull();
});

test("mergeProposal replaces only untouched proposals on the target", () => {
  const t1 = frameTarget(frames, t0, 1);
  const other = pred("x@timelapse_0:5", "timelapse_0");
  const out = mergeProposal([other, pred("old@timelapse_1:5", "timelapse_1")], t1, [{ id: "zp1@timelapse_0:5", to_name: "timelapse_1", value: {} }]);
  expect(out.map((r) => r.id)).toEqual(["x@timelapse_0:5", "zp1@timelapse_1:5"]);
  expect(out[1]).toMatchObject({ origin: "prediction", item_index: 5 });
});

test("acceptTarget marks the confirmed image as touched", () => {
  const out = acceptTarget([pred("a", "timelapse_0"), pred("b", "timelapse_1")], t0);
  expect(out.map((r) => r.origin)).toEqual(["prediction-changed", "prediction"]);
});
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `cd web && npx jest --config apps/labelstudio/jest.config.ts apps/labelstudio/src/pages/DataManager/__tests__/propagation.test.js`
Expected: FAIL — `Cannot find module '../propagation'`

- [ ] **Step 3: Viết `propagation.js` theo Interfaces**

- [ ] **Step 4: Chạy test, phải qua**

Run: `cd web && npx jest --config apps/labelstudio/jest.config.ts apps/labelstudio/src/pages/DataManager/__tests__/propagation.test.js`
Expected: PASS

- [ ] **Step 5: Tách `applyResults`, thêm `runFrameDone`, nút, ô tuỳ chọn và phím tắt vào `DataManager.jsx` theo luồng ở trên; Smart Prediction phải chạy y như cũ**

- [ ] **Step 6: Kiểm tra lint và build của app**

Run: `cd web && npx biome check apps/labelstudio/src/pages/DataManager && npx nx run labelstudio:build`
Expected: không lỗi

- [ ] **Step 7: Commit**

```bash
git add web/apps/labelstudio/src/pages/DataManager
git commit -m "feat(editor): confirm an image with Shift+Enter and propagate it to the next frames"
```

---

### Task 8: Cấu hình triển khai, tài liệu và kiểm tra đầu cuối

**Files:**
- Modify: `docker-compose.tasdf.yml` (`environment:` thêm `JOURNAL_DIR: /data/tasdf-online/propagation`, cùng tầng với các đường dẫn khác)
- Modify: `integrations/tasdf-online-learning/tasdf.env.example` (các dòng `ADAPT_*` với giá trị mặc định ở Global Constraints, kèm một dòng giải thích)
- Modify: `integrations/tasdf-online-learning/README.md` (mục mới "Tầng ngắn hạn: Xong ảnh": thao tác, bốn luật, adapter, cổng, nhật ký, giới hạn chỉ 4 lớp TASDF được nắn biên)

**Interfaces:**
- Consumes: toàn bộ Task 1–7.

- [ ] **Step 1: Sửa ba file trên; commit**

```bash
git add docker-compose.tasdf.yml integrations/tasdf-online-learning/tasdf.env.example integrations/tasdf-online-learning/README.md
git commit -m "docs(tasdf): document the Done-image propagation and its settings"
```

- [ ] **Step 2: Hỏi người dùng trước khi thêm project 38 vào `.data/tasdf-online/stages.json` (dữ liệu vận hành, không thuộc repo); chỉ sửa khi được đồng ý**

- [ ] **Step 3: Build và khởi động lại backend + Label Studio qua `$SAFE`, CPU < 60°C trước khi build**

Run (từ thư mục gốc repo, cùng bộ file compose container đang dùng):
```bash
$SAFE docker compose -f docker-compose.yml -f docker-compose.tasdf.yml -f docker-compose.tasdf-gpu.yml up -d --build tasdf-online
$SAFE docker compose -f docker-compose.yml build webapp
docker compose -f docker-compose.yml -f docker-compose.tasdf.yml -f docker-compose.tasdf-gpu.yml up -d --no-deps webapp
```
Expected: `docker logs label-studio-tasdf-online-1` có dòng `settings: {...}` chứa `crop_size` và không có traceback.

- [ ] **Step 4: Kiểm tra đầu cuối trên project 38 (nhờ người dùng hard-refresh Ctrl+Shift+R)**

Mở well W1, frame 1, slice 6. Vẽ một vòng ZP và một rect PN, bấm Shift+Enter với N=3.
Expected:
- frame 2–4 cùng slice có đúng hai vùng (ZP đã nắn, PN copy nguyên), không vùng thừa;
- vẽ tay một vùng ở frame 3 rồi bấm lại ở frame 1: chuỗi dừng ở frame 3, vùng tay còn nguyên;
- `.data/tasdf-online/propagation/38/<task>.jsonl` có dòng `gold` và `propose`; có file `.npz`;
- `POST /train {"force": true, "dry_run": true}` vẫn chạy xong.
