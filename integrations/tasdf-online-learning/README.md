# TASDF online-learning ML backend

Đây là ML backend dùng cho Label Studio (fork Fujinet). Nó tự cập nhật mô hình TASDF (phân vùng phôi nang ZP/TE/ICM/BC) mỗi khi một annotation đã được review được lưu.

- **Đặc tả:** `D:\Workspace\model\context\context\online_learning.md`, mục 8.
- **Kế hoạch:** nhánh `ML-backend-design`.

## Luồng hoạt động

```
LS mở task ──/predict──> ensemble(net1,net2) của version active ──> polygon/ring (tasdf-vN)
LS lưu annotation ──/webhook (tự tạo khi thêm backend)──> queue (trả 200 ngay)
  worker: Y = polygon bác sĩ, P_init = parent_prediction (đúng thứ bác sĩ đã thấy)
          → episode: student (norm_and_heads: 311,570 tham số, BN stats đóng băng) + replay
          → EMA α=0.65 → đánh giá holdout → cổng chống quên (so v_{t-1} và v0)
          → nếu đạt: lưu delta (~1.27 MB), đổi teacher, đồng bộ prediction sang LS
```

## Cấu hình (env)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `DEVICE` | `auto` | `cpu` / `cuda` / `cuda:N`. Nếu đặt `cuda` mà máy không có GPU thì service dừng ngay khi khởi động |
| `TORCH_THREADS` | 0 | Số luồng khi chạy CPU |
| `AMP` | `auto` | fp16 chỉ bật khi chạy cuda |
| `BASELINE_CKPT` | — | `best.pt` của M2-B (checkpoint đầy đủ hoặc state_dict). Bỏ trống thì dùng model ngẫu nhiên, chỉ để test |
| `BASE_CHANNELS` / `EROSION` | 64 / `cross` | Phải khớp với checkpoint |
| `HOLDOUT_NPZ` / `HOLDOUT_LIMIT` | — / 0 | Tập dùng cho cổng chống quên. 0 = dùng toàn bộ. Thiếu holdout thì mọi candidate đều bị từ chối |
| `ANCHOR_NPZ` | — | Mẫu neo cho replay khi chưa có đủ mẫu đã duyệt |
| `EPISODE_EPOCHS` / `REPLAY_SIZE` / `LR` / `EMA_ALPHA` / `CORR_BETA` / `TOPO_SUP_WEIGHT` | 3 / 4 / 1e-4 / 0.65 / 0.5 / 0.1 | Tham số của một episode |
| `STATE_DIR` | `/data/tasdf-online` | Nơi lưu `registry.json` và `versions/vN.pt` cho từng project |
| `LS_URL`, `LS_TOKEN`, `LS_PROJECT_ID` | — | `LS_TOKEN` phải là token của user có role **Manager trong project**. Org Admin chỉ có quyền đọc trong fork này |
| `MIN_QUALITY_LEVEL` | 2 | Chỉ học từ annotation đã được Reviewer/Manager duyệt. Đặt 1 khi demo |
| `OL_TOKEN` | — | Chỉ nhận qua header `X-OL-Token` (không nhận `?token=` để tránh lộ trong access log), dùng cho `/telemetry/live`, `/rollback`, `/refresh_ls_predictions`, `/dashboard/toggle`. Trên `/dashboard` nhập token vào ô Token. Bỏ trống thì các endpoint này bị khoá |
| `DASHBOARD_ENABLED` | `1` | Bật/tắt công cụ debug & theo dõi `/dashboard` (hoặc bật/tắt lúc đang chạy qua `POST /dashboard/toggle`). Có thể chạy tách rời hoàn toàn bằng `BACKEND_URL=http://localhost:9090 DASHBOARD_PORT=9091 python -m tasdf_online.dashboard` |
| `REFRESH_NEXT_N` | 0 | Số task chưa gán nhãn được predict sẵn sau mỗi lần promote (và lúc khởi động). Mỗi lần đồng bộ đều xoá prediction cũ trên task chưa gán nhãn rồi PATCH `project.model_version`. `0` = on-demand: chỉ predict sẵn 1 task (LS yêu cầu có ≥1 prediction của version mới mới cho đổi `model_version`); các task khác tự predict khi mở từ Data Manager. Luồng Label Stream (`/next/`) chỉ có gợi ý cho task đã được predict sẵn |

Tạo file holdout và anchor từ repo nghiên cứu (định dạng folder `images/` + `masks/`):

```bash
python scripts/export_npz.py --tasdf-repo <tasdf-audit> --run-json <run.json> --split test --root-dir <tasdf-audit>/data/D10_eval --out holdout.npz
```

## Chạy

**CPU** (cùng docker network với LS). Đặt các file `baseline.pt`, `holdout.npz` và `tasdf.env` vào thư mục `.data/tasdf-online/`:

```bash
docker compose -f docker-compose.yml -f docker-compose.tasdf.yml up -d --build
```

**GPU** (cần NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.tasdf.yml -f docker-compose.tasdf-gpu.yml up -d --build
```

**Không dùng Docker** (ví dụ trên GPU server):

```bash
DEVICE=cuda BASELINE_CKPT=... HOLDOUT_NPZ=... LS_URL=... LS_TOKEN=... LS_PROJECT_ID=... OL_TOKEN=... python -m tasdf_online.app
```

## Cấu hình Label Studio

1. Tạo project và dán nội dung `label-config.xml`. ZP và TE vẽ dạng ring; ICM và BC vẽ polygon thường.
2. Tạo một user dịch vụ, cho vào org với role Member và gán **Manager** trong project (Settings → Members). Lấy token của user này làm `LS_TOKEN`.
3. Vào Settings → Model, thêm URL của backend (ví dụ `http://tasdf-online:9090`). **Không** bật "Interactive preannotations". LS sẽ tự tạo webhook tới `/webhook`.
4. Khi khởi động, backend tự PATCH `project.model_version = tasdf-<active>`, nhờ đó LS hiển thị đúng prediction của version đang chạy.

## Test

```bash
pip install -r requirements-dev.txt torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

Test CUDA tự bỏ qua nếu máy không có GPU.

## Giới hạn đã biết

- Mỗi ring polygon chỉ có **một lỗ**, đúng như giới hạn của editor fork.
- Gate chỉ đo quên trên miền dữ liệu cũ (holdout), không đo mức cải thiện trên ảnh của project. Hãy theo dõi `next_task_dice` trong `/telemetry/live`.
- `local_probe` (AR_Δ) đo trên chính mẫu vừa train, nên không phản ánh khả năng khái quát hoá.
- Mỗi project LS chỉ dùng được một ML backend. Không thể dùng chung project với backend OpenCV interactive.
