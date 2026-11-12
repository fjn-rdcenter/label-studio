# TASDF online-learning ML backend

Đây là ML backend dùng cho Label Studio (fork Fujinet). Nó huấn luyện lại mô hình TASDF (phân vùng phôi nang ZP/TE/ICM/BC) **theo đợt** trên toàn bộ annotation đã duyệt của dự án, và chỉ thăng cấp khi đo được là không quên (D10) và không kém đi trên chính dự án. Thiết kế: [`docs/superpowers/specs/2026-11-12-tasdf-batch-training-design.md`](../../docs/superpowers/specs/2026-11-12-tasdf-batch-training-design.md).

- **Đặc tả:** `D:\Workspace\model\context\context\online_learning.md`, mục 8.
- **Kế hoạch:** nhánh `ML-backend-design`.

## Luồng hoạt động

```
LS mở task ──/predict──> ensemble(net1,net2) của version active ──> polygon/ring (tasdf-vN)
LS lưu annotation ──/webhook──┐  (trả ngay, job chạy nền, 1 job một lúc)
LS "Start Training" ─/train─────┤
                                ▼
  job: kéo mọi task + annotation từ LS (không tin payload) → annotation mới nhất, quality >= MIN_QUALITY_LEVEL
       → nhãn 256: ngoài ZP = nền, trong ZP không vẽ = ignore (lớp bị bỏ không bị dạy thành nền)
       → chia cố định theo task: ~20% project-val (không bao giờ học), còn lại train
       → đủ TRAIN_EVERY_N task train mới (hoặc /train)? → học từ v0: norm_and_heads (311,570 tham số,
         BN stats đóng băng), CE + Dice, mỗi bước 4 mẫu dự án + tối đa 4 anchor D10, giữ checkpoint tốt nhất
       → cổng: chống quên trên D10/valid (so active và v0) + project-val (≥ PROJECT_VAL_MIN ảnh: phải tăng
         PROMOTE_MIN_GAIN; ít hơn: không được giảm quá 0.01)
       → đạt: lưu vN (~1.27 MB), đổi teacher, đồng bộ prediction sang LS
```

## Cấu hình (env)

Khi chạy bằng Docker, mỗi biến chỉ có **một** nơi đặt:

| Tầng | Đặt ở đâu | Biến |
|---|---|---|
| Đường dẫn trong container | `environment:` của `docker-compose.tasdf.yml` (cố định, ghi đè `tasdf.env`) | `LS_URL`, `STATE_DIR`, `BASELINE_CKPT`, `HOLDOUT_NPZ`, `ANCHOR_NPZ` |
| Thông số theo môi trường | `.env` ở thư mục gốc repo hoặc `docker compose --env-file <file>`. Mẫu: [`compose.env.example`](compose.env.example) | `TASDF_DATA_DIR`, `TASDF_UID`/`TASDF_GID`, `TASDF_PUBLISH`, `TASDF_DEVICE`, `TASDF_TORCH_THREADS`, `TASDF_HOLDOUT_LIMIT`, `TASDF_DASHBOARD`, `TASDF_TORCH_INDEX_URL` |
| Secret và các tham số app còn lại | `${TASDF_DATA_DIR}/tasdf.env`, không commit. Mẫu: [`tasdf.env.example`](tasdf.env.example) | `LS_TOKEN`, `LS_PROJECT_ID`, `OL_TOKEN`, `MIN_QUALITY_LEVEL`, `REFRESH_NEXT_N`, … |

Lưu ý:
- Biến thuộc hai tầng trên mà đặt trong `tasdf.env` sẽ **bị bỏ qua mà không báo lỗi**. Ví dụ: `DASHBOARD_ENABLED=0` đặt trong `tasdf.env` không tắt được dashboard, phải dùng `TASDF_DASHBOARD=0`.
- Lúc khởi động, backend log dòng `settings: {...}` (đã bỏ token) ra `docker logs`. Đó là giá trị thực tế đang dùng.

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `DEVICE` | `auto` | `cpu` / `cuda` / `cuda:N`. Nếu đặt `cuda` mà máy không có GPU thì service dừng ngay khi khởi động |
| `TORCH_THREADS` | 0 | Số luồng khi chạy CPU |
| `AMP` | `auto` | fp16 chỉ bật khi chạy cuda |
| `BASELINE_CKPT` | — | `best.pt` của M2-B (checkpoint đầy đủ hoặc state_dict). Bỏ trống thì dùng model ngẫu nhiên, chỉ để test |
| `BASE_CHANNELS` / `EROSION` | 64 / `cross` | Phải khớp với checkpoint |
| `HOLDOUT_NPZ` / `HOLDOUT_LIMIT` | — / 0 | D10/valid (8 ảnh, đủ 4 lớp) cho cổng chống quên. 0 = dùng toàn bộ |
| `ANCHOR_NPZ` | — | D10/train (32 mask) trộn vào mỗi bước huấn luyện, không vượt số mẫu dự án. Tuỳ chọn; thiếu file thì chỉ học trên dữ liệu dự án |
| `TRAIN_EVERY_N` / `TRAIN_STEPS` / `EVAL_EVERY` / `LR` | 10 / 200 / 20 / 3e-4 | Ngưỡng task train mới để tự chạy job; số bước; đo mỗi bao nhiêu bước để giữ checkpoint tốt nhất |
| `PROJECT_VAL_MIN` / `PROMOTE_MIN_GAIN` | 10 / 0.005 | Từ bao nhiêu ảnh project-val thì bắt buộc tăng mới thăng cấp |
| `STATE_DIR` | `/data/tasdf-online` | Nơi lưu `registry.json` và `versions/vN.pt` cho từng project |
| `LS_URL`, `LS_TOKEN`, `LS_PROJECT_ID` | — | `LS_TOKEN` phải là token của user có role **Manager trong project**. Org Admin chỉ có quyền đọc trong fork này |
| `MIN_QUALITY_LEVEL` | 2 | Chỉ học từ annotation đã được Reviewer/Manager duyệt |
| `OL_TOKEN` | — | Bảo vệ `/telemetry/live`, `/rollback`, `/refresh_ls_predictions`, `/dashboard/toggle`; chỉ nhận qua header `X-OL-Token` (không nhận `?token=`), trên `/dashboard` nhập vào ô Token. Bỏ trống: các endpoint này **chỉ mở khi `DASHBOARD_ENABLED=1`** (môi trường dev), còn khi `DASHBOARD_ENABLED=0` thì bị khoá. Không dùng chuỗi `tasdf-secret-token` (từng được build vào `web/dist`) |
| `DASHBOARD_ENABLED` | `1` | `/dashboard` là công cụ kiểm tra lúc phát triển, **không thuộc sản phẩm**: triển khai thật đặt `TASDF_DASHBOARD=0` (kèm `OL_TOKEN` nếu vẫn cần rollback/refresh). Dashboard chạy chung tiến trình gunicorn với ML backend (container `tasdf-online`, cổng 9090), không có service riêng. `0` = không mount: `/dashboard` và `/dashboard/toggle` không tồn tại, không bật lại được lúc chạy. Dev mở trực tiếp `http://<host>:9090/dashboard` (Data Manager không có nút). Khi khởi động với `1`, bật/tắt chụp telemetry lúc chạy qua `POST /dashboard/toggle`. Có thể chạy tách rời bằng `BACKEND_URL=http://localhost:9090 DASHBOARD_PORT=9091 python -m tasdf_online.dashboard` |
| `REFRESH_NEXT_N` | 0 | Số task chưa gán nhãn được predict sẵn sau mỗi lần promote (và lúc khởi động). Mỗi lần đồng bộ PATCH `project.model_version`; prediction cũ **không bị xoá** (annotation đang mở có thể trỏ tới nó qua `parent_prediction`), LS chỉ hiển thị prediction đúng version và tự predict lại task chưa có version hiện hành khi mở (`label_studio/tasks/api.py`). `0` = on-demand: chỉ predict sẵn 1 task (LS yêu cầu có ≥1 prediction của version mới mới cho đổi `model_version`); các task khác tự predict khi mở từ Data Manager. Luồng Label Stream (`/next/`) chỉ có gợi ý cho task đã được predict sẵn |

Tạo file holdout và anchor từ D10 (folder `images/` + `masks/`, lớp 0..4). Script từ chối mask thiếu lớp; **không** dùng D10 `test` (không có mask):

```bash
python scripts/export_npz.py --root <Research>/dataset/D10 --split valid --out holdout.npz
python scripts/export_npz.py --root <Research>/dataset/D10 --split train --out anchors.npz
```

## Chạy

**CPU** (cùng docker network với LS):

```bash
cp integrations/tasdf-online-learning/compose.env.example .env      # sửa theo môi trường (khối cuối file)
mkdir -p .data/tasdf-online                                          # = TASDF_DATA_DIR; compose không tự tạo
cp integrations/tasdf-online-learning/tasdf.env.example .data/tasdf-online/tasdf.env   # điền token
# đặt baseline.pt, holdout.npz (và anchors.npz nếu có) vào .data/tasdf-online/
docker compose -f docker-compose.yml -f docker-compose.tasdf.yml up -d --build
```

Container chạy với `TASDF_UID:TASDF_GID` (mặc định 1000:1000), và user này phải ghi được `TASDF_DATA_DIR`:
- Linux: đặt bằng `id -u` / `id -g`. Không dùng biến `UID` của shell, vì bash không export nó và cũng không cho gán lại.
- Rootless Docker: đặt `0`.
- Docker Desktop: giá trị nào cũng được.

Nếu cấu hình sai, `docker logs` sẽ báo `PermissionError ... registry.json.tmp`.

**GPU** (cần NVIDIA Container Toolkit). Overlay GPU tự đặt `DEVICE=cuda`, `HOLDOUT_LIMIT=0`; chỉ bỏ comment `TASDF_*` trong `.env` khi muốn khác (vd. `TASDF_DEVICE=cuda:1`). Đặt `TASDF_DEVICE=cuda` mà chạy không có overlay GPU thì service dừng ngay:

```bash
docker compose -f docker-compose.yml -f docker-compose.tasdf.yml -f docker-compose.tasdf-gpu.yml up -d --build
```

**Không dùng Docker** (ví dụ trên GPU server). Dùng đường dẫn trên host. Không dùng lại `tasdf.env` của Docker: một file env chung cho cả hai cách chạy đã từng làm đường dẫn host lọt vào container.

```bash
DEVICE=cuda BASELINE_CKPT=... HOLDOUT_NPZ=... LS_URL=... LS_TOKEN=... LS_PROJECT_ID=... OL_TOKEN=... python -m tasdf_online.app
```

## Cấu hình Label Studio

1. Tạo project và dán nội dung `label-config.xml`. ZP và TE vẽ dạng ring; ICM và BC vẽ polygon thường.
2. Tạo một user dịch vụ, cho vào org với role Member và gán **Manager** trong project (Settings → Members). Lấy token của user này làm `LS_TOKEN`.
3. Vào Settings → Model, thêm URL của backend (ví dụ `http://tasdf-online:9090`). **Không** bật "Interactive preannotations". LS sẽ tự tạo webhook tới `/webhook`.
4. Khi khởi động, backend tự PATCH `project.model_version = tasdf-<active>`, nhờ đó LS hiển thị đúng prediction của version đang chạy.
5. Trong màn hình gán nhãn, bấm **Smart Prediction** (thanh trên): gọi model cho task đang mở (`force`, luôn tạo prediction mới) và nạp kết quả vào annotation hiện tại; nếu đã có vùng tự vẽ cùng nhãn, hộp xác nhận hỏi giữ vùng tự vẽ (OK) hay thay bằng dự đoán (Cancel — xoá vùng tự vẽ cùng nhãn). Chỉ dùng backend non-interactive.

**An toàn:**
- Webhook chỉ là tín hiệu: worker đọc lại annotation (`/api/annotations/<id>`) và task (`/api/tasks/<id>`) từ LS bằng `LS_TOKEN`; `result`, `quality_level`, ảnh trong payload không được tin. Payload thiếu `project` bị bỏ qua.
- `/setup`, `/predict`, `/webhook`, `/train` không có xác thực (giao thức ML backend của LS) và compose publish cổng 9090 ra mọi interface: chỉ chạy trong mạng nội bộ tin cậy; nếu phải lộ ra ngoài, đặt reverse proxy có Basic Auth và chọn `Basic Auth` khi thêm backend trong LS. `LS_TOKEN` là bắt buộc: thiếu nó job từ chối train (chỉ học dữ liệu đọc lại từ LS).
- `registry.json` ghi fingerprint (sha256 baseline, sha256 holdout, `HOLDOUT_LIMIT`): đổi holdout → tự đo lại metrics của v0 và version active; đổi `BASELINE_CKPT` với `STATE_DIR` cũ → dừng khi khởi động.

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
