# TASDF online-learning ML backend

Đây là ML backend dùng cho Label Studio (fork Fujinet). Nó phân vùng phôi nang bằng mô hình TASDF:
- **Backbone dùng chung**, cộng **một adapter (BN + head) cho mỗi giai đoạn phôi**. Mỗi project LS thuộc về một giai đoạn, khai báo trong `stages.json`.
- Model **chỉ đổi qua tầng học dài hạn** (hằng tuần hoặc `POST /train`). Phiên bản mới chỉ lên production khi tốt hơn thật trên dữ liệu chưa từng học, không học vẹt và không quên.

Thiết kế: [`docs/superpowers/specs/2026-11-13-tasdf-stage-adapters-long-term-design.md`](../../docs/superpowers/specs/2026-11-13-tasdf-stage-adapters-long-term-design.md).

## Model nằm ở đâu, thay đổi thế nào

```
${TASDF_DATA_DIR}/                     (= .data/tasdf-online trên host, /data/tasdf-online trong container)
  baseline.pt                          model gốc M2-B, không bao giờ đổi
  stages.json                          giai đoạn -> project LS, holdout/anchor (mẫu: stages.example.json)
  registry/registry.json               danh sách bundle, con trỏ production/previous, lịch sử thăng cấp
  registry/decoders/dec-*.pt           decoder dùng chung (fp16, ~49 MB)
  registry/adapters/<stage>/ad-*.pt    adapter của từng giai đoạn (~0,3 MB)
  registry/legacy/<project>/           registry cũ v1..vN (chỉ tra cứu, không phục vụ)
  reports/                             learning curve (scripts/learning_curve.py)
```

- **Tên phiên bản:**
  - Bundle: `tasdf-<ngày>-<hash6>`, duy nhất và không dùng lại.
  - LS thấy `model_version = <bundle>.<stage>`, ví dụ `tasdf-20261120-4be7a2.blastocyst`.
  - `tasdf-base` là model gốc.
- **Xem:** `GET /registry` (header `X-OL-Token`), hoặc mở `/dashboard`.
- **Đổi production:**
  - Tự động: job dài hạn qua cổng.
  - Thủ công: `POST /promote {"bundle": "...", "reason": "..."}`. Cổng vẫn chạy; muốn bỏ qua thì thêm `"override": true`, lý do được ghi vào `history`.
- **Quay lại:** `POST /rollback` đặt production về previous.
- **Thêm project vào một giai đoạn:** sửa `stages.json` rồi khởi động lại container. Project chưa khai báo sẽ bị `/setup` và `/predict` từ chối kèm thông báo rõ.

## Luồng hoạt động

```
LS mở task ──/predict (project → stage)──> backbone production + adapter của stage ──> polygon (<bundle>.<stage>)
LS lưu annotation ──/webhook──> chỉ đếm nhãn mới
lịch TRAIN_CRON (mặc định CN 01:00, TZ) hoặc POST /train ──> job dài hạn (1 job một lúc, hoãn nếu thiếu RAM/VRAM)
  → đọc lại mọi annotation đã duyệt từ LS; nhãn 256: ngoài ZP = nền, trong ZP không vẽ = ignore
  → chia train/val theo hash nội dung ảnh (~20% val, không bao giờ học); cần ≥ TRAIN_MIN_NEW nhãn mới
  → học tiếp từ production: decoder + BN + head (encoder đóng băng), augmentation, AdamW + cosine,
    D10 anchor ≤ 25% mỗi batch, dừng sớm, giới hạn TRAIN_MAX_HOURS; hiệu chỉnh BN nếu ≥ BN_RECAL_MIN ảnh
  → cứ AUDIT_EVERY job: học thêm một bản từ baseline, thay bản warm-start nếu bản từ baseline tốt hơn
  → cổng: val tăng ≥ max(PROMOTE_MIN_GAIN, 2 × std bootstrap) · train − val ≤ MAX_TRAIN_VAL_GAP
          · không quên trên D10/valid (so production và baseline) · encoder không đổi
  → đạt: production ← bundle mới, previous ← cũ, PATCH model_version của mọi project trong stage
```

Số đo trên RTX 3060 Laptop (6 GB): **49 ms/ảnh/epoch** khi học nặng, VRAM đỉnh **2,4 GB**. 1000 ảnh × 60 epoch mất khoảng 50 phút. Chạy thử 1 epoch trên dữ liệu thật: `POST /train {"force": true, "dry_run": true}`.

**Có tốt lên thật không?** Sau mỗi đợt gán nhãn, chạy:
```bash
I=integrations/tasdf-online-learning C=label-studio-tasdf-online-1
docker cp $I/scripts/learning_curve.py $C:/tmp/
SAFE_RUN_CONTAINER=$C $I/scripts/safe_run.sh docker exec -e PYTHONPATH=/app $C python /tmp/learning_curve.py --projects 36,37
```
Script so sánh học nhẹ / nặng-từ-base / nặng-từ-production theo số nhãn, trên val chưa từng học (dùng leave-one-out khi val < 10 ảnh). Kết quả ghi vào `reports/history.md`.
Mỗi dòng đo được ghi ngay vào `reports/learning_curve-<thời điểm>.jsonl`. Bị ngắt thì chạy lại với `--resume <file.jsonl>` để bỏ qua phần đã đo.
Với leave-one-out, có thể chạy từng fold (`--fold i`, không ghi báo cáo), rồi chạy một lần `--resume` cuối để tổng hợp. Như vậy GPU được nghỉ giữa các fold.

**Chạy an toàn trên laptop.** Máy đã tự tắt khi GPU chạy 100% liên tục. Vì vậy mọi lệnh nặng (pytest, learning curve) chạy qua `scripts/safe_run.sh`:
- Script từ chối chạy khi máy không cắm sạc.
- Tạm dừng lệnh khi GPU ≥ 80°C hoặc CPU ≥ 90°C, chạy tiếp khi cả hai < 70°C.
- Ghi nhiệt độ vào `reports/safe_run.log`.

Job trong backend cũng hoãn khi máy chạy pin hoặc GPU ≥ 75°C, và nghỉ giữa các epoch khi GPU ≥ 80°C.

## Cấu hình (env)

Khi chạy bằng Docker, mỗi biến chỉ có **một** nơi đặt:

| Tầng | Đặt ở đâu | Biến |
|---|---|---|
| Đường dẫn trong container | `environment:` của `docker-compose.tasdf.yml` (cố định, ghi đè `tasdf.env`) | `LS_URL`, `STATE_DIR`, `BASELINE_CKPT`, `HOLDOUT_NPZ`, `ANCHOR_NPZ` |
| Thông số theo môi trường | `.env` ở thư mục gốc repo hoặc `docker compose --env-file <file>`. Mẫu: [`compose.env.example`](compose.env.example) | `TASDF_DATA_DIR`, `TASDF_UID`/`TASDF_GID`, `TASDF_PUBLISH`, `TASDF_DEVICE`, `TASDF_TORCH_THREADS`, `TASDF_HOLDOUT_LIMIT`, `TASDF_DASHBOARD`, `TASDF_TORCH_INDEX_URL` |
| Secret và các tham số app còn lại | `${TASDF_DATA_DIR}/tasdf.env`, không commit. Mẫu: [`tasdf.env.example`](tasdf.env.example) | `LS_TOKEN`, `OL_TOKEN`, `MIN_QUALITY_LEVEL`, `REFRESH_NEXT_N`, tham số huấn luyện … |

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
| `TRAIN_CRON` / `TZ` | `0 1 * * 0` / `Asia/Ho_Chi_Minh` | Lịch học dài hạn (cron 5 trường, giờ địa phương); rỗng thì tắt lịch. Trong compose: `TASDF_TRAIN_CRON`, `TASDF_TZ` |
| `TRAIN_MIN_NEW` / `TRAIN_EPOCHS` / `TRAIN_MAX_HOURS` / `LR` | 10 / 60 / 2 / 3e-4 | Ngưỡng nhãn mới để chạy theo lịch; số epoch tối đa; giới hạn giờ; learning rate |
| `PROJECT_VAL_MIN` / `PROMOTE_MIN_GAIN` / `MAX_TRAIN_VAL_GAP` | 10 / 0.005 / 0.15 | Cổng: từ bao nhiêu ảnh val thì bắt buộc tăng (vượt cả nhiễu bootstrap); giới hạn khoảng cách train − val |
| `AUDIT_EVERY` / `AUDIT_MARGIN` / `BN_RECAL_MIN` | 4 / 0.01 / 32 | Kiểm tra định kỳ bằng một bản học từ baseline; ngưỡng ảnh để hiệu chỉnh thống kê BN |
| `MIN_FREE_RAM_GB` / `MIN_FREE_VRAM_GB` | 3 / 3 | Job hoãn (ghi lý do) khi bộ nhớ trống thấp hơn. Compose giới hạn container `TASDF_MEM_LIMIT=6g` |
| `STAGES_FILE` / `REGISTRY_DIR` / `STATE_DIR` | `/data/tasdf-online/{stages.json,registry,state}` | Ánh xạ giai đoạn → project; registry; registry cũ (tự chuyển vào `registry/legacy`) |
| `LS_URL`, `LS_TOKEN` | — | `LS_TOKEN` phải là token của user có role **Manager trong project**. Org Admin chỉ có quyền đọc trong fork này |
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
DEVICE=cuda BASELINE_CKPT=... STAGES_FILE=... REGISTRY_DIR=... LS_URL=... LS_TOKEN=... OL_TOKEN=... python -m tasdf_online.app
```

## Cấu hình Label Studio

1. Tạo project và dán nội dung `label-config.xml`. ZP và TE vẽ dạng ring; ICM và BC vẽ polygon thường.
2. Tạo một user dịch vụ, cho vào org với role Member và gán **Manager** trong project (Settings → Members). Lấy token của user này làm `LS_TOKEN`.
3. Vào Settings → Model, thêm URL của backend (ví dụ `http://tasdf-online:9090`). **Không** bật "Interactive preannotations". LS sẽ tự tạo webhook tới `/webhook`.
4. Thêm id project vào giai đoạn tương ứng trong `stages.json` rồi khởi động lại backend. Khi khởi động và sau mỗi lần thăng cấp, backend PATCH `project.model_version = <bundle>.<stage>` cho mọi project đã khai báo.
5. Muốn LS tự gọi `/train` sau mỗi N annotation: đặt `min_annotations_to_start_training` của project. Job vẫn chỉ chạy khi đủ `TRAIN_MIN_NEW` nhãn mới.
6. Trong màn hình gán nhãn, bấm **Smart Prediction** (thanh trên): gọi model cho task đang mở (`force`, luôn tạo prediction mới) và nạp kết quả vào annotation hiện tại; nếu đã có vùng tự vẽ cùng nhãn, hộp xác nhận hỏi giữ vùng tự vẽ (OK) hay thay bằng dự đoán (Cancel — xoá vùng tự vẽ cùng nhãn). Chỉ dùng backend non-interactive.

**An toàn:**
- Webhook chỉ để đếm. Job đọc lại mọi task và annotation từ LS bằng `LS_TOKEN`; không có gì trong payload webhook được tin.
- `/setup`, `/predict`, `/webhook`, `/train` không có xác thực (giao thức ML backend của LS) và compose publish cổng 9090 ra mọi interface: chỉ chạy trong mạng nội bộ tin cậy; nếu phải lộ ra ngoài, đặt reverse proxy có Basic Auth và chọn `Basic Auth` khi thêm backend trong LS. `LS_TOKEN` là bắt buộc: thiếu nó job từ chối train (chỉ học dữ liệu đọc lại từ LS).
- `registry.json` ghi sha256 của baseline; đổi `BASELINE_CKPT` mà vẫn giữ registry cũ thì backend dừng ngay khi khởi động. Metric D10 của baseline được đo lại khi file holdout của một giai đoạn thay đổi.

## Test

```bash
pip install -r requirements-dev.txt torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

Test CUDA tự bỏ qua nếu máy không có GPU.

## Giới hạn đã biết

- Mỗi ring polygon chỉ có **một lỗ**, đúng như giới hạn của editor fork.
- Phần A chỉ huấn luyện **một giai đoạn** (`blastocyst`). Học chung nhiều giai đoạn với tập lớp khác nhau thuộc phần C.
- Khi val có dưới `PROJECT_VAL_MIN` ảnh, job vẫn học và ghi kết quả (`insufficient_val`) nhưng **không bao giờ tự thăng cấp**: vài ảnh chưa chứng minh được gì. Muốn dùng bản đó thì `POST /promote` với `override` và lý do. Hãy gán thêm nhãn và theo dõi `reports/history.md`.
- Tầng ngắn hạn (adapter theo phôi) và script cắt phôi khỏi đĩa thuộc phần B, chưa làm.
