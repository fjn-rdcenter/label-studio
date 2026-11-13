# TASDF phần A: registry theo bộ + tầng học dài hạn (một giai đoạn "blastocyst")

Ngày: 2026-11-13. Trạng thái: đã chốt, kể cả các điểm ở mục 10 (người dùng chốt ngày 2026-11-13).
Tổng hợp brainstorm: `~/.claude/plans/federated-churning-crayon.md`.
Spec trước đó (bản thử nghiệm "đủ 10 nhãn/học nhẹ"): `2026-11-12-tasdf-batch-training-design.md`. Spec này **thay thế** cơ chế kích hoạt và mức học của bản đó, và giữ lại luật nhãn, cách chọn mẫu và cổng.

## 1. Mục tiêu và phạm vi
- Một nơi duy nhất quản lý model. Tên phiên bản là duy nhất, nhìn tên là biết model nào. Có thăng cấp và rollback rõ ràng.
- Model = **backbone chung** (encoder + decoder) + **adapter theo giai đoạn** (BN riêng + head riêng). Project = một giai đoạn.
- Tầng **dài hạn**: hằng tuần, ngoài giờ, học nặng trên toàn bộ nhãn đã duyệt. Chỉ tầng này được thay production.
- Phần A chỉ chạy **một giai đoạn `blastocyst`** (project 36, 37), nhưng định dạng registry và cấu hình đã hỗ trợ nhiều giai đoạn.

Ngoài phạm vi:
- Tầng ngắn hạn và script cắt phôi (phần B).
- Thêm giai đoạn mới với tập lớp khác (phần C).
- Dropdown chọn phiên bản trên giao diện LS.

## 2. Thuật ngữ và quy ước tên
| Thành phần | Nội dung | Kích thước |
|---|---|---|
| **base** | `baseline.pt` (M2-B), cố định, nhận diện bằng sha256 | 745 MB |
| **decoder** `dec-<yyyymmdd>-<hash6>` | Trọng số conv của `up1..up4` (cả hai nhánh net1/net2), lưu fp16 | ~49 MB |
| **adapter** `ad-<stage>-<yyyymmdd>-<hash6>` | Của một giai đoạn: γ/β, `running_mean`/`running_var` của mọi BatchNorm, và `pixel_head`/`topo_head` | ~0.3 MB |
| **bộ (bundle)** `tasdf-<yyyymmdd>-<hash6>` | base sha + một decoder + một adapter cho mỗi giai đoạn | chỉ là bản ghi |

`hash6` là 6 ký tự đầu của sha256 nội dung, nên **không bao giờ trùng hay dùng lại**.

`model_version` ghi vào LS = `<bundle>.<stage>`, ví dụ `tasdf-20261120-4be7a2.blastocyst`. Nhìn chuỗi này là biết chính xác backbone và adapter nào đã sinh ra prediction.

Bộ ban đầu `tasdf-base` = base, decoder gốc, adapter gốc trích từ base.

## 3. Lưu trữ
```
${TASDF_DATA_DIR}/
  baseline.pt
  stages.json                      # cấu hình giai đoạn (bên dưới)
  registry/
    registry.json                  # bundles, pointers, history, last_data_snapshot
    decoders/<dec-id>.pt
    adapters/<stage>/<ad-id>.pt
    jobs/<job-id>/                 # checkpoint để chạy tiếp, log; xoá sau khi job xong
    legacy/                        # state/36, state/37 cũ, chỉ để tra cứu
```

`stages.json`:
```json
{"blastocyst": {"classes": ["ZP", "TE", "ICM", "BC"], "aliases": {"Blastocoel": "BC"},
                "projects": [36, 37], "anchors_npz": "anchors.npz", "holdout_npz": "holdout.npz"}}
```
- Thay `LS_PROJECT_ID` bằng danh sách project lấy từ `stages.json`.
- Webhook và job chỉ chấp nhận các project có trong file này.

`registry.json` chứa:
- `bundles{}`: thành phần, metric theo giai đoạn, dữ liệu đã dùng (task id + digest theo project), job sinh ra bundle, thời điểm.
- `pointers{production, previous}`: dùng chung cho mọi giai đoạn, vì một bộ đã chứa adapter của mọi giai đoạn.
- `history[]`: ai, lúc nào, đổi pointer từ đâu sang đâu, lý do (cổng tự động, người vận hành, rollback).

Giữ lại: `production`, `previous` và **3 bộ gần nhất**. Decoder/adapter không còn bộ nào giữ thì bị xoá. Base không bao giờ bị xoá.

**Chuyển dữ liệu cũ** (chạy một lần lúc khởi động khi chưa có `registry/`):
- Tạo `tasdf-base` và đặt `production = tasdf-base`. Lý do: trên holdout đúng, `v8` cũ có mDice 0.503, kém base 0.519.
- Chuyển `state/36` và `state/37` vào `legacy/`, kèm ghi chú ánh xạ tên cũ (`tasdf-vN` của project P).
- Không còn route hay env nào dùng registry cũ.

## 4. Phục vụ dự đoán
- Lúc khởi động, nạp model của bộ `production`: base, đè decoder lên, rồi giữ sẵn adapter của mọi giai đoạn trên GPU.
- `/predict`:
  - Lấy project từ payload của LS (`project = "<id>.<ts>"`), tra ra giai đoạn qua `stages.json`, rồi nạp adapter đó vào model. Phép đổi chỉ chạm γ/β, thống kê BN và head, rất nhỏ; được thực hiện dưới lock.
  - Trả `model_version = <bundle>.<stage>`.
  - Project không có trong `stages.json` thì báo lỗi rõ ràng. Không lấy đại một giai đoạn nào.
- `/setup` trả `model_version` theo giai đoạn của project gọi tới.

## 5. Tầng dài hạn: job huấn luyện
**Kích hoạt:**
- Lịch: `TRAIN_CRON` dạng cron 5 trường (phút giờ ngày tháng thứ; hỗ trợ `*`, danh sách, khoảng, bước), mặc định `0 1 * * 0` (CN 01:00). Múi giờ theo `TZ`, mặc định `Asia/Ho_Chi_Minh` trong compose. Để trống thì tắt lịch. Một luồng hẹn giờ trong backend chạy job nếu có đủ dữ liệu mới.
- Người vận hành: `POST /train`. Trả 409 nếu đang có job. Có thể truyền `{"force": true}` để bỏ qua ngưỡng.
- Hằng tuần: chỉ chạy nếu có **≥ `TRAIN_MIN_NEW` (mặc định 10)** annotation mới hoặc vừa sửa so với `last_data_snapshot`, tính trên các project của giai đoạn. Chưa đủ thì ghi log và bỏ qua.
- Webhook **chỉ cập nhật bộ đếm** cho dashboard, không huấn luyện.

**Dữ liệu** (cho mỗi giai đoạn):
- Gom toàn bộ annotation đã duyệt từ các project của giai đoạn, dùng lại `trainer.select_samples` và luật nhãn `annotation_target`:
  - Bên trong ZP mà không vẽ thì gán 255.
  - Bỏ rectangle và keypoint.
  - `MIN_QUALITY_LEVEL ≥ 2`.
  - Mỗi task lấy annotation mới nhất.
- **Chia val theo ảnh:** `sha1(nội dung ảnh) mod 5 == 0` thì vào val (khoảng 20%). Cùng một ảnh ở hai project luôn rơi về cùng một phía.
- **Ảnh có nhiều bác sĩ vẽ:** giữ mọi annotation làm các mẫu học riêng. Ở val, chấm với từng người rồi lấy trung bình.
- **Anchor D10:** dùng `anchors_npz` của giai đoạn, chiếm ≤ 25% mỗi batch.

**Huấn luyện** (học nặng):
- Mỗi job **học tiếp từ bộ `production`** (warm start, người dùng chốt). Để bảo đảm tiến bộ là thật chứ không phải học vẹt:
  - Val chia theo ảnh và không bao giờ được học.
  - Cổng yêu cầu tăng vượt mức nhiễu (mục "Cổng").
  - Khoảng cách train − val bị giới hạn.
  - **Kiểm tra định kỳ:** mỗi `AUDIT_EVERY=4` job, học thêm một bản **từ `tasdf-base`** trên cùng dữ liệu. Nếu bản warm-start kém bản từ base quá `AUDIT_MARGIN=0.01` trên val thì thăng cấp bản từ base, với lý do `audit`. Cách này phát hiện trôi tích luỹ.
- **Phần được học:** conv của decoder (`up1..up4`, 24,4 triệu tham số, 39%), dùng chung cho mọi giai đoạn; cùng γ/β và head **riêng** của từng giai đoạn. Conv của encoder (37,7 triệu) đóng băng.
- **Mỗi bước:** micro-batch theo từng giai đoạn. Model nạp adapter của giai đoạn rồi forward, nên mỗi ảnh chỉ dạy đầu ra của giai đoạn của nó. Gradient được cộng dồn vào decoder chung.
- **Loss:** CE + soft-Dice có ignore 255, trên cả hai nhánh. Không dùng CPS/EMA.
- **Augmentation:** lật ngang/dọc, xoay 90°, độ sáng/tương phản ±15%.
- **Tối ưu:**
  - AdamW, lr `TRAIN_LR=3e-4`, giảm theo cosine.
  - Batch hiệu dụng 8: micro-batch 2 × cộng dồn 4. Có AMP.
  - Tối đa `TRAIN_EPOCHS=60`. Đã đo trên RTX 3060: 49 ms/ảnh/epoch, VRAM đỉnh 2,4 GB; 1000 ảnh × 60 epoch ≈ 50 phút.
- **Dừng sớm:** đo sau mỗi epoch, dừng nếu 5 epoch liên tiếp không tiến bộ. Thước đo là Dice của val giai đoạn nếu có ≥ `PROJECT_VAL_MIN=10` ảnh; nếu ít hơn thì dùng mDice trên D10/valid. Giữ checkpoint tốt nhất.
- **Hiệu chỉnh thống kê BN theo giai đoạn** (sau khi học xong): chạy forward không gradient trên ảnh train của từng giai đoạn để tính lại `running_mean`/`running_var` của adapter đó. Chỉ làm khi có ≥ `BN_RECAL_MIN=32` ảnh; ít hơn thì giữ thống kê cũ.
- **Giới hạn và chạy tiếp:**
  - Checkpoint mỗi epoch trong `jobs/<job-id>/`. Nếu container khởi động lại giữa chừng, job chạy tiếp từ epoch cuối.
  - Đạt `TRAIN_MAX_HOURS=2` thì dừng và dùng checkpoint tốt nhất tới lúc đó.
- **Hiệu chỉnh tham số trước khi bật lịch:** một lần chạy thử (`POST /train {"force": true, "dry_run": true}`) chạy 1 epoch rồi ghi thời gian mỗi epoch và VRAM đỉnh vào telemetry. Các giá trị `TRAIN_EPOCHS`/`TRAIN_MAX_HOURS` được chỉnh theo số đo, và số đo được ghi lại trong README.

**Cổng thăng cấp** (cho từng giai đoạn; bộ chỉ được thăng cấp khi **mọi** giai đoạn đều qua):
1. Không quên: `learner.gate` trên `holdout_npz` của giai đoạn (D10/valid đủ 4 lớp), so với production và với `tasdf-base`.
2. Tốt hơn thật trên val của giai đoạn, so với production:
   - Val ≥ 10 ảnh: Dice phải tăng ≥ max(`PROMOTE_MIN_GAIN=0.005`, 2 × độ lệch chuẩn bootstrap của chênh lệch Dice theo ảnh).
   - Val ít hơn: không giảm quá 0.01, và ghi `insufficient_val`.
3. Không học vẹt: Dice train − Dice val ≤ `MAX_TRAIN_VAL_GAP=0.15`.
4. Encoder không đổi (kiểm tra bằng `learner.layer_shifts`).

Qua cổng thì:
- `previous ← production`, `production ← bundle mới`, và ghi `history`.
- Đổi model phục vụ dưới lock.
- Với mỗi project của giai đoạn: PATCH `project.model_version = <bundle>.<stage>`. Tuỳ chọn: dự đoán sẵn `REFRESH_NEXT_N` task.

Không qua cổng: bộ được lưu với `status=rejected` và lý do. Các bộ bị loại không nằm trong 3 bộ gần nhất thì bị xoá theo luật giữ lại.

## 6. API và dashboard
Các route cần `X-OL-Token`, trừ `/train` (giữ mức tin cậy như `/webhook`, vì nút "Start Training" của LS không gửi token):
- `GET /registry`: liệt kê bundles, pointers, history, `last_data_snapshot`, số annotation mới theo giai đoạn, lần chạy kế tiếp.
- `POST /promote {"bundle": "..."}`: thăng cấp thủ công. Vẫn chạy cổng; muốn bỏ qua phải có `{"override": true}`, và lý do được ghi vào history.
- `POST /rollback`: `production ← previous`.
- `POST /train`: như mô tả ở mục 5.

Dashboard hiển thị:
- Bảng bundle (thành phần, metric theo giai đoạn, trạng thái, pointer).
- Tiến độ job đang chạy (epoch, loss, val).
- Lịch chạy kế tiếp và số annotation mới.
- Bỏ các phần của bản thử nghiệm cũ.

## 7. Thay đổi so với code hiện tại (working tree)
| Thành phần | Xử lý |
|---|---|
| `tasdf_online/schedule.py` (mới) | Bộ phân tích cron 5 trường tối giản và hàm tính lần chạy kế tiếp, không thêm dependency |
| `tasdf_online/registry.py` | Viết lại thành registry theo bộ ở mục 3; hàm chuyển dữ liệu từ `state/<p>` |
| `tasdf_online/trainer.py` | Giữ `select_samples`, `mean_dice_ignore`, `project_dice`; đổi `split_of` sang hash ảnh; thay `train` bằng vòng theo epoch nhiều giai đoạn (mục 5); thêm hiệu chỉnh BN |
| `tasdf_online/learner.py` | Thêm tập tham số "decoder + BN + heads", tách/nạp adapter, trích decoder; giữ `gate`, `evaluate`, `layer_shifts` |
| `tasdf_online/app.py` | Thay `LS_PROJECT_ID` bằng `stages.json`; chọn adapter theo project; hẹn giờ lịch chạy; `/registry`, `/promote`, `/rollback` mới; webhook chỉ đếm |
| `tasdf_online/config.py` | Bỏ `TRAIN_EVERY_N`/`TRAIN_STEPS`/`EVAL_EVERY`, `LS_PROJECT_ID`. Thêm `TRAIN_CRON`, `TRAIN_MIN_NEW`, `TRAIN_EPOCHS`, `TRAIN_LR`, `TRAIN_MAX_HOURS`, `MAX_TRAIN_VAL_GAP`, `BN_RECAL_MIN`, `AUDIT_EVERY`, `AUDIT_MARGIN`, `STAGES_FILE` |
| `ls_format.py`, `scripts/export_npz.py` | Giữ nguyên |
| Compose, env mẫu, README | Cập nhật biến; README mô tả nơi lưu model và cách thăng cấp/rollback |

## 8. Lỗi và vận hành
- Mỗi lúc chỉ chạy một job.
- Rollback hoặc promote trong lúc job đang chạy thì ứng viên của job đó không được tự thăng cấp, và lý do được ghi lại.
- LS không truy cập được, hết VRAM, hoặc không có dữ liệu: job thất bại, `last_error` được ghi vào telemetry, `last_data_snapshot` giữ nguyên để lần sau thử lại.
- `stages.json` sai (project trùng giữa hai giai đoạn, thiếu file npz): backend dừng ngay khi khởi động với thông báo rõ.

## 9. Kiểm thử
- **Unit:**
  - Tên bundle/decoder/adapter là duy nhất và đúng định dạng.
  - Lưu/nạp bộ khớp từng bit; pointer, history và luật giữ lại.
  - Chuyển dữ liệu từ `state/<p>`.
  - Ánh xạ project → giai đoạn, kể cả cấu hình sai.
  - Chia val theo hash ảnh.
  - Tính lần chạy kế tiếp từ `TRAIN_SCHEDULE`.
  - Hiệu chỉnh BN không đụng conv.
  - Tập tham số được học: encoder không đổi.
- **Tích hợp** (model nhỏ, LS giả):
  - Job chạy trọn và tạo bộ có metric theo giai đoạn.
  - Chạy tiếp từ checkpoint sau khi bị ngắt.
  - `TRAIN_MAX_HOURS` dừng đúng lúc.
  - Cổng loại thì production giữ nguyên.
  - `/predict` trả `<bundle>.<stage>` theo project.
  - `/rollback`, `/promote` và `/train` trả 409.
- **Thật:**
  - Chạy thử 1 epoch để đo thời gian/epoch và VRAM.
  - Một job thật trên project 36 + 37, báo cáo metric so với `tasdf-base`.

## 10. Các điểm đã chốt (2026-11-13)
| # | Điểm | Chốt |
|---|---|---|
| 1 | Chi phí trên RTX 3060 | Đã đo: 49 ms/ảnh/epoch, VRAM đỉnh 2,4 GB/6 GB. Mặc định `TRAIN_EPOCHS=60`, `TRAIN_MAX_HOURS=2` |
| 2 | Học có thắng base khi nhãn còn ít không | Người dùng tự gán thêm nhãn. `scripts/learning_curve.py` đo nhẹ / nặng-base / nặng-warm theo số nhãn sau mỗi đợt (báo cáo trong `.data/tasdf-online/reports/`) |
| 3 | Học từ base hay từ production | Từ **production**, có val không học, ngưỡng vượt nhiễu, giới hạn khoảng cách train − val, và kiểm tra định kỳ từ base |
| 4 | Hiệu chỉnh BN khi ít ảnh | Chỉ khi ≥ `BN_RECAL_MIN=32` ảnh |
| 5 | Ký tự trong `model_version` | Dùng `<bundle>.<stage>`. Validator của LS (`projects/serializers.py:207`) chỉ yêu cầu chuỗi trùng với version backend báo hoặc với prediction đã có |
| 6 | Lịch và múi giờ | `TRAIN_CRON` (cron 5 trường) cùng `TZ` (mặc định `Asia/Ho_Chi_Minh`) |
| 7 | Ngưỡng của cổng | Thêm biên vượt nhiễu bằng bootstrap (mục 5); các giá trị khác giữ như spec |
| 8 | Hai bác sĩ cùng một ảnh | Giữ như spec. Báo cáo bất đồng giữa bác sĩ để sang phần sau |
| 9 | Decoder fp16 | Lưu fp16 nếu nạp lại khớp fp32; nếu không thì lưu fp32 (có test) |
| 10 | `/train` ban ngày | Cho phép; job chỉ dùng tối đa khoảng 2,4 GB VRAM |
| 11 | v1–v8 cũ | Chỉ lưu vào `legacy/` |
| 12 | Mốc commit | Đã có mốc (`2f7efbada`, `111fdd773`). Từ nay mỗi đợt sửa là một commit theo Conventional Commits, đứng tên `ldhv-04` |
| 13 | Holdout/anchor cho giai đoạn khác | Để sang phần C |
