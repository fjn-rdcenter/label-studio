# TASDF backend: huấn luyện theo đợt, đo được trước khi học

Ngày: 2026-11-12. Trạng thái: đã duyệt qua trao đổi, đang triển khai thử nghiệm.
Bối cảnh và bằng chứng: báo cáo đối chiếu online learning (E1–E9). Tóm tắt:
- Học từng annotation không cải thiện ảnh của dự án.
- Holdout/anchor xuất nhầm từ D10 *test*: không có BC, TE gộp cả khoang phôi.
- Nhãn thiếu lớp bị dạy thành nền.

## Mục tiêu
- Model chỉ thăng cấp khi **đo được** rằng nó không quên (D10) và không kém đi trên chính dự án (project-val).
- Huấn luyện có giám sát theo đợt trên toàn bộ annotation hợp lệ của dự án. Kích hoạt khi đủ N nhãn mới hoặc khi gọi `POST /train`.

Ngoài phạm vi: học online cho timelapse; thay đổi frontend; nhiều project trên một backend.

## Kiến trúc (phương án A: mở rộng backend hiện có)
```
/webhook (ANNOTATION_CREATED/UPDATED, đúng project) ─┐  luồng nền, không chặn webhook
POST /train (no token, like /webhook) ───────────────┤
                                                      ▼
TrainJob (train_lock, một job mỗi lúc)
 1. Kéo toàn bộ task + annotation của LS_PROJECT_ID
 2. Mỗi task: annotation mới nhất, không huỷ, quality >= MIN_QUALITY_LEVEL (mặc định 2)
    → nhãn 256x256 theo luật ignore (bên dưới)
 3. Chia cố định: sha1("<project>:<task_id>") % 5 == 0 → project-val, còn lại → train
 4. Đếm task train có digest khác lần job trước; < TRAIN_EVERY_N và không bị ép → dừng
 5. Từ v0: select_trainable (norm_and_heads), Adam lr=LR, CE + soft-Dice (ignore 255), không CPS/EMA
    mỗi bước: micro-batch 2 từ train dự án + anchor D10/train (số anchor <= số mẫu dự án), AMP
    cứ EVAL_EVERY bước đo; giữ checkpoint tốt nhất (project-val nếu >= PROJECT_VAL_MIN ảnh, nếu không thì D10)
 6. Cổng: (a) learner.gate trên D10/valid so với active (đo lại) và v0
          (b) project-val >= PROJECT_VAL_MIN: Dice >= active + PROMOTE_MIN_GAIN
              project-val <  PROJECT_VAL_MIN: Dice >= active - 0.01 (ghi insufficient_project_val)
 7. Registry: vN, metric D10 + project-val, task train/val, lý do cổng; last_job.digests bền vững
 8. Thăng cấp → đổi teacher dưới lock → refresh_ls
```

## Luật nhãn
- Chỉ nhận `polygonlabels`, `ellipsebasedpolygonlabels`, `brushlabels`. Loại `rectanglelabels` và `keypointlabels`.
- Pixel được vẽ: lấy lớp đó (thứ tự painter ZP, TE, BC, ICM).
- Có ZP: pixel ngoài vùng ZP (vùng lấp kín, bỏ lỗ) là nền 0; pixel trong ZP mà không được vẽ là 255.
- Không có ZP: mọi pixel không được vẽ là 255.
- Annotation không có pixel lớp nào được bỏ qua.

## Dữ liệu D10
`scripts/export_npz.py` đọc trực tiếp `D10/<split>/{images,masks}` và resize về 256 (ảnh nội suy tuyến tính, mask nội suy nearest). Script từ chối nếu tập mask thiếu lớp nào trong 1–4.
- `holdout.npz` lấy từ D10/valid (8 ảnh).
- `anchors.npz` lấy từ D10/train (32 mask).

## Cấu hình
- Thêm: `TRAIN_EVERY_N=10`, `TRAIN_STEPS=200`, `LR=3e-4`, `EVAL_EVERY=20`, `PROJECT_VAL_MIN=10`, `PROMOTE_MIN_GAIN=0.005`.
- Bỏ: `EPISODE_EPOCHS`, `REPLAY_SIZE`, `EMA_ALPHA`, `CORR_BETA`, `TOPO_SUP_WEIGHT`.
- Đổi: `MIN_QUALITY_LEVEL` = 2.

## Gỡ bỏ
Episode từng mẫu (`_episode`, queue, `replay`, `last_predicted_mask`, `local_probe`, `run_episode`, `episode_loss`), cùng ảnh "vừa sửa" trên dashboard.

## Lỗi
- `/train` khi đang có job: trả 409. `/train` không đòi `X-OL-Token`, vì nút "Start Training" của LS không gửi header này. Mức tin cậy giống `/webhook` (mạng nội bộ); dữ liệu luôn được đọc lại từ LS bằng token của backend.
- Job lỗi (LS, VRAM, dữ liệu rỗng): ghi `last_error`, không thăng cấp, không cập nhật `last_job`.
- Rollback trong lúc job chạy: ứng viên bị loại.

## Kiểm thử
- **Unit:**
  - Luật nhãn (ngoài ZP, trong ZP, không ZP, loại rectangle).
  - Chọn annotation (mới nhất, quality, huỷ).
  - Chia tập ổn định.
  - Dice có ignore.
  - Bộ xuất D10 từ chối mask thiếu lớp.
  - Hai chế độ của cổng dự án.
- **Tích hợp** (model nhỏ, LS giả):
  - `/train` tạo phiên bản có metric hai tập.
  - Gọi `/train` lần hai khi đang chạy trả 409.
  - Ngưỡng N.
  - `last_job` còn sau khi khởi động lại.
- **Đo thật:**
  - Một job trên dữ liệu hiện có, so với v0: D10/valid có BC ≠ 0, và đo trên project-val.
  - Chỉ kết luận "học được" khi project-val ≥ 10 ảnh và có ≥ 30 annotation đủ lớp.
