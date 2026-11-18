# TASDF tầng ngắn hạn: adapter theo well và lan truyền đề xuất

Ngày: 2026-11-19. Trạng thái: đã chốt qua trao đổi (người dùng chốt toàn bộ thông tin triển khai adapter ngày 2026-11-19).
Nền tảng: `2026-11-13-tasdf-stage-adapters-long-term-design.md` (tầng dài hạn, registry bundle, cổng thăng cấp; giữ nguyên).
Bằng chứng tham khảo: `~/workspace/Attech/Research/reports/Lan truyền nhãn phôi timelapse.md`.

## 1. Mục đích
Bác sĩ chỉ sửa và submit. Nỗi đau lớn nhất là vẽ lại cùng một vật thể trên nhiều frame. Khi bác sĩ xác nhận xong một ảnh, hệ thống mang các vật thể của ảnh đó sang N ảnh kế tiếp, và một adapter riêng của well nắn biên cho khớp ảnh mới. Thành công: thời gian và độ dài biên phải sửa trên mỗi ảnh giảm; đề xuất không bao giờ chứa vùng rác.

## 2. Phạm vi
- Dữ liệu timelapse: 1 task = 1 well; frame = trang Repeater `timelapse_{i}`; slice = `item_index` trong `valueList`.
- Lan sang **các frame phía sau**, cùng slice với ảnh vừa xác nhận. Không lan ngược.
- Mọi nhãn Object được mang sang. Chỉ nhãn model biết (TASDF: ZP, TE, ICM, BC/Blastocoel, dạng `polygonlabels`) được nắn biên; nhãn khác (PN, Blastomere, rect, ellipse, keypoint…) được copy nguyên.
- Ngoài phạm vi: Event, Clinical, bộ định tuyến nhiều model theo ngày, học phong cách từng bác sĩ, model lan truyền học được (giai đoạn C), che nguồn đề xuất trong UI, gán đôi.

## 3. Thao tác của bác sĩ (cách a)
- Nút **"Xong ảnh"** trên thanh công cụ, phím tắt **Shift+Enter**. Một thao tác vừa chốt ảnh hiện tại là nhãn vàng (kể cả khi chấp nhận nguyên đề xuất), vừa cho adapter học, vừa lan đề xuất.
- Ô tuỳ chọn cạnh nút, lưu theo trình duyệt của từng bác sĩ: **số ảnh lan N** (mặc định 3, 0–10), **ngưỡng dừng** (mặc định 0.8), **tự lan** (bật). N = 0 hoặc tắt tự lan: chỉ học, không lan.
- Ảnh chỉ lướt qua không bao giờ thành nhãn.

## 4. Bốn luật đề xuất
1. **Chỉ những vật thể bác sĩ đã vẽ.** Mỗi vùng của ảnh nguồn là một vật thể, giữ nguyên id gốc qua các frame; model không sinh vùng mới. Vật thể biến mất thì dừng chuỗi.
2. **Chỉ nắn biên.** Kết quả = lõi của mask mang sang (co `BAND_PX`) ∪ (mask model ∩ vùng nở `BAND_PX`), giữ thành phần liên thông lớn nhất chạm mask mang sang; ZP/TE giữ một lỗ (lỗ lớn nhất). Không ghi đè frame đã có vùng của bác sĩ.
3. **Thay, không chồng.** Đề xuất mới chỉ thay vùng `origin = "prediction"` chưa ai đụng trên ảnh đích. Ảnh đích có vùng `manual` hoặc `prediction-changed` là mốc: dừng chuỗi, không gọi model.
4. **Dừng đúng lúc.** Với mỗi vật thể được nắn: dừng nếu độ khớp biên (boundary F, dung sai 2 px ở 256) giữa mask mang sang và dự đoán tự do của model < ngưỡng; hoặc diện tích sau nắn / trước nằm ngoài [0.67, 1.5]; hoặc ZP/TE mất lỗ; hoặc vật thể rỗng. Khi dừng: trả hình mang sang (copy) cho ảnh đó kèm lý do, frontend báo và ngắt chuỗi.

## 5. Adapter theo well
- **Là gì:** BN affine + heads (`bundle_registry.is_adapter_key`), khởi tạo từ adapter production của stage. Backbone production dùng chung, đóng băng. Thống kê chạy BN đóng băng (`model.eval()`), chỉ học γ/β.
- **Sống:** trong RAM theo khoá (project, task id); tối đa 16 well, bỏ well không dùng quá 2 giờ; xoá tất cả khi production đổi.
- **Mỗi lần "Xong ảnh":** thêm mẫu (crop ảnh, nhãn vàng, mặt nạ pixel đã sửa so với đề xuất đã hiện, dự đoán argmax của production) vào bộ nhớ của well (giữ 32 mẫu gần nhất), rồi học `ADAPT_STEPS=3` bước AdamW `ADAPT_LR=1e-4` trên toàn bộ bộ nhớ, có lật/xoay.
- **Loss:** CE + soft-Dice trên pixel có nhãn, trọng số pixel = 1 + 2·(pixel bác sĩ đã sửa); + 0.1 × CE tới argmax production trên pixel IGNORE; + 1e-3 × L2 giữa adapter và adapter production. Cả net1 và net2.
- **Rollback:** sau khi học, nếu Dice trung bình trên bộ nhớ thấp hơn trước khi học quá 0.01 thì bỏ lần học đó.
- **Cổng liên tục:** mỗi đề xuất được tính cho ba nguồn: `copy`, `production` (nắn bằng production), `adapter` (nắn bằng adapter của well). Khi bác sĩ xác nhận ảnh đó, chấm Dice của cả ba so với nhãn vàng. Nguồn dùng cho đề xuất tiếp theo: `production` cho tới khi có 3 ảnh được chấm; sau đó nguồn có Dice trung bình cao nhất trên 5 ảnh chấm gần nhất. Đây là luật "B phải thắng A, A phải thắng copy", chạy theo từng well.
- **GPU:** adapter được ưu tiên: khi adapter đang học hoặc dự đoán, job dài hạn tạm dừng giữa các bước.

## 6. Dữ liệu (dataloader)
- Ảnh gốc 964×1288; crop **600×600** tâm ở tâm hộp bao các vùng của ảnh nguồn (kẹp trong ảnh; ảnh nhỏ hơn thì lấy cả ảnh), rồi resize 256 cho model. Toạ độ đề xuất được đổi ngược về % của ảnh gốc.
- Nhật ký: mỗi lần "Xong ảnh" và mỗi đề xuất ghi một dòng JSONL vào `${JOURNAL_DIR}/<project>/<task>.jsonl` (thời điểm, ảnh, khoảng cách tới ảnh vàng gần nhất, nguồn dùng, điểm ba nguồn, lý do dừng, thống kê học). Mỗi "Xong ảnh" lưu kèm `.npz` (crop box, nhãn vàng 256, đề xuất ba nguồn 256) để phát lại offline và làm dữ liệu cho giai đoạn C.
- Tầng dài hạn không đổi: vẫn chỉ học nhãn đã duyệt (quality ≥ 2) đọc lại từ LS. Adapter học từ ảnh chưa duyệt nhưng chỉ dùng để đề xuất trong well đó.

## 7. Đường gọi
Editor → LS `POST /api/ml/{pk}/predict/test` với `{task, image, context}` → TASDF `POST /predict` (context chuyển nguyên) → trả `results[0]` kèm `propagation` → LS trả `{data, propagation}`. LS kiểm `context.gold.url` thuộc task như đã kiểm `image`. Mỗi đề xuất tạo một Prediction (như Smart Prediction hiện tại).

## 8. Đã quyết bởi người viết (có thể chỉnh khi đo)
`BAND_PX=8` (ở 256), ngưỡng diện tích [0.67, 1.5], dung sai biên 2 px, cửa sổ cổng 3/5, `ADAPT_LR=1e-4`, trọng số loss 2 / 0.1 / 1e-3, 16 well / 2 giờ, phím Shift+Enter.
