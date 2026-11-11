# Kế hoạch Khắc phục Lỗi ML Backend (`tasdf-online`) Không Kết Nối Được Tới Label Studio

## 1. Chẩn đoán Thực nghiệm & Nguyên nhân Gốc rễ (Empirically Verified Root Causes)

Qua kiểm tra trực tiếp trạng thái container (`docker ps -a`), log khởi động (`docker logs label-studio-tasdf-online-1`), biến môi trường (`docker inspect`) và quyền tập tin trên host (`ls -lan .data/tasdf-online`), hệ thống ghi nhận container `label-studio-tasdf-online-1` đang rơi vào vòng lặp **`Restarting (3)`** (Gunicorn worker chết ngay khi boot, nên cổng `9090` không mở và `MLBackend` id=8 trong Label Studio bị chuyển sang trạng thái `DI` - Disconnected).

Có **3 nguyên nhân nối tiếp nhau** gây ra lỗi này:

### Nguyên nhân 1 (Đang chặn trực tiếp lúc boot): Lệch UID giữa Container (`10001`) và thư mục Host (`1000`)
- **Bằng chứng thực nghiệm từ `docker logs label-studio-tasdf-online-1`**:
  ```text
  File "/app/tasdf_online/registry.py", line 19, in _atomic_write_bytes
    write(tmp)
  PermissionError: [Errno 13] Permission denied: '/data/tasdf-online/state/37/registry.json.tmp'
  [ERROR] Worker (pid:7) exited with code 3
  [ERROR] Reason: Worker failed to boot.
  ```
- **Cơ chế**:
  - Trong [`integrations/tasdf-online-learning/Dockerfile`](file:///home/terminator/workspace/Attech/label_tool/label-studio/integrations/tasdf-online-learning/Dockerfile#L21-L26), tiến trình chạy dưới `USER appuser` có **`UID=10001`**.
  - Trên máy host, thư mục bind-mount `.data/tasdf-online` và `.data/tasdf-online/state/37` thuộc sở hữu của user `terminator` (**`UID=1000, GID=1000`**) với quyền `drwxrwxr-x` (`0775`).
  - Khi bind-mount `.data/tasdf-online:/data/tasdf-online`, `UID 10001` không có quyền ghi vào `/data/tasdf-online/state/37/`, khiến `self.registry.sync_fingerprint(...)` ném `PermissionError` ngay khi khởi tạo `OnlineService`.

### Nguyên nhân 2 (Sẽ crash ngay sau khi sửa Nguyên nhân 1): Biến `ANCHOR_NPZ` trong `.data/tasdf-online/tasdf.env` trỏ tới đường dẫn tuyệt đối của máy Host
- **Bằng chứng thực nghiệm từ `docker inspect label-studio-tasdf-online-1`**:
  ```text
  ANCHOR_NPZ=/home/terminator/workspace/Attech/label_tool/label-studio/.data/tasdf-online/anchors.npz
  ```
- **Cơ chế**:
  - Tệp `.data/tasdf-online/tasdf.env` được tạo từ lần chạy trực tiếp trên host trước đây, nên chứa các đường dẫn tuyệt đối của máy host (`/home/terminator/workspace/...`) và `LS_URL=http://localhost:8080`.
  - Trong [`docker-compose.tasdf.yml`](file:///home/terminator/workspace/Attech/label_tool/label-studio/docker-compose.tasdf.yml#L18-L27), các biến `LS_URL`, `STATE_DIR`, `BASELINE_CKPT`, `HOLDOUT_NPZ` được khai báo lại trong mục `environment:` nên ghi đè được `tasdf.env`. Tuy nhiên, biến **`ANCHOR_NPZ`** đã bị bỏ khỏi `docker-compose.tasdf.yml` (ở commit `004320e8a`), khiến container nhận nguyên đường dẫn `/home/terminator/.../anchors.npz` từ `tasdf.env`.
  - Bên trong container, đường dẫn `/home/terminator/...` không tồn tại (vì được mount tại `/data/tasdf-online/anchors.npz`), nên dòng 90 của [`app.py`](file:///home/terminator/workspace/Attech/label_tool/label-studio/integrations/tasdf-online-learning/tasdf_online/app.py#L90) (`_load_npz(s.anchor_npz)`) sẽ ném `FileNotFoundError` ngay lập tức.

### Nguyên nhân 3: URL kết nối hai chiều giữa `label_studio` và `tasdf-online` trong mạng Docker
- Trong `.data/tasdf-online/tasdf.env`, `LS_URL=http://localhost:8080` nếu chạy độc lập trong container mà không có override sẽ trỏ vào chính loopback của container `tasdf-online` thay vì container `label_studio` (`http://webapp:8080`).
- Ngược lại, từ phía `label_studio` (nằm cùng mạng Docker `label-studio_default` với `tasdf-online` và có `host.docker.internal`), URL của ML Backend có thể dùng `http://tasdf-online:9090` (nội bộ mạng Docker, ổn định nhất) hoặc `http://host.docker.internal:9090` (qua cổng `9090:9090` publish ra host).

---

## 2. Phân tích Trade-Off 5 Chiều (Rule 2.7)

| Chiều đánh giá | Phương án A (Khuyến nghị): Đồng bộ `user: "${UID:-1000}:${GID:-1000}"` (hoặc UID 1000) + Chuẩn hóa đường dẫn `/data/tasdf-online/...` trong `tasdf.env` & `docker-compose.tasdf.yml` | Phương án B: Chạy `chmod -R 777 .data/tasdf-online` trên host |
| :--- | :--- | :--- |
| **1. System Resources** | Không phát sinh thêm tài nguyên CPU/RAM/I/O. | Không phát sinh tài nguyên. |
| **2. Complexity** | Rất thấp: chỉnh cấu hình Docker/env và quyền sở hữu chuẩn. | Thấp nhưng là cách vá tạm bợ. |
| **3. Long-Term Stability** | Cao: mọi file `registry.json` và `versions/vN.pt` do container sinh ra có cùng UID `1000:1000` với host user, không bao giờ tái phát lỗi `PermissionError`. | Thấp: các file/thư mục mới do container (`UID 10001`) tạo ra sau này sẽ có chủ sở hữu `10001`, gây xung đột quyền khi chạy script trên host. |
| **4. Extensibility** | Dễ dàng chuyển đổi giữa chạy Docker Compose và chạy local. | Kém. |
| **5. Maintainability** | Tường minh, tuân thủ chuẩn 12-Factor App (Rule 2.11). | Khó bảo trì và tiềm ẩn rủi ro bảo mật (`0777`). |

---

## 3. Proposed Changes

### Thành phần 1: Docker Compose & Cấu hình Môi trường (`tasdf-online`)

#### [MODIFY] [docker-compose.tasdf.yml](file:///home/terminator/workspace/Attech/label_tool/label-studio/docker-compose.tasdf.yml)
- Thêm `user: "${UID:-1000}:${GID:-1000}"` (hoặc đồng bộ UID trong `Dockerfile`) để tiến trình trong container chạy đúng quyền sở hữu của thư mục `.data/tasdf-online` trên host.
- Khai báo `ANCHOR_NPZ: /data/tasdf-online/anchors.npz` trong `environment:` để luôn dùng đúng đường dẫn bên trong container.

```diff
 services:
   tasdf-online:
     build:
       context: ./integrations/tasdf-online-learning
       args:
         TORCH_INDEX_URL: ${TASDF_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
+    user: "${UID:-1000}:${GID:-1000}"
     restart: unless-stopped
     depends_on:
       - webapp
     env_file:
       - path: ./.data/tasdf-online/tasdf.env
         required: false
     environment:
       DEVICE: ${TASDF_DEVICE:-cpu}
       TORCH_THREADS: ${TASDF_TORCH_THREADS:-4}
       LS_URL: http://webapp:8080
       STATE_DIR: /data/tasdf-online/state
       BASELINE_CKPT: /data/tasdf-online/baseline.pt
       HOLDOUT_NPZ: /data/tasdf-online/holdout.npz
+      ANCHOR_NPZ: /data/tasdf-online/anchors.npz
       # CPU-friendly defaults; raise on GPU
       EPISODE_EPOCHS: ${TASDF_EPISODE_EPOCHS:-1}
```

#### [MODIFY] [.data/tasdf-online/tasdf.env](file:///home/terminator/workspace/Attech/label_tool/label-studio/.data/tasdf-online/tasdf.env)
- Cập nhật các đường dẫn trong `.data/tasdf-online/tasdf.env` sang đường dẫn chuẩn của container (`/data/tasdf-online/...`) và `LS_URL=http://webapp:8080` để không bị xung đột.

---

### Thành phần 2: Đồng bộ trạng thái kết nối ML Backend trong Label Studio
- Khởi động lại service `tasdf-online` (`docker compose -f docker-compose.yml -f docker-compose.tasdf.yml up -d tasdf-online`).
- Gọi `update_state()` trên các bản ghi `MLBackend` của dự án để chuyển trạng thái từ `DI` (Disconnected) sang `CO` (Connected).

---

## 4. Verification Plan

### Kiểm tra Tự động (Automated Verification)
1. Kiểm tra trạng thái container `label-studio-tasdf-online-1` đạt `Up (healthy)` và phản hồi `GET http://localhost:9090/health` trả về `{"status": "UP", ...}`.
2. Kiểm tra từ bên trong container `label_studio` gọi sang `tasdf-online` và xác nhận `MLBackend.objects.get(id=8).state == 'CO'`.
3. Kiểm tra từ bên trong container `label-studio-tasdf-online-1` gọi ngược về `http://webapp:8080/api/projects/37` với `LS_TOKEN` trả về `200 OK`.
