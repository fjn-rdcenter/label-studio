# Hướng dẫn cài đặt Label Studio trong mạng FJS

Các bước dưới đây theo đúng thứ tự đã build và chạy thành công ngày 2026-11-17, sau proxy công ty (`proxy.fujinet.vn:8080`). Proxy này giải mã HTTPS bằng cert nội bộ, nên mọi công cụ tải gói (yarn, pip, poetry, curl) phải được cấu hình để tin cert FJS.

Kết quả: 3 container `label_studio` (cổng 8080), `ls-db` (Postgres 14) và `label-studio-opencv-segmentation-1`.

---

## 0. Yêu cầu

- Docker Desktop (Windows, chạy WSL2), đã đăng nhập mạng công ty.
- Repo đã được clone, terminal mở tại thư mục gốc của repo.
- File cert proxy công ty: `FJS_SSL_Certificates.crt`, dạng DER hoặc PEM đều được.

## 1. Đặt cert vào thư mục `certs/`

```
label-studio/
└── certs/
    └── FJS_SSL_Certificates.crt
```

- `.dockerignore` đã có dòng `!certs/**`. Nếu thiếu dòng này, Docker không thấy file cert và báo lỗi ở bước `COPY ./certs/...`.
- Cert dạng DER sẽ được tự đổi sang PEM trong lúc build.
- Thư mục `certs/` chỉ dùng nội bộ, **không commit lên git**.

## 2. Kiểm tra cấu hình proxy trong `Dockerfile`

Stage `frontend-builder` (khoảng dòng 12–17) đặt proxy cho yarn qua các biến `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`:

```dockerfile
ENV http_proxy=http://<user>:<mật khẩu>@proxy.fujinet.vn:8080
```

- Thay `<user>:<mật khẩu>` bằng tài khoản proxy của bạn. Mật khẩu có ký tự đặc biệt thì mã hóa URL, ví dụ `@` thành `%40`.
- **Không commit mật khẩu thật lên git.**

## 3. Đặt proxy trong Docker Desktop là **System**

Docker Desktop → **Settings** → **Resources** → **Proxies**:

- **Tắt** *Manual proxy configuration*, để Docker dùng proxy hệ thống của Windows.
- Không dùng Manual với `http://proxy.fujinet.vn:8080` không kèm tài khoản: proxy sẽ trả 403.
- Docker Desktop tự xóa `user:pass` nếu bạn nhập vào ô proxy, nên cách đó cũng không dùng được.

Bấm **Apply & restart** nếu có thay đổi.

> Trong `docker info` sẽ luôn thấy `HTTP Proxy: http.docker.internal:3128`. Đây là proxy nội bộ của Docker Desktop, nó chuyển tiếp tiếp tới proxy công ty, là bình thường.

## 4. Tải sẵn các base image (bắt buộc)

Trong mạng công ty, **`docker pull` tải được**, nhưng **khi build lại bị 403** lúc tự tra cứu image (`failed to fetch anonymous token ... Forbidden`). Vì vậy cần tải sẵn mọi base image trước khi build:

```bash
docker pull docker/dockerfile:1
docker pull node:18
docker pull ubuntu:22.04
docker pull python:3.10-slim
docker pull postgres:14
```

Chạy lại bước này mỗi khi dọn image (`docker system prune`) hoặc khi build lại báo 403.

## 5. Build và chạy

```bash
docker compose up -d --build
```

Lần đầu mất khoảng 10–20 phút: `apt-get`, `yarn install`, `yarn build` và `poetry install` đều chậm. Các lần sau dùng cache nên nhanh hơn nhiều.

Nếu đã build `webapp` trước đó và chỉ muốn chạy lại:

```bash
docker compose up -d
```

## 6. Kiểm tra

```bash
docker compose ps
```

Cả 3 container phải ở trạng thái `Up`, riêng opencv-segmentation phải `(healthy)`.

```bash
curl --noproxy "*" http://localhost:8080/health
```

Phải trả về `{"status": "UP"}` (HTTP 200). Sau đó mở http://localhost:8080 và đăng nhập bằng `LABEL_STUDIO_USERNAME` / `LABEL_STUDIO_PASSWORD` trong `docker-compose.yml`.

Dữ liệu nằm ở `.data/` (Postgres, dữ liệu app, ảnh), được mount từ máy vào nên không mất khi xóa container.

## 7. (Tùy chọn) ML backend TASDF online-learning

Service `tasdf-online` nằm trong file overlay `docker-compose.tasdf.yml`. Nó cũng cần cert FJS và base image `python:3.10-slim` đã tải sẵn ở bước 4.

### 7.1. Build image

```bash
docker pull python:3.10-slim
docker compose -f docker-compose.yml -f docker-compose.tasdf.yml build tasdf-online
```

Bước này tải PyTorch CPU (vài trăm MB) từ `download.pytorch.org`, nên lần đầu mất khá lâu.

### 7.2. Chuẩn bị dữ liệu trước khi chạy

Compose **không tự tạo** thư mục dữ liệu. Nếu thiếu, `up` sẽ lỗi ngay:

```bash
cp integrations/tasdf-online-learning/compose.env.example .env
mkdir -p .data/tasdf-online
cp integrations/tasdf-online-learning/tasdf.env.example .data/tasdf-online/tasdf.env
cp integrations/tasdf-online-learning/stages.example.json .data/tasdf-online/stages.json
```

- `.env`: sửa các biến `TASDF_*` theo máy (khối cuối file).
- `.data/tasdf-online/tasdf.env`: điền `LS_TOKEN` (token API của Label Studio) và các secret khác. **Không commit.**
- `.data/tasdf-online/stages.json`: khai báo giai đoạn ứng với id project LS.
- Đặt `baseline.pt` và `holdout.npz` (bắt buộc), cùng `anchors.npz` (nếu có) vào `.data/tasdf-online/`. Muốn chạy không có `holdout.npz` thì đặt `TASDF_HOLDOUT_NPZ=` (để trống) trong `.env`.

Chi tiết từng biến xem `integrations/tasdf-online-learning/README.md`.

### 7.3. Chạy và kiểm tra

```bash
docker compose -f docker-compose.yml -f docker-compose.tasdf.yml up -d
curl --noproxy "*" http://localhost:9090/health
```

Bản GPU thêm `-f docker-compose.tasdf-gpu.yml` (cần NVIDIA Container Toolkit), và base image vẫn là `python:3.10-slim`.

---

## Cấu hình cert đã thêm (để tham khảo)

| File | Thay đổi | Lý do |
|---|---|---|
| `.dockerignore` | Thêm `!certs/**` | Cho thư mục cert vào build context |
| `Dockerfile` (stage frontend) | Cài cert FJS, `NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt` | Node/yarn dùng bộ CA riêng, bỏ qua kho cert của hệ thống |
| `Dockerfile` (stage Python) | `COPY --from=frontend-builder` cert đã đổi sang PEM, `update-ca-certificates`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, `PIP_CERT` | pip/poetry tin cert FJS khi tải từ PyPI |
| `docker-compose.yml` (opencv-segmentation) | `additional_contexts: certs: ./certs` | Build context của service này không chứa `certs/` |
| `integrations/opencv-segmentation/Dockerfile` | `COPY --from=certs`, đổi DER sang PEM, `update-ca-certificates`, `PIP_CERT`, `REQUESTS_CA_BUNDLE` | pip tin cert FJS |
| `docker-compose.tasdf.yml` | `additional_contexts: certs: ./certs` | Như opencv-segmentation |
| `integrations/tasdf-online-learning/Dockerfile` | Giống Dockerfile của opencv-segmentation | pip tin cert FJS khi tải PyPI và PyTorch |

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `COPY ./certs/FJS_SSL_Certificates.crt`: not found | Thiếu file, hoặc `.dockerignore` thiếu `!certs/**` | Bước 1 |
| `yarn install`: `self-signed certificate in certificate chain` | Node không tin cert FJS | Kiểm tra `NODE_EXTRA_CA_CERTS` trong `Dockerfile` |
| `failed to fetch anonymous token ... Forbidden` | Bước build tự tra cứu image bị proxy chặn | Bước 4: `docker pull` image được nêu trong lỗi rồi build lại |
| `pip`: `CERTIFICATE_VERIFY_FAILED` | Stage hoặc service đó chưa cài cert FJS | Xem bảng cấu hình cert ở trên |
| `failed to connect to the docker API ... docker_engine` | Docker Desktop đang tắt | `docker desktop start` |
| `curl` trên Windows: `CRYPT_E_NO_REVOCATION_CHECK` | Windows không kiểm tra được trạng thái thu hồi của cert proxy | Thêm `--ssl-no-revoke` (chỉ ảnh hưởng `curl.exe`, không ảnh hưởng Docker) |
