## Development guide line
### Build label studio image
Việc build lại sẽ đóng gói code của Django và build lại code của front end trong ứng dụng đảm bảo cho việc triển khai.
Câu lệnh build image như sau:

```bash
docker build -t label-studio:1.13.1.20241028.1.fjs -f Dockerfile --platform=linux/amd64 .
```

Sau khi build image xong thì có thể mang đi triển khai theo setting tại `file docker-compose.yml`
Chỉ cần chọn đúng image tag vừa build sau đó chạy docker compose là có thể dev local.

### Chỉnh sửa code liên quan đến front và các tính năng
Fujinet đã thực hiện chỉnh sửa, custom nhiều tính năng liên quan đến việc đánh label có thể kể đến như custom template, custom thanh slider hiển thị, bật fflag,... Còn có nhiều tính năng có thể bị tắt do fflag mặc định của ứng dụng cần cái nào thì bật lên là được, tham khảo trong file `file docker-compose.yml`

Để test và lập trình chỉnh sửa các tính năng mới cần làm như sau:

Chạy label studio image trước để có server django cung cấp API và serving image.

Cd vào thưu mục web, sau đó chạy câu lệnh 
```bash
yarn lfs:serve
```
Sau khi chạy xong sex có giao diện của phần test mặc định của label studio, các test này được lấy từ ```web\libs\editor\src\examples```

Tiến hành chỉnh sửa file ```web\libs\editor\src\examples\image_polygons\config.xml```
để có được chế độ đánh label mong muốn và test các tính năng.

Sau khi hoàn tất việc dev và test tính năng thì tiến hành build lại và deploy docker image như trên để hoàn tất quá trình dev frontend.
