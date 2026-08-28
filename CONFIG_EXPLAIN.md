# CONFIG_EXPLAIN.md

## Giám sát Tư thế Ngồi — Tài liệu Cấu hình & Thuật toán

Tài liệu này giải thích mọi thông số có thể tùy chỉnh của ứng dụng Streamlit (`app.py`) và
các khái niệm xử lý tín hiệu đứng sau ba chỉ số tư thế được đo. Nội dung được đồng bộ với
phần triển khai hiện tại (Tỷ lệ Phối cảnh 2D cho Forward Lean, góc đường nối hai tai cho
Lateral Tilt, bộ lọc trung bình động và debounce).

### Mục lục
- 1. Tổng quan xử lý (Pipeline Overview)
- 2. Ba chỉ số tư thế
  - 2.1 Forward Lean (Cổ rùa) — Tỷ lệ Phối cảnh 2D
  - 2.2 Lateral Tilt (Vẹo người / Ngoẹo cổ) — Góc đường nối hai tai
  - 2.3 Shoulder Imbalance (Lệch vai) — Góc đường vai
- 3. Bộ lọc trung bình động (bộ đệm 10 frame)
- 4. Hiệu chuẩn (Baseline Calibration)
- 5. Ngưỡng cảnh báo (Thresholds)
- 6. Số frame chống rung (Debounce Frames)
- 7. Điểm tư thế (0-100)
- 8. Thông số mặc định khuyến nghị
- 9. Ví dụ cấu hình JSON

---

## 1. Tổng quan xử lý

Ứng dụng thực hiện các bước sau cho mỗi khung hình camera:

1. **Thu khung hình** — OpenCV (`cv2.VideoCapture`, DirectShow backend trên Windows),
   640x480, hình ảnh được lật (mirrored).
2. **Phát hiện tư thế** — MediaPipe Pose trả về 33 điểm mốc (landmark) với tọa độ chuẩn hóa
   `(nx, ny, nz)` và tọa độ pixel `(x, y)`.
3. **Tính chỉ số thô** — `compute_posture_metrics_3d()` tính ra ba giá trị thô.
4. **Bộ lọc trung bình động** — giá trị thô được đưa vào ba bộ đệm tròn (`deque`,
   `maxlen=10`); giá trị trung bình cộng của mỗi bộ đệm được dùng cho các bước tiếp theo.
5. **Hiệu chuẩn / Baseline** — lấy giá trị đã làm mịn trừ đi baseline đã lưu để có `deviations`.
6. **So sánh ngưỡng** — các độ lệch được kiểm tra với ba thanh trượt.
7. **Debounce** — bộ đếm tăng lên ở mỗi frame vi phạm và đặt về 0 nếu không vi phạm.
   Khi bộ đếm đạt giới hạn `debounce_limit`, cờ `alert_active` chuyển thành `True`.

Ứng dụng hiển thị camera trực tiếp, các dòng chú thích (`Forward Ratio`, `Lateral Tilt`,
`Shoulder Angle`), điểm tư thế và các thông báo cảnh báo/trạng thái.

---

## 2. Ba chỉ số tư thế

### 2.1 Forward Lean (Cổ rùa) — Tỷ lệ Phối cảnh 2D

Giá trị `z` của MediaPipe có biên độ rất nhỏ và nhiễu, vì vậy độ nhô đầu về phía trước được
đo bằng **tỷ lệ phối cảnh 2D**:

```
forward_ratio = face_width / shoulder_width
```

- `face_width` = khoảng cách **Euclidean** giữa **Tai Trái và Tai Phải** (tọa độ pixel x, y).
  Dùng **Mắt Trái / Mắt Phải** thay thế khi tai bị che / không phát hiện được.
- `shoulder_width` = khoảng cách **Euclidean** giữa **Vai Trái và Vai Phải** (tọa độ pixel x, y).
- Nếu `shoulder_width == 0`, tỷ lệ được coi là `0.0`.

**Tại sao hiệu quả:** khi đầu chồm về phía camera, khuôn mặt trông to hơn trong 2D trong khi
vai gần như không đổi, nên tỷ lệ **tăng lên**. Tỷ lệ này bất biến với tỉ lệ thu phóng:
khoảng cách/zoom tác động đều cả hai đại lượng sẽ không làm đổi tỷ lệ.

**Quy tắc cảnh báo:** `deviation = current_ratio - baseline_ratio`.
Cảnh báo chỉ kích hoạt khi `deviation > forward_thresh` (tỷ lệ tăng; ngả đầu ra sau sẽ
không cảnh báo).

### 2.2 Lateral Tilt (Vẹo người / Ngoẹo cổ) — Góc đường nối hai tai

Đo theo góc xoay của khuôn mặt thay cho vector mũi–ngực (phương pháp cũ thất bại khi mũi
vẫn thẳng hàng trục dọc với ngực nhưng đầu nghiêng sang một bên):

```
lateral_deg = degrees( atan2(dy, dx) )
d = ( right_ref - left_ref )   # Tai Phải - Tai Trái (fallback: Mắt Phải - Mắt Trái)
```

- Dùng tọa độ chuẩn hóa `(nx, ny)`.
- Khoảng **0°** khi đầu thẳng; dương/âm khi đầu nghiêng sang hai bên.
- Phát hiện ngoẹo đầu/cổ ngay cả khi mũi vẫn thẳng đứng trên đường vai.

**Quyết định cảnh báo:** `|deviation| > lateral_thresh`.

### 2.3 Shoulder Imbalance (Lệch vai) — Góc đường vai

```
shoulder_deg = | degrees(atan2(sdy, sdx)) |
s = vai_phải - vai_trái
```

- Dùng tọa độ chuẩn hóa; **0°** = hai vai ngang bằng, lớn hơn = một bên cao hơn.
- `abs()` làm mất dấu, chỉ quan tâm độ lớn.

**Cảnh báo:** `|deviation| > shoulder_thresh`.

---

## 3. Bộ lọc trung bình động (10 frame) + EMA (α = 0.3)

### 3a. Trung bình động (Moving Average, bộ đệm 10 frame)

- Ba deque được lưu trong `st.session_state`: `lean_buffer`, `tilt_buffer`, `imbalance_buffer`.
- Mỗi deque có `maxlen=10` — chỉ giữ 10 giá trị thô gần nhất (giá trị cũ tự động bị loại).
- Mỗi frame mới, giá trị thô được **thêm vào** (append), sau đó lấy *trung bình cộng* của bộ đệm.

### 3b. Lọc mũ EMA (Exponential Moving Average, α = 0.3)

- Trạng thái EMA lưu trong `st.session_state.ema_smoother` (lớp `EMASmoother`, giữ
  `prev_values` — giá trị làm mịn của frame trước cho từng đại lượng).
- Công thức: `Value_smooth = α * Value_current + (1 - α) * Value_previous`, với `EMA_ALPHA = 0.3`
  (đáp ứng nhanh nhưng không rung; hạ α xuống để mượt hơn, nâng lên để phản hồi nhanh hơn).
- **Frame đầu tiên hoặc ngay sau khi nhấn "Calibrate"** (và sau khi Tắt/Bật lại camera):
  EMA bị reset và gán trực tiếp `Value_smooth = Value_current` (không trộn với dữ liệu cũ).
- Hai đại lượng góc (`lateral_tilt`, `shoulder_imbalance`) được làm mịn **an toàn khi vắt biên ±180°**:
  dùng hiệu góc ngắn nhất có dấu thay vì trừ trực tiếp, tránh lỗi khi góc `atan2` vắt qua
  biên +180/-180 (ví dụ 179° và -179° chỉ cách nhau 2°).
- Kết quả là 4 giá trị `*_smooth` dùng cho **cả hiển thị lẫn phân loại ngưỡng**:
  `forward_ratio_smooth`, `lateral_tilt_smooth`, `shoulder_imbalance_smooth`, `posture_score_smooth`.

### 3c. Thứ tự pipeline

`raw metrics` → MA (10 frame) → **EMA (α = 0.3)** → deviations vs baseline → phân loại
Good/Bad + debounce → score → UI (metric cards, Posture Score, Live Trend Chart).
Trên khung video chỉ còn vẽ landmark/guide line, gợi ý đỏ 'No pose detected' khi mất pose và
cảnh báo lớn 'BAD POSTURE!' khi alert — các nhãn số liệu vàng đã bị bỏ vì trùng cột Status.

**Tác dụng:** MA làm mịn nhiễu thất thường của landmark; EMA khử rung frame-to-frame còn sót lại
trước khi render. Ở 30 fps, MA thêm độ trễ ~0.33 s, EMA α=0.3 thêm hằng số thời gian ~3 frame (~0.1 s).

---

## 4. Hiệu chuẩn (Baseline Calibration)

- Nhấn **"Calibrate"**: reset EMA rồi lưu các chỉ số đã làm mịn (MA + EMA) hiện tại thành baseline
  `{'forward_lean', 'lateral_tilt', 'shoulder_imbalance'}`. Frame hiệu chuẩn được gán trực tiếp
  (`Value_smooth = Value_current`) nên baseline không bị "kéo" bởi lịch sử EMA cũ.
- **Tự động hiệu chuẩn:** nếu chưa có baseline, frame hợp lệ đầu tiên sẽ trở thành baseline
  (ngăn cảnh báo sai ngay từ đầu).
- Mọi chỉ số sau đó được so sánh theo: `deviation = smoothed_metric - baseline_metric`.
- Luôn hiệu chuẩn khi ngồi thẳng, nhìn thẳng, thả lỏng vai trong 3-5 giây.

---

## 5. Ngưỡng cảnh báo (thanh trượt bên)

| Thanh trượt | Khóa cấu hình | Dải | Mặc định | Đơn vị / ý nghĩa |
|---|---|---|---|---|
| Forward Lean (Cổ rùa) | `forward_thresh` | 0.0 - 0.5 | **0.05** | độ tăng *tỷ lệ mặt/vai* so với baseline |
| Lateral Tilt (Vẹo người) | `lateral_thresh` | 0 - 45 | **10.0** | thay đổi tuyệt đối góc đầu/tai (độ) |
| Shoulder Imbalance | `shoulder_thresh` | 0 - 45 | **10.0** | thay đổi tuyệt đối góc vai (độ) |

---

## 6. Số frame chống rung (Debounce Frames)

- Mặc định **8** frame; có thể chỉnh qua ô "Debounce frames".
- Bộ đếm tăng trên mỗi frame vi phạm và đặt về `0` nếu không vi phạm.
- `alert_active` chỉ chuyển `True` khi bộ đếm đạt `debounce_limit`.
- Ở 30fps, 8 frame tương đương khoảng **0.27 s** trước khi cảnh báo xuất hiện: đủ để bỏ qua
  nhiễu ngắn song vẫn phản hồi kịp tư thế xấu kéo dài.

---

## 7. Điểm tư thế (0-100)

Với mỗi chỉ số: `metric_score = 100 * (1 - min(1, |deviation| / threshold))`

- Nếu threshold = `0`, điểm chỉ số đó là `100`.
- Điểm cuối `score = (s_forward + s_lateral + s_shoulder) / 3`, làm tròn số nguyên.
- Điểm này được làm mịn tiếp bằng EMA (`posture_score_smooth`) trước khi hiển thị; khi không
  detect được pose, điểm hiển thị mặc định 100 và **không** đưa vào EMA.

---

## 8. Thông số mặc định khuyến nghị

| Thông số | Khuyến nghị | Trường hợp dùng |
|---|---|---|
| `forward_thresh` | 0.05 | nhẹ nhàng; tăng lên 0.08-0.10 để cho phép cử động tự nhiên |
| `lateral_thresh` | 10 độ | cân bằng mặc định; hạ xuống 5-7 độ nếu muốn giám sát chặt |
| `shoulder_thresh` | 8 - 10 độ | lệch vai thường là góc nhỏ |
| `debounce_limit` | 8 frame | mặc định; tăng lên 12-15 cho người hay di chuyển |
| bộ đệm trung bình động | 10 | cố định trong code |

---

## 9. Ví dụ cấu hình JSON

```json
{
  "camera": {
    "device_index": 0,
    "width": 640,
    "height": 480
  },
  "baseline": {
    "forward_lean": 0.25,
    "lateral_tilt": -1.3,
    "shoulder_imbalance": 2.4
  },
  "thresholds": {
    "forward_lean_increase": 0.05,
    "lateral_tilt_deg": 10,
    "shoulder_imbalance_deg": 10
  },
  "filter": {
    "moving_average_window": 10,
    "ema_alpha": 0.3,
    "debounce_limit": 8,
    "mode": "consecutive"
  }
}
```

> Ví dụ phản ánh đúng tên biến trong code. Trong ứng dụng thực tế, các ngưỡng và debounce
> được đặt qua thanh bên dựa trên giá trị mặc định ở trên; file JSON chỉ áp dụng nếu bạn
> nối nó vào code sau này.

---

## 10. Live Trend Chart (Posture Score, 60 giây)

- Biểu đồ đường `st.line_chart` đặt dưới cột **Status** (placeholder `trend_chart`) hiển thị
  `posture_score_smooth` (đã qua MA + EMA) trong **60 giây gần nhất**.
- Dữ liệu: `st.session_state.score_history` — deque các mẫu `(timestamp, score)`,
  `maxlen=600`; hàm `append_score_sample()` tự cắt các mẫu cũ hơn `TREND_WINDOW_SECONDS = 60`.
- `score_trend_dataframe()` chuyển samples thành DataFrame với **DatetimeIndex** nên trục X là
  thời gian thực (khoảng dừng camera vẫn hiện rõ trên trục thời gian).
- Tắt camera biểu đồ vẫn giữ nguyên lịch sử của phiên; chỉ bị xóa khi reload trang (mất session).
- Cần `pandas` (đã là dependency sẵn có của Streamlit).

---

### Ghi chú

- Kiểm tra đơn vị trong code: `forward_ratio` không thứ nguyên; `lateral_tilt` và
  `shoulder_imbalance` được đo bằng độ.
- MediaPipe `z` không còn được dùng cho forward lean; tỷ lệ phối cảnh 2D tránh tín hiệu
  độ sâu không tin cậy.
- Lưu metadata hiệu chuẩn (thời gian, fps, camera id) để gỡ lỗi khi cảnh báo sai.