# -*- coding: utf-8 -*-
"""
config.py
=========
Tập trung TOÀN BỘ biến số cấu hình, màu sắc (BGR cho OpenCV) và
ngưỡng (threshold) của ứng dụng Posture Monitoring (app.py).

Quy ước:
  - Tất cả màu OpenCV đều là tuple BGR.
  - Các hằng số ngưỡng được nhóm theo từng metric để app.py dễ
    khởi tạo slider: (MIN, MAX, DEFAULT, STEP).
"""

# =====================================================================
# Camera
# =====================================================================
CAMERA_DEVICE_INDEX = 0      # Index camera mặc định (dùng cv2.VideoCapture)
CAMERA_FRAME_WIDTH = 800     # Độ rộng khung hình mong muốn (CAP_PROP_FRAME_WIDTH)
CAMERA_FRAME_HEIGHT = 600    # Độ cao khung hình mong muốn (CAP_PROP_FRAME_HEIGHT)
CAMERA_FLIP_CODE = 1         # Tham số cv2.flip(frame, 1) -> ảnh selfie không bị lật ngược

# =====================================================================
# MediaPipe Pose
# =====================================================================
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# =====================================================================
# Vòng lặp xử lý khung hình (mỗi lần Streamlit rerun)
# =====================================================================
FRAMES_PER_RUN = 5           # Số khung xử lý trong 1 lần rerun để UI (nút Stop) giữ phản hồi
LOOP_SLEEP_SECONDS = 0.02    # time.sleep sau mỗi khung -> giới hạn frame rate của while loop

# =====================================================================
# Bộ lọc Moving Average (lọc nhiễu từng trục đo)
# =====================================================================
MA_WINDOW_SIZE = 10          # Số mẫu gần nhất giữ trong deque cho mỗi metric

# =====================================================================
# Bộ lọc EMA (Exponential Moving Average)
# =====================================================================
#   Value_smooth = alpha * Value_current + (1 - alpha) * Value_previous
# alpha ~ 0.3 -> phản hồi nhanh với thay đổi tư thế thật, vẫn triệt nhiễu
# jitter giữa các khung hình. Giá trị thấp hơn = mượt hơn / chậm hơn.
EMA_ALPHA = 0.3

# =====================================================================
# Biểu đồ xu hướng Posture Score (cửa sổ 60 s)
# =====================================================================
TREND_WINDOW_SECONDS = 60.0  # Độ rộng cửa sổ trượt (giây) hiển thị trên biểu đồ
TREND_MAX_SAMPLES = 600      # Giới hạn số mẫu lưu trữ (đủ cho 60 s ở ~10 mẫu/s)
TREND_CHART_HEIGHT = 200     # Chiều cao st.line_chart (px)

# =====================================================================
# Ngưỡng (Threshold) — Forward Lean (Turtle Neck)
# =====================================================================
# Đo bằng tỷ lệ face_width / shoulder_width; chỉ cảnh báo khi tăng so với baseline.
FORWARD_THRESH_MIN = 0.0
FORWARD_THRESH_MAX = 0.5
FORWARD_THRESH_DEFAULT = 0.04
FORWARD_THRESH_STEP = 0.01

# =====================================================================
# Ngưỡng (Threshold) — Lateral Tilt (Body Lean, deg)
# =====================================================================
LATERAL_THRESH_MIN = 0.0
LATERAL_THRESH_MAX = 45.0
LATERAL_THRESH_DEFAULT = 10.0
LATERAL_THRESH_STEP = 0.5

# =====================================================================
# Ngưỡng (Threshold) — Shoulder Imbalance (deg)
# =====================================================================
SHOULDER_THRESH_MIN = 0.0
SHOULDER_THRESH_MAX = 45.0
SHOULDER_THRESH_DEFAULT = 10.0
SHOULDER_THRESH_STEP = 0.5

# =====================================================================
# Debounce (số khung hình liên tiếp vượt ngưỡng trước khi cảnh báo)
# =====================================================================
DEBOUNCE_DEFAULT_FRAMES = 15  # Giá trị mặc định của st.session_state.debounce_limit
DEBOUNCE_MIN_FRAMES = 1
DEBOUNCE_MAX_FRAMES = 300
DEBOUNCE_STEP = 1

# =====================================================================
# Màu sắc (BGR) dùng cho overlay OpenCV
# =====================================================================
COLOR_OK = (0, 255, 0)            # Xanh lá: tư thế tốt / không cảnh báo
COLOR_ALERT = (0, 0, 255)         # Đỏ: cảnh báo tư thế xấu
COLOR_TILT_LINE = (0, 255, 255)   # Vàng: đường nối hai tai (head tilt)
COLOR_TEXT_BG = (0, 0, 0)         # Nền đen bán trong suốt phía sau chữ

# Độ dày nét vẽ chung (cv2.line / landmark DrawingSpec)
LINE_THICKNESS = 2

# Chữ viết lên khung hình (đường nền nửa trong suốt)
TEXT_BG_ALPHA = 0.5        # Độ đậm của nền chữ khi alpha-blend
TEXT_BG_PAD_X = 5          # Khoảng cách ngang mở rộng khung nền chữ
TEXT_BG_PAD_Y = 6          # Khoảng cách dọc mở rộng khung nền chữ

# Landmark / connection khi vẽ bằng MediaPipe drawing_utils
LANDMARK_COLOR_OK = (0, 255, 0)      # Không cảnh báo
LANDMARK_COLOR_ALERT = (0, 0, 255)   # Đang cảnh báo
LANDMARK_THICKNESS = 2
LANDMARK_CIRCLE_RADIUS = 2

# Overlay cảnh báo / thông báo trên video
WARN_FONT_SCALE = 0.7      # Thông báo 'No pose detected'
WARN_FONT_THICKNESS = 2
ALERT_FONT_SCALE = 2.0     # Thông báo 'BAD POSTURE!'
ALERT_FONT_THICKNESS = 4

# =====================================================================
# Giao diện Streamlit
# =====================================================================
UI_COLUMNS_RATIO = [3, 1]  # Tỷ lệ cột video : cột thông tin trạng thái
# Kích thước khung đen placeholder khi chưa mở camera (trùng CHIỀU camera)
PLACEHOLDER_WIDTH = CAMERA_FRAME_WIDTH
PLACEHOLDER_HEIGHT = CAMERA_FRAME_HEIGHT
PLACEHOLDER_CHANNELS = 3