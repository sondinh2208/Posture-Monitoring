# -*- coding: utf-8 -*-
"""
utils.py
========
Bộ xử lý toán học cốt lõi (business logic) của ứng dụng Posture Monitoring,
đã được bóc tách khỏi app.py để tách biệt logic khỏi phần UI (Streamlit).

Module bao gồm:
  - Khoảng cách / góc        : euclidean_distance, line_tilt_deg, shortest_angle_diff.
  - Đo lường tư thế          : compute_posture_metrics_3d (+ các hàm phụ trợ).
  - Bộ lọc EMA               : ema_smooth, ema_smooth_angle và lớp EMASmoother.
  - Điểm số tư thế           : metric_score.

Tuân thủ Clean Code:
  - Mỗi hàm có ĐÚNG MỘT trách nhiệm, tên tự giải thích (self-documenting).
  - Thuần túy (không đổi trạng thái ngoài phạm vi hàm) -> dễ kiểm thử đơn vị.
  - Type hints đầy đủ; hằng số tách khỏi các giá trị "ma thuật".
  - EMASmoother là nơi DUY NHẤT quản lý trạng thái lọc theo metric.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, TypedDict

from config import EMA_ALPHA

# =====================================================================
# Hằng số toán học (đơn vị độ)
# =====================================================================
FULL_CIRCLE_DEG = 360.0   # Góc cả vòng tròn.
HALF_CIRCLE_DEG = 180.0   # Góc nửa vòng tròn (xử lý wrap-around của atan2).
FOLD_ANGLE_DEG = 90.0     # Góc "gập" đường không định hướng về [0, 90].

# =====================================================================
# Hằng số điểm số
# =====================================================================
PERFECT_SCORE = 100.0     # Điểm tối đa của một metric khi không lệch chuẩn.

# =====================================================================
# Tên landmark MediaPipe dùng trong tính toán
# (khớp với key sinh bởi landmarks_to_dict() trong app.py)
# =====================================================================
LANDMARK_LEFT_SHOULDER = 'LEFT_SHOULDER'
LANDMARK_RIGHT_SHOULDER = 'RIGHT_SHOULDER'
LANDMARK_LEFT_EAR = 'LEFT_EAR'
LANDMARK_RIGHT_EAR = 'RIGHT_EAR'
LANDMARK_LEFT_EYE = 'LEFT_EYE'
LANDMARK_RIGHT_EYE = 'RIGHT_EYE'

# =====================================================================
# Kiểu dữ liệu cấp cao
# =====================================================================


class Landmark(TypedDict):
    """Một landmark của MediaPipe: tọa độ chuẩn hóa (nx, ny, nz) lẫn tọa độ pixel (x, y)."""

    nx: float
    ny: float
    nz: float
    x: int
    y: int


# Dict ánh xạ tên landmark (xem các hằng số LANDMARK_*) -> Landmark.
LandmarkDict = Dict[str, Landmark]

# Điểm trong mặt phẳng 2D (tọa độ pixel hoặc bất kỳ đơn vị thống nhất nào).
Point = Tuple[float, float]


class PostureMetrics(TypedDict):
    """Ba chỉ số tư thế thô (đầu ra các phép đo trước bộ lọc EMA)."""

    forward_lean: float
    lateral_tilt: float
    shoulder_imbalance: float


class PostureMetricsSmooth(TypedDict):
    """Ba chỉ số tư thế đã qua bộ lọc EMA (dùng cho hiển thị & phân loại cảnh báo)."""

    forward_ratio_smooth: float
    lateral_tilt_smooth: float
    shoulder_imbalance_smooth: float


# =====================================================================
# Khoảng cách & góc
# =====================================================================


def euclidean_distance(point_a: Point, point_b: Point) -> float:
    """Tính khoảng cách Euclid giữa hai điểm (cùng đơn vị)."""
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return math.hypot(dx, dy)


def line_tilt_deg(dx: float, dy: float) -> float:
    """Góc nghiêng tuyệt đối của một đường KHÔNG ĐỊNH HƯỚNG so với trục ngang, trong [0, 90] độ.

    atan2(dy, dx) trả về hướng CÓ DẤU trong (-180, 180]; nhưng một đường thẳng không có
    hướng, nên mọi góc ngoài ±90 độ được "gập" về độ nghiêng tương đương
    (vd: 180° == 0°, -170° == 10°). Kết quả ~0 độ cho đường ngang và tăng dần khi đường
    nghiêng, nên tư thế thẳng đọc ~0 độ và tư thế ngồi điển hình nằm trong khoảng
    0-45 độ của thanh trượt ngưỡng.

    Truyền delta CÙNG ĐƠN VỊ (tọa độ pixel) để ra góc hình học thực, không bị bóp méo
    bởi tỷ lệ khung hình (aspect ratio) của camera.
    """
    theta = math.degrees(math.atan2(dy, dx))
    if theta > FOLD_ANGLE_DEG:
        theta -= HALF_CIRCLE_DEG
    elif theta < -FOLD_ANGLE_DEG:
        theta += HALF_CIRCLE_DEG
    return abs(theta)


def shortest_signed_angle_diff(current_deg: float, reference_deg: float) -> float:
    """Hiệu góc ngắn nhất CÓ DẤU (kết quả trong [-180, 180)).

    Xử lý wrap-around của atan2: hai góc nằm cạnh ranh giới +180/-180
    (vd: current=179, baseline=-179) ra hiệu ~2 độ thay vì ~358 độ.
    Giữ dấu để bộ lọc EMA biết hướng đi ngắn nhất qua ranh giới.
    """
    return (current_deg - reference_deg + HALF_CIRCLE_DEG) % FULL_CIRCLE_DEG - HALF_CIRCLE_DEG


def shortest_angle_diff(current_deg: float, baseline_deg: float) -> float:
    """Hiệu góc ngắn nhất (trị tuyệt đối), kết quả trong [0, 180].

    Dùng để so sánh độ lệch với baseline: tránh nhiễu giả khi góc đi qua
    ranh giới +180/-180 của atan2 (không dùng phép trừ tuyến tính).
    """
    return abs(shortest_signed_angle_diff(current_deg, baseline_deg))


# =====================================================================
# Đo lường tư thế
# =====================================================================


def _pixel_point(landmark: Landmark) -> Point:
    """Trích tọa độ pixel (x, y) từ một landmark."""
    return (landmark['x'], landmark['y'])


def _first_available_landmark(
    landmarks: LandmarkDict,
    preferred: str,
    fallback: str,
) -> Optional[Landmark]:
    """Chọn mốc ưu tiên; nếu bị che khuất/thiếu thì dùng mốc dự phòng."""
    return landmarks.get(preferred) or landmarks.get(fallback)


def _forward_lean_ratio(
    left_face: Landmark,
    right_face: Landmark,
    left_shoulder: Landmark,
    right_shoulder: Landmark,
) -> float:
    """Tỷ lệ chiều rộng khuôn mặt / chiều rộng vai (2D perspective ratio).

    Khi đầu tiến về phía camera, khuôn mặt trông lớn hơn nên tỷ lệ tăng lên —
    đó là dấu hiệu phát hiện lệch về trước (turtle neck). Tính trên tọa độ pixel.
    """
    face_width = euclidean_distance(_pixel_point(left_face), _pixel_point(right_face))
    shoulder_width = euclidean_distance(_pixel_point(left_shoulder), _pixel_point(right_shoulder))
    if shoulder_width <= 0:
        return 0.0
    return face_width / shoulder_width


def _segment_tilt_deg(left: Landmark, right: Landmark) -> float:
    """Góc nghiêng (độ) của đoạn thẳng nối hai landmark so với phương ngang (chuẩn [0, 90])."""
    dx = right['x'] - left['x']
    dy = right['y'] - left['y']
    return line_tilt_deg(dx, dy)


def compute_posture_metrics_3d(landmarks: LandmarkDict) -> Optional[PostureMetrics]:
    """Tính ba chỉ số tư thế từ dict landmark (đầu ra của landmarks_to_dict).

    - forward_lean (tỷ lệ) : face_width / shoulder_width; chỉ cảnh báo khi tăng so baseline.
    - lateral_tilt (độ)    : độ nghiêng đầu so phương ngang
      (đường tai-đến-tai, dự phòng mắt; khi đầu thẳng/ngay đọc ~0 độ).
    - shoulder_imbalance (độ): độ nghiêng đường vai so phương ngang
      (0 độ = vai ngang bằng; lớn hơn = một vai cao hơn vai kia).

    Trả về None nếu thiếu mốc vai hoặc không có mốc tham chiếu khuôn mặt hợp lệ.
    """
    left_shoulder = landmarks.get(LANDMARK_LEFT_SHOULDER)
    right_shoulder = landmarks.get(LANDMARK_RIGHT_SHOULDER)
    if left_shoulder is None or right_shoulder is None:
        return None

    left_face = _first_available_landmark(landmarks, LANDMARK_LEFT_EAR, LANDMARK_LEFT_EYE)
    right_face = _first_available_landmark(landmarks, LANDMARK_RIGHT_EAR, LANDMARK_RIGHT_EYE)
    if left_face is None or right_face is None:
        return None

    return {
        'forward_lean': _forward_lean_ratio(
            left_face, right_face, left_shoulder, right_shoulder
        ),
        'lateral_tilt': _segment_tilt_deg(left_face, right_face),
        'shoulder_imbalance': _segment_tilt_deg(left_shoulder, right_shoulder),
    }


# =====================================================================
# Bộ lọc EMA (Exponential Moving Average)
# =====================================================================


def ema_smooth(
    current: float,
    previous: Optional[float],
    alpha: float = EMA_ALPHA,
) -> float:
    """EMA (Exponential Moving Average) cho đại lượng vô hướng.

    Công thức: Value_smooth = alpha * Value_current + (1 - alpha) * Value_previous.
    Khung đầu tiên (previous=None): gán trực tiếp giá trị hiện tại để lọc
    không bị trễ khởi động (không blend với dữ liệu cũ).
    """
    if previous is None:
        return float(current)
    return alpha * float(current) + (1.0 - alpha) * float(previous)


def ema_smooth_angle(
    current_deg: float,
    previous_deg: Optional[float],
    alpha: float = EMA_ALPHA,
) -> float:
    """EMA bảo toàn wrap-around cho góc (đơn vị độ, phạm vi atan2 -180..180).

    Blend góc kiểu ngây thơ (vd: current=179, previous=-179) cho ~0 độ — sai.
    Thay vào đó, đi từ góc trước dọc theo hiệu ngắn nhất hướng về góc hiện tại:
    previous + alpha * shortest_signed_angle_diff(current, previous).
    """
    if previous_deg is None:
        return float(current_deg)
    return previous_deg + alpha * shortest_signed_angle_diff(current_deg, previous_deg)


class EMASmoother:
    """Bộ lọc EMA theo từng metric, giữ trạng thái bộ nhớ (theo phiên làm việc/trình duyệt).

    Lưu giá trị smoothed mới nhất cho từng khóa lọc:
      - 'forward_ratio'       -> rendered as forward_ratio_smooth
      - 'lateral_tilt'        -> rendered as lateral_tilt_smooth
      - 'shoulder_imbalance'  -> rendered as shoulder_imbalance_smooth
      - 'posture_score'       -> rendered as posture_score_smooth

    Khung đầu tiên, hoặc ngay sau reset() (Calibrate / khởi động lại camera):
    gán trực tiếp Value_smooth = Value_current (không blend với dữ liệu cũ).
    """

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        """Khởi tạo bộ lọc với hệ số làm mượt alpha trong khoảng (0, 1]."""
        self.alpha = float(alpha)
        self._previous_values: Dict[str, float] = {}

    def reset(self) -> None:
        """Xóa toàn bộ lịch sử EMA để khung kế tiếp được gán trực tiếp (không blend)."""
        self._previous_values.clear()

    def smooth(self, key: str, value: float, *, angular: bool = False) -> float:
        """Blend EMA cho `value` của `key`, lưu làm previous mới và trả về giá trị smoothed."""
        previous = self._previous_values.get(key)
        if previous is None:
            smoothed = float(value)
        elif angular:
            smoothed = ema_smooth_angle(value, previous, self.alpha)
        else:
            smoothed = ema_smooth(value, previous, self.alpha)
        self._previous_values[key] = smoothed
        return smoothed

    def smooth_metrics(
        self,
        metrics: Optional[PostureMetrics],
    ) -> Optional[PostureMetricsSmooth]:
        """Áp dụng EMA cho cả ba chỉ số tư thế (sau bộ lọc moving-average).

        Trả về dict với các khóa *_smooth để render; trả về None nếu đầu vào None
        (không phát hiện tư thế) — không kéo dữ liệu rác vào bộ lọc.
        """
        if metrics is None:
            return None
        return {
            'forward_ratio_smooth': self.smooth('forward_ratio', metrics['forward_lean']),
            # Góc đã là độ lớn trong [0, 90] (không wrap); vẫn dùng angular EMA
            # như lưới an toàn (safety net), kết quả tương đương EMA thường.
            'lateral_tilt_smooth': self.smooth(
                'lateral_tilt', metrics['lateral_tilt'], angular=True
            ),
            'shoulder_imbalance_smooth': self.smooth(
                'shoulder_imbalance', metrics['shoulder_imbalance'], angular=True
            ),
        }

    def smooth_score(self, score: float) -> float:
        """EMA-blend điểm tư thế 0-100; trả về posture_score_smooth."""
        return self.smooth('posture_score', score)


# =====================================================================
# Điểm số tư thế
# =====================================================================


def metric_score(
    deviation: float,
    threshold: Optional[float],
    is_normalized: bool = False,
) -> float:
    """Điểm (0-100) cho một metric từ độ lệch so với ngưỡng; cao hơn = tư thế tốt hơn.

    - deviation = 0               -> 100 điểm.
    - deviation >= threshold      -> 0 điểm.
    - Ở giữa                     -> giảm tuyến tính (100 * (1 - ratio)).

    `is_normalized` giữ lại vì lý do tương thích API với app.py (forward_lean truyền
    True nhưng độ lệch đã là tỷ lệ phi thứ nguyên). Nếu ngưỡng None hoặc <= 0
    (chủ ý vô hiệu hóa cảnh báo) thì luôn trả về điểm tối đa.
    """
    del is_normalized  # Giữ tương thích API; cả hai đường tính đều cho kết quả như nhau.
    if threshold is None or threshold <= 0:
        return PERFECT_SCORE
    normalized_deviation = min(1.0, abs(deviation) / threshold)
    return max(0.0, PERFECT_SCORE * (1.0 - normalized_deviation))