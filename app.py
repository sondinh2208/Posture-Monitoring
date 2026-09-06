# -*- coding: utf-8 -*-
"""
app.py
======
Ứng dụng Streamlit giám sát tư thế ngồi (Seated Posture Monitoring).

Cấu trúc (trách nhiệm tách bạch):
  - config.py  : mọi biến cấu hình, màu sắc, ngưỡng.
  - utils.py   : tính toán hình học + bộ lọc EMA + điểm số.
  - database.py: lưu trữ/truy xuất dữ liệu tư thế (SQLite).
  - app.py     : UI Streamlit với 2 tabs — Real-time Monitor & Analytics Dashboard.

Tổ chức giao diện:
  - Tab 1 "🎥 Real-time Monitor": vòng lặp camera + overlay + ghi log DB (throttle).
  - Tab 2 "📊 Analytics Dashboard": thống kê dữ liệu lịch sử bằng plotly.express.
"""

import os
import time
import warnings
from collections import deque

# ------------------------------------------------------------------
# Tắt cảnh báo/log hệ thống không cần thiết.
# BẮT BUỘC đặt TRƯỚC khi import mediapipe (mediapipe kéo theo tensorflow).
# ------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
# Chặn UserWarning: 'SymbolDatabase.GetPrototype() is deprecated' (mediapipe/tf).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Giảm log TensorFlow: 0 = tất cả, 1 = info, 2 = warning+error, 3 = chỉ error.
# ------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    ALERT_FONT_SCALE,
    ALERT_FONT_THICKNESS,
    CAMERA_DEVICE_INDEX,
    CAMERA_FLIP_CODE,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    COLOR_ALERT,
    COLOR_OK,
    COLOR_TEXT_BG,
    COLOR_TILT_LINE,
    DATABASE_PATH,
    DEBOUNCE_DEFAULT_FRAMES,
    DEBOUNCE_MAX_FRAMES,
    DEBOUNCE_MIN_FRAMES,
    DEBOUNCE_STEP,
    EMA_ALPHA,
    FORWARD_THRESH_DEFAULT,
    FORWARD_THRESH_MAX,
    FORWARD_THRESH_MIN,
    FORWARD_THRESH_STEP,
    FRAMES_PER_RUN,
    LANDMARK_CIRCLE_RADIUS,
    LANDMARK_COLOR_ALERT,
    LANDMARK_COLOR_OK,
    LANDMARK_THICKNESS,
    LATERAL_THRESH_DEFAULT,
    LATERAL_THRESH_MAX,
    LATERAL_THRESH_MIN,
    LATERAL_THRESH_STEP,
    LINE_THICKNESS,
    LOG_SAVE_INTERVAL_SECONDS,
    LOOP_SLEEP_SECONDS,
    MA_WINDOW_SIZE,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    PLACEHOLDER_CHANNELS,
    PLACEHOLDER_HEIGHT,
    PLACEHOLDER_WIDTH,
    SHOULDER_THRESH_DEFAULT,
    SHOULDER_THRESH_MAX,
    SHOULDER_THRESH_MIN,
    SHOULDER_THRESH_STEP,
    TEXT_BG_ALPHA,
    TEXT_BG_PAD_X,
    TEXT_BG_PAD_Y,
    TREND_CHART_HEIGHT,
    TREND_MAX_SAMPLES,
    TREND_WINDOW_SECONDS,
    UI_COLUMNS_RATIO,
    WARN_FONT_SCALE,
    WARN_FONT_THICKNESS,
)
from utils import EMASmoother, compute_posture_metrics_3d, metric_score, shortest_angle_diff
from database import clear_all_data, get_session_data, init_db, save_log

# --------------------- Terminal status logging (ANSI colors) ---------------------

TERMINAL_ANSI_GREEN = '\033[92m'   # Xanh lá: tư thế Tốt
TERMINAL_ANSI_RED = '\033[91m'     # Đỏ: tư thế Xấu / Cảnh báo
TERMINAL_ANSI_RESET = '\033[0m'    # Reset về màu mặc định
POSTURE_LOG_PREFIX = '[POSTURE MONITOR]'


def log_posture_status_good(score):
    """In trạng thái TƯ THẾ TỐT ra Terminal màu xanh lá."""
    print(f"{TERMINAL_ANSI_GREEN}{POSTURE_LOG_PREFIX} \u2705 Good Posture (Score: {score})"
          f"{TERMINAL_ANSI_RESET}")


def log_posture_status_bad(score):
    """In trạng thái TƯ THẾ XẤU / CẢNH BÁO ra Terminal màu đỏ."""
    print(f"{TERMINAL_ANSI_RED}{POSTURE_LOG_PREFIX} \u274c Bad Posture Detected! (Score: {score})"
          f"{TERMINAL_ANSI_RESET}")


# --------------------------- Helper functions ---------------------------


def ensure_session_state_keys():
    """Khởi tạo các key cần thiết trong st.session_state với giá trị mặc định hợp lý."""
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'pose' not in st.session_state:
        st.session_state.pose = None
    if 'baseline' not in st.session_state:
        st.session_state.baseline = {
            'forward_lean': None, 'lateral_tilt': None, 'shoulder_imbalance': None
        }
    if 'debounce_counter' not in st.session_state:
        st.session_state.debounce_counter = 0
    if 'debounce_limit' not in st.session_state:
        st.session_state.debounce_limit = DEBOUNCE_DEFAULT_FRAMES
    if 'alert_active' not in st.session_state:
        st.session_state.alert_active = False
    # --- Trạng thái tư thế gần nhất đã in ra Terminal (chống in lặp) ---
    if 'last_logged_posture' not in st.session_state:
        st.session_state.last_logged_posture = None
    # --- Trạng thái throttle ghi database (tránh phình DB) ---
    if 'last_log_time' not in st.session_state:
        st.session_state.last_log_time = 0.0
    if 'last_log_status' not in st.session_state:
        st.session_state.last_log_status = None
    # --- Moving Average filter buffers ---
    if 'lean_buffer' not in st.session_state:
        st.session_state.lean_buffer = deque(maxlen=MA_WINDOW_SIZE)
    if 'tilt_buffer' not in st.session_state:
        st.session_state.tilt_buffer = deque(maxlen=MA_WINDOW_SIZE)
    if 'imbalance_buffer' not in st.session_state:
        st.session_state.imbalance_buffer = deque(maxlen=MA_WINDOW_SIZE)
    # --- EMA smoother (trạng thái lọc nằm trong utils.EMASmoother) ---
    if 'ema_smoother' not in st.session_state:
        st.session_state.ema_smoother = EMASmoother(alpha=EMA_ALPHA)
    # --- Live trend chart history ---
    if 'score_history' not in st.session_state:
        st.session_state.score_history = deque(maxlen=TREND_MAX_SAMPLES)
    # --- Last-seen state (khôi phục UI sau khi tắt camera) ---
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'last_metrics' not in st.session_state:
        st.session_state.last_metrics = None
    if 'last_deviations' not in st.session_state:
        st.session_state.last_deviations = None
    if 'last_score' not in st.session_state:
        st.session_state.last_score = None


def release_resources():
    """Giải phóng tài nguyên cv2 và mediapipe đang lưu trong session_state."""
    try:
        if st.session_state.get('cap') is not None:
            try:
                st.session_state.cap.release()
            except Exception:
                pass
            st.session_state.cap = None
    except Exception:
        pass
    try:
        if st.session_state.get('pose') is not None:
            try:
                st.session_state.pose.close()
            except Exception:
                pass
            st.session_state.pose = None
    except Exception:
        pass
    # Close any OpenCV HighGUI windows to fully terminate the webcam process.
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def start_camera():
    """Callback của nút 'Start Camera': bật cờ camera_running."""
    st.session_state.camera_running = True


def stop_camera():
    """Callback của nút 'Stop Camera': dừng stream và giải phóng webcam hoàn toàn."""
    release_resources()
    st.session_state.camera_running = False
    # Reset EMA để phiên camera mới bắt đầu bằng direct assignment (không blend dữ liệu cũ).
    if st.session_state.get('ema_smoother') is not None:
        st.session_state.ema_smoother.reset()


def open_camera(device_index=CAMERA_DEVICE_INDEX):
    """Mở camera, trả về cv2.VideoCapture (dùng DirectShow backend trên Windows)."""
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    return cap


def landmarks_to_dict(landmarks, image_w, image_h):
    """Chuyển normalized landmarks thành dict: name -> {'nx','ny','nz','x','y'} (pixel)."""
    pts = {}
    for lm_enum in mp.solutions.pose.PoseLandmark:
        idx = lm_enum.value
        lm = landmarks[idx]
        nx, ny, nz = lm.x, lm.y, lm.z
        px = int(nx * image_w)
        py = int(ny * image_h)
        pts[lm_enum.name] = {'nx': nx, 'ny': ny, 'nz': nz, 'x': px, 'y': py}
    return pts


# --------------------- Live Trend Chart (Posture Score, last 60 s) ---------------------


def append_score_sample(history, score, now=None):
    """Thêm mẫu (timestamp, score) và loại bỏ mẫu cũ hơn cửa sổ TREND_WINDOW_SECONDS."""
    if now is None:
        now = time.time()
    history.append((float(now), float(score)))
    cutoff = float(now) - TREND_WINDOW_SECONDS
    while history and history[0][0] < cutoff:
        history.popleft()
    return history


def score_trend_dataframe(history):
    """Chuyển (timestamp, score) thành DataFrame cho st.line_chart (DatetimeIndex real time)."""
    if not history:
        return pd.DataFrame({'Posture Score': []})
    times = [t for t, _ in history]
    scores = [s for _, s in history]
    return pd.DataFrame({'Posture Score': scores}, index=pd.to_datetime(times, unit='s'))


def draw_text_with_bg(frame, text, org, font, scale, color, thickness,
                      bg_color=COLOR_TEXT_BG, bg_alpha=TEXT_BG_ALPHA):
    """Vẽ chữ với nền đen bán trong suốt (alpha-blend) để dễ đọc trên mọi nền."""
    h, w = frame.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    # Mở rộng khung nền quanh chữ và cắt theo biên frame.
    p1 = (max(0, x - TEXT_BG_PAD_X), max(0, y - th - TEXT_BG_PAD_Y))
    p2 = (min(w - 1, x + tw + TEXT_BG_PAD_X), min(h - 1, y + baseline + 2))
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, dst=frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    return frame


def draw_guides(frame, pts, metrics):
    """Vẽ đường vai ngang và đường thẳng mũi -> điểm giữa vai (màu phụ thuộc alert)."""
    if not pts:
        return frame
    color = COLOR_OK if not st.session_state.alert_active else COLOR_ALERT

    # Đường nghiêng đầu (VÀNG): nối hai tai (dự phòng hai mắt nếu tai bị che).
    try:
        left_ref = pts.get('LEFT_EAR') or pts.get('LEFT_EYE')
        right_ref = pts.get('RIGHT_EAR') or pts.get('RIGHT_EYE')
        if left_ref is not None and right_ref is not None:
            cv2.line(frame, (left_ref['x'], left_ref['y']),
                     (right_ref['x'], right_ref['y']), COLOR_TILT_LINE, LINE_THICKNESS)
    except Exception:
        pass

    # Đường vai (pixel).
    try:
        left_sh = pts['LEFT_SHOULDER']
        right_sh = pts['RIGHT_SHOULDER']
        cv2.line(frame, (left_sh['x'], left_sh['y']),
                 (right_sh['x'], right_sh['y']), color, LINE_THICKNESS)
    except Exception:
        pass

    # Đường dọc mũi -> điểm giữa vai.
    try:
        nose = pts['NOSE']
        sh_mid_px = (int((left_sh['x'] + right_sh['x']) / 2),
                     int((left_sh['y'] + right_sh['y']) / 2))
        cv2.line(frame, (nose['x'], nose['y']), sh_mid_px, color, LINE_THICKNESS)
    except Exception:
        pass

    return frame


def determine_error_type(forward_violation, lateral_violation, shoulder_violation):
    """Gom các loại vi phạm đang xảy ra thành chuỗi error_type cho database.

    Nhiều lỗi đồng thời được nối bằng " + " (vd: "Turtle Neck + Lateral Tilt").
    Trả về 'None' khi không có vi phạm nào.
    """
    errors = []
    if forward_violation:
        errors.append('Turtle Neck')
    if lateral_violation:
        errors.append('Lateral Tilt')
    if shoulder_violation:
        errors.append('Shoulder Imbalance')
    return " + ".join(errors) if errors else 'None'


# --------------------------- UI render functions ---------------------------


def render_sidebar():
    """Dựng toàn bộ sidebar control, trả về dict các giá trị cần cho monitor."""
    with st.sidebar:
        st.header('Control Panel')
        st.button('Start Camera', type='primary', on_click=start_camera,
                  disabled=st.session_state.camera_running)
        st.button('Stop Camera', on_click=stop_camera,
                  disabled=not st.session_state.camera_running)
        st.caption('Status: ' + ('Running' if st.session_state.camera_running else 'Stopped'))

        st.markdown('---')
        st.subheader('Calibration')
        calibrate_btn = st.button('Calibrate')

        st.markdown('---')
        # Threshold sliders (ngưỡng lấy từ config.py)
        with st.expander('Threshold Settings', expanded=True):
            forward_thresh = st.slider(
                'Forward Lean (Turtle Neck) Threshold (increase in face/shoulder ratio)',
                min_value=FORWARD_THRESH_MIN, max_value=FORWARD_THRESH_MAX,
                value=FORWARD_THRESH_DEFAULT, step=FORWARD_THRESH_STEP, key='forward_thresh')
            lateral_thresh = st.slider(
                'Lateral Tilt (Body Lean) Threshold (deg)',
                min_value=LATERAL_THRESH_MIN, max_value=LATERAL_THRESH_MAX,
                value=LATERAL_THRESH_DEFAULT, step=LATERAL_THRESH_STEP, key='lateral_thresh')
            shoulder_thresh_new = st.slider(
                'Shoulder Imbalance Threshold (deg)',
                min_value=SHOULDER_THRESH_MIN, max_value=SHOULDER_THRESH_MAX,
                value=SHOULDER_THRESH_DEFAULT, step=SHOULDER_THRESH_STEP, key='shoulder_thresh')

        st.markdown('---')
        st.write('Debounce frames (consecutive frames before alert):')
        debounce_frames = st.number_input(
            'Frames', min_value=DEBOUNCE_MIN_FRAMES, max_value=DEBOUNCE_MAX_FRAMES,
            value=st.session_state.debounce_limit, step=DEBOUNCE_STEP, key='debounce_limit')

    return {
        'forward_thresh': forward_thresh,
        'lateral_thresh': lateral_thresh,
        'shoulder_thresh': shoulder_thresh_new,
        'debounce_frames': debounce_frames,
        'calibrate_btn': calibrate_btn,
    }


def render_monitor_tab(controls):
    """Tab 1: Real-time Monitor — camera loop, overlay và ghi log DB."""
    forward_thresh = controls['forward_thresh']
    lateral_thresh = controls['lateral_thresh']
    shoulder_thresh = controls['shoulder_thresh']
    debounce_frames = controls['debounce_frames']
    calibrate_btn = controls['calibrate_btn']

    # Layout: main video + side info
    col1, col2 = st.columns(UI_COLUMNS_RATIO)
    frame_placeholder = col1.empty()

    with col2:
        st.subheader('Status')
        score_text = st.empty()
        warning_text = st.empty()
        st.markdown('---')
        # Live metric cards với value + delta so với baseline.
        m_forward = st.empty()
        m_lateral = st.empty()
        m_shoulder = st.empty()
        st.markdown('---')
        # Live trend chart: Posture Score (0-100) cuối 60 giây.
        st.caption('Posture Score trend (last 60 s)')
        trend_chart = st.empty()
        if st.session_state.alert_active:
            st.error('ALERT: PERSISTENT BAD POSTURE!')

    # Manage camera open/close
    if st.session_state.camera_running:
        # Mở tài nguyên nếu chưa mở.
        if st.session_state.cap is None:
            st.session_state.cap = open_camera()
        if st.session_state.pose is None:
            st.session_state.pose = mp.solutions.pose.Pose(
                min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)

        cap = st.session_state.cap
        pose = st.session_state.pose

        if not cap or not cap.isOpened():
            frame_placeholder.image(
                np.zeros((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, PLACEHOLDER_CHANNELS),
                         dtype=np.uint8))
            st.error('Cannot open camera. Check access permissions.')
            cap.release()
            cv2.destroyAllWindows()
            st.session_state.cap = None
            st.session_state.camera_running = False
        else:
            # Xử lý một batch nhỏ khung hình mỗi lần rerun để UI (nút Stop) giữ phản hồi.
            frames_done = 0
            frame_ok = True
            while st.session_state.camera_running and frames_done < FRAMES_PER_RUN and frame_ok:
                success, frame = cap.read()
                if not success:
                    frame_ok = False
                    break
                frames_done += 1
                frame = cv2.flip(frame, CAMERA_FLIP_CODE)
                image_h, image_w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                metrics = None
                pts = None
                if results.pose_landmarks:
                    pts = landmarks_to_dict(results.pose_landmarks.landmark, image_w, image_h)
                    metrics = compute_posture_metrics_3d(pts)  # từ utils.py


# ===================== Moving Average Filter =====================
                if metrics is not None:
                    st.session_state.lean_buffer.append(metrics['forward_lean'])
                    st.session_state.tilt_buffer.append(metrics['lateral_tilt'])
                    st.session_state.imbalance_buffer.append(metrics['shoulder_imbalance'])

                if len(st.session_state.lean_buffer) > 0:
                    metrics = {
                        'forward_lean': float(np.mean(st.session_state.lean_buffer)),
                        'lateral_tilt': float(np.mean(st.session_state.tilt_buffer)),
                        'shoulder_imbalance': float(np.mean(st.session_state.imbalance_buffer)),
                    }
                else:
                    metrics = None
                # ===================== End Moving Average Filter =====================

                # ===================== EMA Filter =====================
                # Calibrate được bấm: reset EMA để khung hiện tại được gán trực tiếp.
                if calibrate_btn:
                    st.session_state.ema_smoother.reset()

                metrics_smooth = st.session_state.ema_smoother.smooth_metrics(metrics)

                # Lưu baseline khi bấm Calibrate (từ giá trị smoothed).
                if calibrate_btn:
                    if metrics_smooth is not None:
                        st.session_state.baseline = {
                            'forward_lean': metrics_smooth['forward_ratio_smooth'],
                            'lateral_tilt': metrics_smooth['lateral_tilt_smooth'],
                            'shoulder_imbalance': metrics_smooth['shoulder_imbalance_smooth'],
                        }
                        st.success('Calibration baseline saved.')
                    else:
                        st.warning('No pose detected for calibration. Please stand straight in front of the camera.')

                # Auto-calibrate khung đầu tiên nếu chưa có baseline để tránh cảnh báo ngay.
                deviations = {'forward_lean': 0.0, 'lateral_tilt': 0.0, 'shoulder_imbalance': 0.0}
                if metrics_smooth is not None and st.session_state.baseline['forward_lean'] is not None:
                    deviations['forward_lean'] = (
                        metrics_smooth['forward_ratio_smooth']
                        - st.session_state.baseline['forward_lean'])
                    # Hiệu góc ngắn nhất (từ utils.py) — không bị nhiễu khi qua ±180 độ.
                    deviations['lateral_tilt'] = shortest_angle_diff(
                        metrics_smooth['lateral_tilt_smooth'],
                        st.session_state.baseline['lateral_tilt'])
                    deviations['shoulder_imbalance'] = shortest_angle_diff(
                        metrics_smooth['shoulder_imbalance_smooth'],
                        st.session_state.baseline['shoulder_imbalance'])
                elif metrics_smooth is not None and st.session_state.baseline['forward_lean'] is None:
                    st.session_state.baseline = {
                        'forward_lean': metrics_smooth['forward_ratio_smooth'],
                        'lateral_tilt': metrics_smooth['lateral_tilt_smooth'],
                        'shoulder_imbalance': metrics_smooth['shoulder_imbalance_smooth'],
                    }
                    deviations = {'forward_lean': 0.0, 'lateral_tilt': 0.0,
                                  'shoulder_imbalance': 0.0}
                    st.info('Auto-calibrated from the current frame.')

                # ===================== Violation detection =====================
                violating = False
                # Forward lean: chỉ vi phạm khi tỷ lệ TĂNG so với baseline.
                if deviations['forward_lean'] > forward_thresh:
                    violating = True
                if deviations['lateral_tilt'] > lateral_thresh:
                    violating = True
                if deviations['shoulder_imbalance'] > shoulder_thresh:
                    violating = True

                # Cập nhật debounce limit từ UI.
                try:
                    st.session_state.debounce_limit = int(debounce_frames)
                except Exception:
                    pass

                # Debounce: tăng hoặc reset bộ đếm.
                if violating:
                    st.session_state.debounce_counter += 1
                else:
                    st.session_state.debounce_counter = 0
                    st.session_state.alert_active = False

                if st.session_state.debounce_counter >= st.session_state.debounce_limit:
                    st.session_state.alert_active = True


# ===================== Drawing =====================
                frame_drawn = frame.copy()
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose
                if results.pose_landmarks:
                    if st.session_state.alert_active:
                        l_spec = mp_drawing.DrawingSpec(
                            color=LANDMARK_COLOR_ALERT, thickness=LANDMARK_THICKNESS,
                            circle_radius=LANDMARK_CIRCLE_RADIUS)
                        c_spec = mp_drawing.DrawingSpec(
                            color=LANDMARK_COLOR_ALERT, thickness=LANDMARK_THICKNESS)
                    else:
                        l_spec = mp_drawing.DrawingSpec(
                            color=LANDMARK_COLOR_OK, thickness=LANDMARK_THICKNESS,
                            circle_radius=LANDMARK_CIRCLE_RADIUS)
                        c_spec = mp_drawing.DrawingSpec(
                            color=LANDMARK_COLOR_OK, thickness=LANDMARK_THICKNESS)
                    mp_drawing.draw_landmarks(
                        frame_drawn, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=l_spec, connection_drawing_spec=c_spec)

                frame_drawn = draw_guides(frame_drawn, pts, metrics_smooth)

                # Overlay: 'No pose detected' và 'BAD POSTURE!' (màu/scale từ config.py).
                if metrics_smooth is None:
                    draw_text_with_bg(frame_drawn, 'No pose detected', (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, WARN_FONT_SCALE,
                                      COLOR_ALERT, WARN_FONT_THICKNESS)

                if st.session_state.alert_active:
                    draw_text_with_bg(frame_drawn, 'BAD POSTURE!',
                                      (int(image_w / 4), int(image_h / 2)),
                                      cv2.FONT_HERSHEY_DUPLEX, ALERT_FONT_SCALE,
                                      COLOR_ALERT, ALERT_FONT_THICKNESS)

                # ===================== Posture Score =====================
                # Điểm 0-100 từ độ lệch EMA-smoothed so với ngưỡng (utils.metric_score).
                score = 100
                if metrics_smooth is not None:
                    s1 = metric_score(deviations['forward_lean'], forward_thresh)
                    s2 = metric_score(deviations['lateral_tilt'], lateral_thresh)
                    s3 = metric_score(deviations['shoulder_imbalance'], shoulder_thresh)
                    score = int((s1 + s2 + s3) / 3.0)
                    posture_score_smooth = st.session_state.ema_smoother.smooth_score(score)
                else:
                    posture_score_smooth = float(score)

                score_text.metric(label='Posture Score (0-100)',
                                  value=int(round(posture_score_smooth)))

                # Cập nhật live trend chart 60 giây.
                append_score_sample(st.session_state.score_history, posture_score_smooth)
                trend_chart.line_chart(score_trend_dataframe(st.session_state.score_history),
                                       height=TREND_CHART_HEIGHT)

                # Live metric cards với value + delta so với baseline.
                if metrics_smooth is not None and st.session_state.baseline['forward_lean'] is not None:
                    m_forward.metric(label='Forward Ratio',
                                     value=f"{metrics_smooth['forward_ratio_smooth']:.3f}",
                                     delta=f"{deviations['forward_lean']:+.3f}")
                    m_lateral.metric(label='Lateral Tilt (deg)',
                                     value=f"{metrics_smooth['lateral_tilt_smooth']:.1f}",
                                     delta=f"{deviations['lateral_tilt']:+.1f}")
                    m_shoulder.metric(label='Shoulder Imbalance (deg)',
                                      value=f"{metrics_smooth['shoulder_imbalance_smooth']:.1f}",
                                      delta=f"{deviations['shoulder_imbalance']:+.1f}")
                elif metrics_smooth is not None:
                    m_forward.metric(label='Forward Ratio',
                                     value=f"{metrics_smooth['forward_ratio_smooth']:.3f}",
                                     delta="n/a")
                    m_lateral.metric(label='Lateral Tilt (deg)',
                                     value=f"{metrics_smooth['lateral_tilt_smooth']:.1f}",
                                     delta="n/a")
                    m_shoulder.metric(label='Shoulder Imbalance (deg)',
                                      value=f"{metrics_smooth['shoulder_imbalance_smooth']:.1f}",
                                      delta="n/a")
                else:
                    m_forward.metric(label='Forward Ratio', value="--")
                    m_lateral.metric(label='Lateral Tilt (deg)', value="--")
                    m_shoulder.metric(label='Shoulder Imbalance (deg)', value="--")


# Warning text.
                if st.session_state.alert_active:
                    warning_text.markdown(
                        f"**ALERT:** Bad posture for {st.session_state.debounce_counter} consecutive frames")
                elif violating:
                    warning_text.info(
                        f"Bad posture detected (counting: {st.session_state.debounce_counter})")
                else:
                    warning_text.success('Good Posture')

                # ---------------- Terminal status log (chỉ in khi trạng thái ĐỔI) ----------------
                if metrics_smooth is not None:
                    current_posture = 'BAD' if st.session_state.alert_active else 'GOOD'
                    if current_posture != st.session_state.last_logged_posture:
                        terminal_score = int(round(posture_score_smooth))
                        if current_posture == 'BAD':
                            log_posture_status_bad(terminal_score)
                        else:
                            log_posture_status_good(terminal_score)
                        st.session_state.last_logged_posture = current_posture

                # ---------------- Database logging (throttle + status change) ----------------
                # Ghi log DB khi có pose + (status chuyển Good<->Bad HOẶC hết chu kỳ throttle)
                # để tránh làm phình database.
                if metrics_smooth is not None:
                    posture_status = 'Bad' if violating else 'Good'
                    error_type = determine_error_type(
                        deviations['forward_lean'] > forward_thresh,
                        deviations['lateral_tilt'] > lateral_thresh,
                        deviations['shoulder_imbalance'] > shoulder_thresh)
                    now = time.time()
                    status_changed = st.session_state.last_log_status != posture_status
                    throttle_elapsed = (now - st.session_state.last_log_time) >= LOG_SAVE_INTERVAL_SECONDS
                    if status_changed or throttle_elapsed:
                        try:
                            save_log(
                                status=posture_status,
                                score=int(round(posture_score_smooth)),
                                error_type=error_type,
                                db_path=DATABASE_PATH,
                            )
                        except Exception as exc:
                            print(f'{POSTURE_LOG_PREFIX} DB write failed: {exc}')
                        st.session_state.last_log_time = now
                        st.session_state.last_log_status = posture_status

                # Hiển thị frame (BGR -> RGB cho Streamlit).
                frame_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels='RGB')

                # Lưu frame/metrics/số liệu cuối để khôi phục UI sau khi Stop.
                st.session_state.last_frame = frame_rgb
                st.session_state.last_metrics = (
                    metrics_smooth.copy() if metrics_smooth is not None else None)
                st.session_state.last_deviations = {k: float(v) for k, v in deviations.items()}
                st.session_state.last_score = posture_score_smooth

                # sleep nhỏ giới hạn frame rate của while loop.
                time.sleep(LOOP_SLEEP_SECONDS)


# ---- After the while loop exits ----
            if not frame_ok:
                st.session_state.camera_running = False
                st.error('Could not read a frame from the camera.')

            # cap.release() + cv2.destroyAllWindows() ngay sau vòng lặp.
            if not st.session_state.camera_running:
                cap.release()
                cv2.destroyAllWindows()
                st.session_state.cap = None
                if st.session_state.pose is not None:
                    try:
                        st.session_state.pose.close()
                    except Exception:
                        pass
                    st.session_state.pose = None

            # Tiếp tục stream ở lần rerun kế tiếp khi cờ còn bật.
            if st.session_state.camera_running:
                st.rerun()
    else:
        # Camera off: giải phóng tài nguyên, khôi phục frame/metrics cuối để UI không trống.
        release_resources()

        if st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels='RGB')
        else:
            frame_placeholder.image(
                np.zeros((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, PLACEHOLDER_CHANNELS),
                         dtype=np.uint8))

        # Khôi phục trạng thái: giữ score/metrics/baseline cuối thay vì reset.
        last_metrics = st.session_state.last_metrics
        last_devs = st.session_state.last_deviations

        if st.session_state.last_score is not None:
            score_text.metric(label='Posture Score (0-100)',
                              value=int(round(st.session_state.last_score)))
        else:
            score_text.write('Camera is off. Turn on the camera to start monitoring.')

        if last_metrics is not None and st.session_state.baseline['forward_lean'] is not None and last_devs is not None:
            m_forward.metric(label='Forward Ratio',
                             value=f"{last_metrics['forward_ratio_smooth']:.3f}",
                             delta=f"{last_devs['forward_lean']:+.3f}")
            m_lateral.metric(label='Lateral Tilt (deg)',
                             value=f"{last_metrics['lateral_tilt_smooth']:.1f}",
                             delta=f"{last_devs['lateral_tilt']:+.1f}")
            m_shoulder.metric(label='Shoulder Imbalance (deg)',
                              value=f"{last_metrics['shoulder_imbalance_smooth']:.1f}",
                              delta=f"{last_devs['shoulder_imbalance']:+.1f}")
        elif last_metrics is not None:
            m_forward.metric(label='Forward Ratio',
                             value=f"{last_metrics['forward_ratio_smooth']:.3f}", delta="n/a")
            m_lateral.metric(label='Lateral Tilt (deg)',
                             value=f"{last_metrics['lateral_tilt_smooth']:.1f}", delta="n/a")
            m_shoulder.metric(label='Shoulder Imbalance (deg)',
                              value=f"{last_metrics['shoulder_imbalance_smooth']:.1f}", delta="n/a")
        else:
            m_forward.metric(label='Forward Ratio', value="--")
            m_lateral.metric(label='Lateral Tilt (deg)', value="--")
            m_shoulder.metric(label='Shoulder Imbalance (deg)', value="--")

        # Giữ biểu đồ trend 60 giây thu thập trong phiên làm việc.
        if st.session_state.score_history:
            trend_chart.line_chart(score_trend_dataframe(st.session_state.score_history),
                                   height=TREND_CHART_HEIGHT)
        else:
            trend_chart.caption('Start the camera to collect the score trend.')


# --------------------------- Analytics Dashboard (Tab 2) ---------------------------


def render_dashboard_tab():
    """Tab 2: Analytics Dashboard — thống kê dữ liệu tư thế lịch sử từ SQLite.

    Gồm 3 biểu đồ plotly.express: Pie (tỷ lệ Good/Bad), Bar (số lần vi phạm theo
    loại lỗi), Line (biến thiên Posture Score theo thời gian) + nút Clear Data.
    """
    st.subheader('📊 Analytics Dashboard')
    st.caption('Dữ liệu lịch sử được thu thập từ Real-time Monitor (mỗi 2 giây / mỗi trạng thái chuyển đổi).')

    df = get_session_data()

    if df.empty:
        st.info('Chưa có dữ liệu tư thế. Hãy chạy Real-time Monitor một lúc để hệ thống thu thập dữ liệu.')
        return

    # ---- Tóm tắt nhanh ----
    total = len(df)
    good_count = int((df['status'] == 'Good').sum())
    bad_count = total - good_count
    avg_score = float(df['score'].mean())
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    col_sum1.metric('Tổng số log', f'{total}')
    col_sum2.metric('Good Posture', f'{good_count} ({good_count / total * 100:.0f}%)')
    col_sum3.metric('Bad Posture', f'{bad_count} ({bad_count / total * 100:.0f}%)')
    col_sum4.metric('Score trung bình', f'{avg_score:.1f}')
    st.markdown('---')

    # ---- 1) Pie Chart: tỷ lệ thời gian ngồi đúng/sai ----
    pie_df = df['status'].value_counts().rename_axis('status').reset_index(name='count')
    fig_pie = px.pie(
        pie_df,
        names='status',
        values='count',
        title='⏱️ Tỷ lệ thời gian tư thế đúng/sai',
        color='status',
        color_discrete_map={'Good': '#2ecc71', 'Bad': '#e74c3c'},
        hole=0.4,
    )
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(legend_title_text='Tư thế')

    # ---- 2) Bar Chart: số lần vi phạm theo loại lỗi ----
    bad_df = df[df['status'] == 'Bad']
    fig_bar = None
    if not bad_df.empty:
        error_counts = (bad_df['error_type']
                        .str.split(' + ')
                        .explode()
                        .value_counts()
                        .rename_axis('error_type')
                        .reset_index(name='count'))
        fig_bar = px.bar(
            error_counts,
            x='error_type',
            y='count',
            color='error_type',
            title='🔍 Số lần vi phạm theo loại lỗi',
            labels={'error_type': 'Loại lỗi', 'count': 'Số lần vi phạm'},
        )
        fig_bar.update_layout(showlegend=False)

    # ---- 3) Line Chart: Posture Score theo thời gian ----
    line_df = df.sort_values('timestamp')
    fig_line = px.line(
        line_df,
        x='timestamp',
        y='score',
        markers=True,
        title='📈 Biến thiên Posture Score theo thời gian',
        labels={'timestamp': 'Thời gian', 'score': 'Posture Score'},
    )
    fig_line.update_traces(line_color='#3498db')
    fig_line.update_layout(yaxis_range=[0, 100])

    # ---- Render charts ----
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(fig_pie, use_container_width=True)
        if fig_bar is not None:
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info('Chưa có lần vi phạm nào để thống kê.')
    with chart_right:
        st.plotly_chart(fig_line, use_container_width=True)
        # Nội dung phụ: bảng log gần nhất để người dùng theo dõi chi tiết.
        st.markdown('**🧾 Log gần nhất (5 dòng)**')
        st.dataframe(df.tail(5)[['timestamp', 'status', 'score', 'error_type']],
                     use_container_width=True)

    # ---- Clear Data ----
    st.markdown('---')
    if st.button('🗑️ Clear Data', type='secondary'):
        clear_all_data()
        # Reset throttle để phiên giám sát mới ghi lại từ đầu.
        st.session_state.last_log_time = 0.0
        st.session_state.last_log_status = None
        st.rerun()


# --------------------------- Entry point ---------------------------

st.set_page_config(page_title='Posture Monitoring', layout='wide')
st.title('Seated Posture Monitoring - 3D Measurements')

ensure_session_state_keys()
init_db()  # Đảm bảo bảng posture_logs tồn tại ngay khi app khởi động.

controls = render_sidebar()

# Chia giao diện chính thành 2 tabs.
tab_monitor, tab_dashboard = st.tabs(['🎥 Real-time Monitor', '📊 Analytics Dashboard'])

with tab_monitor:
    render_monitor_tab(controls)

with tab_dashboard:
    render_dashboard_tab()