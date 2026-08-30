# -*- coding: utf-8 -*-
"""
app.py
======
Ứng dụng Streamlit giám sát tư thế ngồi (Seated Posture Monitoring) - phiên bản rút gọn.

Sau khi tái cấu trúc:
  - Cấu hình, màu sắc, ngưỡng  -> config.py
  - Tính toán hình học, lọc EMA -> utils.py
  - File này chỉ còn: UI (Streamlit), vòng lặp camera, vẽ overlay.
"""

import os
import time
import warnings
from collections import deque

# ------------------------------------------------------------------
# Tắt các cảnh báo / log hệ thống không cần thiết.
# BẮT BUỘC đặt TRƯỚC khi import mediapipe (mediapipe kéo theo tensorflow).
# ------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
# Chặn cảnh báo: 'SymbolDatabase.GetPrototype() is deprecated'
# (phát ra từ mediapipe/tensorflow mỗi lần tạo model pose).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Giảm log TensorFlow xuống mức tối thiểu:
# 0 = tất cả, 1 = info, 2 = warning+error, 3 = chỉ error.
# ------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
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


# --------------------- Terminal status logging (ANSI colors, in khi state đổi) ---------------------

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


# --------------------------- Streamlit UI & Main Loop ---------------------------

st.set_page_config(page_title='Posture Monitoring', layout='wide')
st.title('Seated Posture Monitoring - 3D Measurements')

ensure_session_state_keys()

# Sidebar controls
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
            if deviations['shoulder_imbalance'] > shoulder_thresh_new:
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
            # Điểm 0-100 từ độ lệch EMA-smoothed so với ngưỡng (hàm metric_score trong utils.py).
            score = 100
            if metrics_smooth is not None:
                s1 = metric_score(deviations['forward_lean'], forward_thresh)
                s2 = metric_score(deviations['lateral_tilt'], lateral_thresh)
                s3 = metric_score(deviations['shoulder_imbalance'], shoulder_thresh_new)
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

            # ---------------- Terminal status log (chỉ in khi trạng thái Good/Bad ĐỔI) ----------------
            # Ghi log ra Terminal chỉ khi phát hiện tư thế và trạng thái thay đổi,
            # tránh in lặp liên tục gây tràn màn hình Terminal.
            if metrics_smooth is not None:
                current_posture = 'BAD' if st.session_state.alert_active else 'GOOD'
                if current_posture != st.session_state.last_logged_posture:
                    terminal_score = int(round(posture_score_smooth))
                    if current_posture == 'BAD':
                        log_posture_status_bad(terminal_score)
                    else:
                        log_posture_status_good(terminal_score)
                    st.session_state.last_logged_posture = current_posture

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