
import math
import time
from collections import deque
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp

# --------------------------- Helper functions ---------------------------

def ensure_session_state_keys():
    """Initialize expected keys in st.session_state with sensible defaults."""
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'pose' not in st.session_state:
        st.session_state.pose = None
    if 'baseline' not in st.session_state:
        # baseline for the three new metrics
        st.session_state.baseline = {'forward_lean': None, 'lateral_tilt': None, 'shoulder_imbalance': None}
    if 'debounce_counter' not in st.session_state:
        st.session_state.debounce_counter = 0
    if 'debounce_limit' not in st.session_state:
        st.session_state.debounce_limit = 8  # default frames
    if 'alert_active' not in st.session_state:
        st.session_state.alert_active = False
    # --- Moving Average Filter buffers (10-frame window) ---
    # Store the 10 most recent values to filter noise on each measurement axis.
    if 'lean_buffer' not in st.session_state:
        st.session_state.lean_buffer = deque(maxlen=10)       # Forward Lean (face/shoulder ratio)
    if 'tilt_buffer' not in st.session_state:
        st.session_state.tilt_buffer = deque(maxlen=10)        # Lateral Tilt (lateral_deg)
    if 'imbalance_buffer' not in st.session_state:
        st.session_state.imbalance_buffer = deque(maxlen=10)   # Shoulder Imbalance (shoulder_deg)
    # --- EMA (Exponential Moving Average) smoother state ---
    # Keeps the previous smoothed value per metric so each new frame is blended with it:
    # Value_smooth = alpha * Value_current + (1 - alpha) * Value_previous
    if 'ema_smoother' not in st.session_state:
        st.session_state.ema_smoother = EMASmoother(alpha=EMA_ALPHA)
    # --- Live trend chart history: (timestamp, posture_score_smooth) samples ---
    if 'score_history' not in st.session_state:
        st.session_state.score_history = deque(maxlen=TREND_MAX_SAMPLES)
    # Last-seen state: keeps the last video frame & metrics so the UI can render them
    # after the camera is stopped instead of showing a blank/black screen.
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None        # RGB ndarray ready for image(..., channels='RGB')
    if 'last_metrics' not in st.session_state:
        st.session_state.last_metrics = None      # smoothed dict metrics (forward_lean/lateral_tilt/shoulder_imbalance)
    if 'last_deviations' not in st.session_state:
        st.session_state.last_deviations = None   # deviations dict for delta display
    if 'last_score' not in st.session_state:
        st.session_state.last_score = None        # last computed posture score


def release_resources():
    """Release cv2 and mediapipe resources stored in session_state."""
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
    """Callback for the 'Start Camera' button: turn the camera_running flag on."""
    st.session_state.camera_running = True


def stop_camera():
    """Callback for the 'Stop Camera' button: stop streaming and fully release the webcam.

    Runs inside a widget callback (before widgets are re-instantiated) so the flag can be
    updated safely. release_resources() calls cap.release() + cv2.destroyAllWindows() to
    close the webcam hardware thoroughly.
    """
    release_resources()
    st.session_state.camera_running = False
    # Reset the EMA history so the next camera session starts with a direct assignment
    # (Value_smooth = Value_current on its first frame) instead of blending with stale values.
    if st.session_state.get('ema_smoother') is not None:
        st.session_state.ema_smoother.reset()


def open_camera(device_index=0):
    """Open camera and return cv2.VideoCapture. Use DirectShow backend on Windows for reliability."""
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def landmarks_to_dict(landmarks, image_w, image_h):
    """Convert normalized landmarks to dict with both normalized and pixel coordinates.
    Returns: dict[name] = {'nx':..,'ny':..,'nz':..,'x':int,'y':int}
    """
    pts = {}
    for lm_enum in mp.solutions.pose.PoseLandmark:
        idx = lm_enum.value
        lm = landmarks[idx]
        nx, ny, nz = lm.x, lm.y, lm.z
        px = int(nx * image_w)
        py = int(ny * image_h)
        pts[lm_enum.name] = {'nx': nx, 'ny': ny, 'nz': nz, 'x': px, 'y': py}
    return pts


def line_tilt_deg(dx, dy):
    """Absolute tilt of an UNDIRECTED line vs the horizontal axis, in [0, 90] degrees.

    atan2(dy, dx) returns a SIGNED direction in (-180, 180]; a line, however, has no
    direction, so any angle beyond +/-90 deg folds back onto its equivalent inclination
    (e.g. 180 deg == 0 deg, -170 deg == 10 deg). The result is ~0 deg for a level line and
    grows as the line tilts, so a normalized straight posture reads ~0 deg and typical
    seated-posture values fall in the 0-45 deg range of the threshold sliders.
    Pass SAME-UNIT deltas (pixel coords) so the angle is the true geometric angle
    regardless of the camera aspect ratio.
    """
    theta = math.degrees(math.atan2(dy, dx))
    if theta > 90.0:
        theta -= 180.0
    elif theta < -90.0:
        theta += 180.0
    return abs(theta)


def compute_posture_metrics_3d(pts):
    """Compute the three required metrics.
    Returns dict: {'forward_lean': float, 'lateral_tilt': float, 'shoulder_imbalance': float}

    Definitions:
    - forward_lean (2D Perspective Ratio): forward_ratio = face_width / shoulder_width.
      face_width is the Euclidean distance between LEFT_EAR and RIGHT_EAR (falls back to
      LEFT_EYE/RIGHT_EYE if ears are occluded/missing); shoulder_width is the Euclidean
      distance between LEFT_SHOULDER and RIGHT_SHOULDER. Using pixel coords (x, y).
      When the head moves forward toward the camera, the face appears larger, so the
      ratio increases and the forward-lean error can be detected.
    - lateral_tilt (face rotation, deg): ABSOLUTE inclination of the line connecting
      LEFT_EAR and RIGHT_EAR vs the horizontal axis (undirected line folded into [0, 90]).
      ~0 deg when the head is straight/upright and grows as the head tilts sideways (even if
      the nose stays on the vertical spine axis). Falls back to LEFT_EYE/RIGHT_EYE if ears
      are occluded/missing. Computed in pixel coords so the angle is the true geometric angle.
    - shoulder_imbalance (degrees): ABSOLUTE inclination of the shoulder line vs the
      horizontal axis (undirected line folded into [0, 90]). 0 deg = shoulders level;
      larger = one shoulder higher than the other. Computed in pixel coords.

    Normalization: both angle metrics read ~0 deg for a straight posture and typical
    bad-posture values fall inside the 0-45 deg range of the threshold sliders.
    """
    try:
        left_sh = pts['LEFT_SHOULDER']
        right_sh = pts['RIGHT_SHOULDER']
    except KeyError:
        return None

    # Perceptual reference points: prefer ears, fall back to eyes if ears are occluded/missing.
    left_ref = pts.get('LEFT_EAR') or pts.get('LEFT_EYE')
    right_ref = pts.get('RIGHT_EAR') or pts.get('RIGHT_EYE')
    if left_ref is None or right_ref is None:
        return None

    # 1) Forward lean: 2D perspective ratio face_width / shoulder_width (pixel coords x, y).
    face_width = math.hypot(right_ref['x'] - left_ref['x'], right_ref['y'] - left_ref['y'])
    shoulder_width = math.hypot(right_sh['x'] - left_sh['x'], right_sh['y'] - left_sh['y'])
    forward_ratio = face_width / shoulder_width if shoulder_width > 0 else 0.0

    # 2) Lateral tilt: ABSOLUTE inclination of the ear/eye line vs horizontal, normalized so
    # a straight head reads ~0 deg (0..90 range; typical seated postures stay within 0..45,
    # matching the threshold sliders). Pixel coords keep the angle geometrically true —
    # normalized coords would stretch the angle by the camera aspect ratio (w/h).
    dx = right_ref['x'] - left_ref['x']
    dy = right_ref['y'] - left_ref['y']
    lateral_deg = line_tilt_deg(dx, dy)

    # 3) Shoulder imbalance: ABSOLUTE inclination of the shoulder line vs horizontal,
    # normalized the same way (0 deg = level shoulders; 0..90 range). Pixel coords.
    sdx = right_sh['x'] - left_sh['x']
    sdy = right_sh['y'] - left_sh['y']
    shoulder_deg = line_tilt_deg(sdx, sdy)

    return {'forward_lean': forward_ratio, 'lateral_tilt': lateral_deg, 'shoulder_imbalance': shoulder_deg}


def shortest_angle_diff(current_deg, baseline_deg):
    """Return the shortest angular difference (deg) between current and baseline angles.

    Handles the -180..180 wrap-around of atan2: naively subtracting two angles that sit
    near the +180/-180 boundary (e.g. current=179, baseline=-179) yields ~358 deg instead
    of the true 2 deg, which triggers false alerts.

    diff = (current_angle - baseline_angle + 180) % 360 - 180
    Returns absolute_diff = abs(diff) in [0, 180].
    """
    diff = (current_deg - baseline_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


# --------------------- EMA (Exponential Moving Average) Filter ---------------------

# Smoothing factor alpha in (0, 1]:
#   Value_smooth = alpha * Value_current + (1 - alpha) * Value_previous
# alpha ~ 0.3 -> responds quickly to real posture changes while suppressing
# frame-to-frame jitter of the MediaPipe landmarks. Lower = smoother/slower.
EMA_ALPHA = 0.3


def shortest_signed_angle_diff(current_deg, reference_deg):
    """Signed shortest angular difference (deg), result in [-180, 180).

    Same atan2 wrap-around handling as shortest_angle_diff() but keeps the sign so the
    EMA can move an angle the short way across the +180/-180 boundary:
    diff = (current - reference + 180) % 360 - 180.
    """
    return (current_deg - reference_deg + 180.0) % 360.0 - 180.0


def ema_smooth(current, previous, alpha=EMA_ALPHA):
    """Plain EMA on a scalar quantity.

    Value_smooth = alpha * Value_current + (1 - alpha) * Value_previous
    First frame (previous is None): the current value is assigned directly
    (Value_smooth = Value_current) so the filter starts without a ramp-up lag.
    """
    if previous is None:
        return float(current)
    return alpha * float(current) + (1.0 - alpha) * float(previous)


def ema_smooth_angle(current_deg, previous_deg, alpha=EMA_ALPHA):
    """Wrap-safe EMA for angle quantities in degrees (atan2 range -180..180).

    Naively blending e.g. current=179 deg with previous=-179 deg would yield ~0 deg;
    instead we step from the previous angle along the shortest signed difference
    toward the current one: previous + alpha * shortest_signed_angle_diff(current, previous).
    """
    if previous_deg is None:
        return float(current_deg)
    return previous_deg + alpha * shortest_signed_angle_diff(current_deg, previous_deg)


class EMASmoother:
    """Per-metric EMA state kept in st.session_state (in-memory, per browser session).

    Stores the previous smoothed value for every filtered quantity:
      - 'forward_ratio'       -> rendered as forward_ratio_smooth
      - 'lateral_tilt'        -> rendered as lateral_tilt_smooth
      - 'shoulder_imbalance'  -> rendered as shoulder_imbalance_smooth
      - 'posture_score'       -> rendered as posture_score_smooth

    First frame, or the first frame after reset() (Calibrate / camera restart):
    Value_smooth = Value_current (direct assignment, no blending with stale data).
    """

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = float(alpha)
        self.prev_values = {}  # metric key -> last smoothed value (missing key = first frame)

    def reset(self):
        """Forget all EMA history so the next smooth() call assigns the current value directly."""
        self.prev_values = {}

    def smooth(self, key, value, angular=False):
        """EMA-blend `value` for `key`, store it as the new previous value and return it."""
        previous = self.prev_values.get(key)
        if previous is None:
            # First frame (or right after Calibrate/reset): direct assignment.
            smoothed = float(value)
        elif angular:
            smoothed = ema_smooth_angle(value, previous, self.alpha)
        else:
            smoothed = ema_smooth(value, previous, self.alpha)
        self.prev_values[key] = smoothed
        return smoothed

    def smooth_metrics(self, metrics):
        """Apply EMA to the three posture metrics (after the moving-average filter).

        metrics: dict from compute_posture_metrics_3d / MA filter, or None (no pose).
        Returns the dict with the *_smooth keys used for rendering, or None untouched.
        """
        if metrics is None:
            return None
        return {
            'forward_ratio_smooth': self.smooth('forward_ratio', metrics['forward_lean']),
            # Angle metrics are magnitudes in [0, 90] (no wrap possible); the angular EMA
            # reduces to the plain EMA and is kept purely as a safety net.
            'lateral_tilt_smooth': self.smooth('lateral_tilt', metrics['lateral_tilt'], angular=True),
            'shoulder_imbalance_smooth': self.smooth('shoulder_imbalance', metrics['shoulder_imbalance'], angular=True),
        }

    def smooth_score(self, score):
        """EMA-blend the 0-100 posture score; returns posture_score_smooth."""
        return self.smooth('posture_score', score)


# --------------------- Live Trend Chart (Posture Score, last 60 s) ---------------------

# Width of the rolling window shown in the live trend chart, in seconds.
TREND_WINDOW_SECONDS = 60.0
# Hard cap on stored samples (enough for 60 s even at ~10 samples/s); samples are also
# pruned by timestamp in append_score_sample() so the window always spans 60 s max.
TREND_MAX_SAMPLES = 600


def append_score_sample(history, score, now=None):
    """Append a (timestamp, score) sample and prune samples older than the 60 s window."""
    if now is None:
        now = time.time()
    history.append((float(now), float(score)))
    cutoff = float(now) - TREND_WINDOW_SECONDS
    while history and history[0][0] < cutoff:
        history.popleft()
    return history


def score_trend_dataframe(history):
    """Convert (timestamp, score) samples into a DataFrame for st.line_chart.

    The DatetimeIndex makes the x-axis show real time, so pauses (camera stopped)
    stay visible and the chart always covers the last 60 seconds of wall-clock time.
    """
    if not history:
        return pd.DataFrame({'Posture Score': []})
    times = [t for t, _ in history]
    scores = [s for _, s in history]
    return pd.DataFrame({'Posture Score': scores}, index=pd.to_datetime(times, unit='s'))


def draw_text_with_bg(frame, text, org, font, scale, color, thickness, bg_color=(0, 0, 0), bg_alpha=0.5):
    """Draw text on the frame with a semi-transparent black rectangle behind it for readability.

    A translucent cv2.rectangle is blended (alpha ~0.5) underneath the text so the labels
    stay readable on any background. The geometry is computed from cv2.getTextSize and
    clipped to the frame bounds.
    """
    h, w = frame.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    # Expand the box slightly around the text and clip to the frame bounds
    p1 = (max(0, x - 5), max(0, y - th - 6))
    p2 = (min(w - 1, x + tw + 5), min(h - 1, y + baseline + 2))
    # Semi-transparent overlay (alpha blend)
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, dst=frame)
    # Draw the text on top of the translucent box
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    return frame


def draw_guides(frame, pts, metrics):
    """Draw vertical line from nose to shoulder midpoint and horizontal shoulder line for visualization.
    pts: pixel coords dict from landmarks_to_dict.
    metrics: dict computed by compute_posture_metrics_3d (for coloring decisions)
    """
    if not pts:
        return frame
    h, w = frame.shape[:2]
    color = (0, 255, 0) if not st.session_state.alert_active else (0, 0, 255)

    # Draw head-tilt line (YELLOW) connecting the two ears to visualize the face angle.
    # If ears are occluded/missing, fall back to the eyes.
    try:
        left_ref = pts.get('LEFT_EAR') or pts.get('LEFT_EYE')
        right_ref = pts.get('RIGHT_EAR') or pts.get('RIGHT_EYE')
        if left_ref is not None and right_ref is not None:
            pt_left = (left_ref['x'], left_ref['y'])
            pt_right = (right_ref['x'], right_ref['y'])
            cv2.line(frame, pt_left, pt_right, (0, 255, 255), 2)  # yellow
    except Exception:
        pass

    # Draw shoulder line (pixel)
    try:
        left_sh = pts['LEFT_SHOULDER']
        right_sh = pts['RIGHT_SHOULDER']
        pt_left = (left_sh['x'], left_sh['y'])
        pt_right = (right_sh['x'], right_sh['y'])
        cv2.line(frame, pt_left, pt_right, color, 2)
    except Exception:
        pass

    # Draw vertical line from nose to shoulder midpoint
    try:
        nose = pts['NOSE']
        sh_mid_px = (int((left_sh['x'] + right_sh['x']) / 2), int((left_sh['y'] + right_sh['y']) / 2))
        pt_nose = (nose['x'], nose['y'])
        cv2.line(frame, pt_nose, sh_mid_px, color, 2)
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
    # Start/Stop buttons controlled by on_click callbacks + camera_running flag.
    # (No checkbox; resources are released explicitly and immediately on Stop.)
    st.button('Start Camera', type='primary', on_click=start_camera, disabled=st.session_state.camera_running)
    st.button('Stop Camera', on_click=stop_camera, disabled=not st.session_state.camera_running)
    st.caption('Status: ' + ('Running' if st.session_state.camera_running else 'Stopped'))

    st.markdown('---')
    st.subheader('Calibration')
    calibrate_btn = st.button('Calibrate')

    st.markdown('---')
    # Threshold sliders grouped inside an expander for a cleaner sidebar
    with st.expander('Threshold Settings', expanded=True):
        forward_thresh = st.slider('Forward Lean (Turtle Neck) Threshold (increase in face/shoulder ratio)', min_value=0.0, max_value=0.5, value=0.04, step=0.01, key='forward_thresh')
        lateral_thresh = st.slider('Lateral Tilt (Body Lean) Threshold (deg)', min_value=0.0, max_value=45.0, value=10.0, step=0.5, key='lateral_thresh')
        shoulder_thresh_new = st.slider('Shoulder Imbalance Threshold (deg)', min_value=0.0, max_value=45.0, value=10.0, step=0.5, key='shoulder_thresh')

    st.markdown('---')
    st.write('Debounce frames (consecutive frames before alert):')
    debounce_frames = st.number_input('Frames', min_value=1, max_value=300, value=st.session_state.debounce_limit, step=1, key='debounce_limit')

# Layout: main video + side info
col1, col2 = st.columns([3, 1])
frame_placeholder = col1.empty()

with col2:
    st.subheader('Status')
    score_text = st.empty()
    warning_text = st.empty()
    st.markdown('---')
    # Live metric cards with current value + delta vs baseline
    # (replaces the raw JSON baseline display)
    m_forward = st.empty()
    m_lateral = st.empty()
    m_shoulder = st.empty()
    st.markdown('---')
    # Live trend chart: Posture Score (0-100) over the last 60 seconds, so the user can
    # track how stable their sitting posture is at a glance.
    st.caption('Posture Score trend (last 60 s)')
    trend_chart = st.empty()
    if st.session_state.alert_active:
        st.error('ALERT: PERSISTENT BAD POSTURE!')

# Manage camera open/close
if st.session_state.camera_running:
    # Open resources if not already open
    if st.session_state.cap is None:
        st.session_state.cap = open_camera(0)
    if st.session_state.pose is None:
        st.session_state.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = st.session_state.cap
    pose = st.session_state.pose

    if not cap or not cap.isOpened():
        frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8))
        st.error('Cannot open camera. Check access permissions.')
        # Stop the flag so we don't retry the failed camera on every rerun.
        cap.release()
        cv2.destroyAllWindows()
        st.session_state.cap = None
        st.session_state.camera_running = False
    else:
        # Process a small batch of frames per rerun so the UI (Stop button) stays responsive.
        # The loop re-checks st.session_state.camera_running on every iteration and exits
        # immediately when the flag turns False.
        FRAMES_PER_RUN = 5
        frames_done = 0
        frame_ok = True
        while st.session_state.camera_running and frames_done < FRAMES_PER_RUN and frame_ok:
            success, frame = cap.read()
            if not success:
                frame_ok = False
                break
            frames_done += 1
            frame = cv2.flip(frame, 1)
            image_h, image_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            metrics = None
            pts = None
            if results.pose_landmarks:
                pts = landmarks_to_dict(results.pose_landmarks.landmark, image_w, image_h)
                metrics = compute_posture_metrics_3d(pts)

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

            # ===================== EMA (Exponential Moving Average) Filter =====================
            # Calibrate pressed: clear the EMA history first so smoothing restarts cleanly and
            # the calibration frame is assigned directly (Value_smooth = Value_current).
            if calibrate_btn:
                st.session_state.ema_smoother.reset()

            # EMA update: Value_smooth = alpha * Value_current + (1 - alpha) * Value_previous.
            # First frame (no previous value) -> direct assignment, handled inside EMASmoother.
            # Produces the *_smooth values used for BOTH rendering and threshold classification.
            metrics_smooth = st.session_state.ema_smoother.smooth_metrics(metrics)

            # If calibrate button clicked and metrics available, store baseline (smoothed values)
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

            # If not calibrated yet, auto-calibrate on first good frame to avoid immediate alerts
            deviations = {'forward_lean': 0.0, 'lateral_tilt': 0.0, 'shoulder_imbalance': 0.0}
            if metrics_smooth is not None and st.session_state.baseline['forward_lean'] is not None:
                # Deviations are computed from the EMA-smoothed values so both the threshold
                # classification (Good/Bad posture) and the displayed deltas are jitter-free.
                deviations['forward_lean'] = metrics_smooth['forward_ratio_smooth'] - st.session_state.baseline['forward_lean']
                # Angle metrics are non-negative [0, 90] magnitudes, so this angular difference
                # equals a plain difference; kept for robustness anyway.
                deviations['lateral_tilt'] = shortest_angle_diff(metrics_smooth['lateral_tilt_smooth'], st.session_state.baseline['lateral_tilt'])
                deviations['shoulder_imbalance'] = shortest_angle_diff(metrics_smooth['shoulder_imbalance_smooth'], st.session_state.baseline['shoulder_imbalance'])
            elif metrics_smooth is not None and st.session_state.baseline['forward_lean'] is None:
                st.session_state.baseline = {
                    'forward_lean': metrics_smooth['forward_ratio_smooth'],
                    'lateral_tilt': metrics_smooth['lateral_tilt_smooth'],
                    'shoulder_imbalance': metrics_smooth['shoulder_imbalance_smooth'],
                }
                deviations = {'forward_lean': 0.0, 'lateral_tilt': 0.0, 'shoulder_imbalance': 0.0}
                st.info('Auto-calibrated from the current frame.')

            # Determine violations
            violating = False
            # Forward lean: detect only when the ratio INCREASES vs baseline
            # (head moves toward the camera => face appears larger than baseline).
            if deviations['forward_lean'] > forward_thresh:
                violating = True
            # Lateral tilt: shortest angular difference (absolute_diff) vs threshold
            if deviations['lateral_tilt'] > lateral_thresh:
                violating = True
            # Shoulder imbalance: shortest angular difference (absolute_diff) vs threshold
            if deviations['shoulder_imbalance'] > shoulder_thresh_new:
                violating = True

            # Update debounce limit from UI input
            try:
                st.session_state.debounce_limit = int(debounce_frames)
            except Exception:
                pass

            # Debounce: increment or reset counter
            if violating:
                st.session_state.debounce_counter += 1
            else:
                st.session_state.debounce_counter = 0
                st.session_state.alert_active = False

            if st.session_state.debounce_counter >= st.session_state.debounce_limit:
                st.session_state.alert_active = True

            # Draw pose landmarks lightly and draw guides
            frame_drawn = frame.copy()
            # draw landmarks using MediaPipe drawing utils (green/red depending on alert)
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            if results.pose_landmarks:
                if st.session_state.alert_active:
                    l_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    c_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                else:
                    l_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                    c_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                mp_drawing.draw_landmarks(frame_drawn, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=l_spec, connection_drawing_spec=c_spec)

            # Draw vertical nose->shoulder_mid and shoulder line
            frame_drawn = draw_guides(frame_drawn, pts, metrics_smooth)

            # Video overlay: only the red 'No pose detected' hint is drawn here. The yellow
            # metric labels (Forward Ratio / Lateral Tilt / Shoulder Angle) were removed to
            # keep the video clean — the live values are shown in the Status column instead.
            if metrics_smooth is None:
                draw_text_with_bg(frame_drawn, 'No pose detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if st.session_state.alert_active:
                draw_text_with_bg(frame_drawn, 'BAD POSTURE!', (int(image_w/4), int(image_h/2)), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 4)

            # Compute a simple score 0-100 (higher = better) using thresholds
            # (computed from the EMA-smoothed deviations, so the score itself is jitter-free)
            score = 100
            if metrics_smooth is not None:
                def metric_score(dev, thresh, is_normalized=False):
                    if thresh is None or thresh == 0:
                        return 100.0
                    if is_normalized:
                        ratio = min(1.0, abs(dev) / thresh)
                    else:
                        ratio = min(1.0, abs(dev) / thresh)
                    return max(0.0, 100.0 * (1.0 - ratio))

                s1 = metric_score(deviations['forward_lean'], forward_thresh, is_normalized=True)
                s2 = metric_score(deviations['lateral_tilt'], lateral_thresh)
                s3 = metric_score(deviations['shoulder_imbalance'], shoulder_thresh_new)
                score = int((s1 + s2 + s3) / 3.0)
                # EMA-smooth the posture score as well -> posture_score_smooth.
                # Only fed when a pose is actually measured, so 'no pose' frames do not
                # drag the average toward the default 100.
                posture_score_smooth = st.session_state.ema_smoother.smooth_score(score)
            else:
                posture_score_smooth = float(score)

            score_text.metric(label='Posture Score (0-100)', value=int(round(posture_score_smooth)))

            # Record the smoothed score and refresh the live 60 s trend chart
            append_score_sample(st.session_state.score_history, posture_score_smooth)
            trend_chart.line_chart(score_trend_dataframe(st.session_state.score_history), height=200)

            # Live metric cards with current (EMA-smoothed) value + delta vs baseline
            if metrics_smooth is not None and st.session_state.baseline['forward_lean'] is not None:
                m_forward.metric(label='Forward Ratio', value=f"{metrics_smooth['forward_ratio_smooth']:.3f}", delta=f"{deviations['forward_lean']:+.3f}")
                m_lateral.metric(label='Lateral Tilt (deg)', value=f"{metrics_smooth['lateral_tilt_smooth']:.1f}", delta=f"{deviations['lateral_tilt']:+.1f}")
                m_shoulder.metric(label='Shoulder Imbalance (deg)', value=f"{metrics_smooth['shoulder_imbalance_smooth']:.1f}", delta=f"{deviations['shoulder_imbalance']:+.1f}")
            elif metrics_smooth is not None:
                # metrics available but no baseline yet
                m_forward.metric(label='Forward Ratio', value=f"{metrics_smooth['forward_ratio_smooth']:.3f}", delta="n/a")
                m_lateral.metric(label='Lateral Tilt (deg)', value=f"{metrics_smooth['lateral_tilt_smooth']:.1f}", delta="n/a")
                m_shoulder.metric(label='Shoulder Imbalance (deg)', value=f"{metrics_smooth['shoulder_imbalance_smooth']:.1f}", delta="n/a")
            else:
                m_forward.metric(label='Forward Ratio', value="--")
                m_lateral.metric(label='Lateral Tilt (deg)', value="--")
                m_shoulder.metric(label='Shoulder Imbalance (deg)', value="--")

            # Warning text
            if st.session_state.alert_active:
                warning_text.markdown(f"**ALERT:** Bad posture for {st.session_state.debounce_counter} consecutive frames")
            else:
                if violating:
                    warning_text.info(f"Bad posture detected (counting: {st.session_state.debounce_counter})")
                else:
                    warning_text.success('Good Posture')

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels='RGB')

            # Persist the latest frame/SMOOTHED metrics/score so the UI can restore them after Stop
            st.session_state.last_frame = frame_rgb
            st.session_state.last_metrics = metrics_smooth.copy() if metrics_smooth is not None else None
            st.session_state.last_deviations = {k: float(v) for k, v in deviations.items()}
            st.session_state.last_score = posture_score_smooth

            # tiny sleep (limits the frame rate of the while loop)
            time.sleep(0.02)

        # ---- After the while loop exits ----
        if not frame_ok:
            st.session_state.camera_running = False
            st.error('Could not read a frame from the camera.')

        # ---- cap.release() + cv2.destroyAllWindows() right after the loop ----
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

        # Continue streaming on a new rerun while the flag stays on
        if st.session_state.camera_running:
            st.rerun()

else:
    # Camera is off: release resources if allocated, then RESTORE the last seen
    # frame + metrics so the UI does not go blank/black.
    release_resources()

    # --- Restore the last video frame (or a black placeholder if never started) ---
    if st.session_state.last_frame is not None:
        frame_placeholder.image(st.session_state.last_frame, channels='RGB')
    else:
        frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8))

    # --- Restore Status: keep the final score/metrics/baseline instead of resetting ---
    last_metrics = st.session_state.last_metrics
    last_devs = st.session_state.last_deviations

    if st.session_state.last_score is not None:
        score_text.metric(label='Posture Score (0-100)', value=int(round(st.session_state.last_score)))
    else:
        score_text.write('Camera is off. Turn on the camera to start monitoring.')

    if last_metrics is not None and st.session_state.baseline['forward_lean'] is not None and last_devs is not None:
        m_forward.metric(label='Forward Ratio', value=f"{last_metrics['forward_ratio_smooth']:.3f}", delta=f"{last_devs['forward_lean']:+.3f}")
        m_lateral.metric(label='Lateral Tilt (deg)', value=f"{last_metrics['lateral_tilt_smooth']:.1f}", delta=f"{last_devs['lateral_tilt']:+.1f}")
        m_shoulder.metric(label='Shoulder Imbalance (deg)', value=f"{last_metrics['shoulder_imbalance_smooth']:.1f}", delta=f"{last_devs['shoulder_imbalance']:+.1f}")
    elif last_metrics is not None:
        # metrics available but no baseline yet
        m_forward.metric(label='Forward Ratio', value=f"{last_metrics['forward_ratio_smooth']:.3f}", delta="n/a")
        m_lateral.metric(label='Lateral Tilt (deg)', value=f"{last_metrics['lateral_tilt_smooth']:.1f}", delta="n/a")
        m_shoulder.metric(label='Shoulder Imbalance (deg)', value=f"{last_metrics['shoulder_imbalance_smooth']:.1f}", delta="n/a")
    else:
        m_forward.metric(label='Forward Ratio', value="--")
        m_lateral.metric(label='Lateral Tilt (deg)', value="--")
        m_shoulder.metric(label='Shoulder Imbalance (deg)', value="--")

    # --- Keep showing the 60 s Posture Score trend chart collected during the session ---
    if st.session_state.score_history:
        trend_chart.line_chart(score_trend_dataframe(st.session_state.score_history), height=200)
    else:
        trend_chart.caption('Start the camera to collect the score trend.')

# Camera start/stop is handled by the 'Start Camera' / 'Stop Camera' sidebar buttons whose
# on_click callbacks set st.session_state.camera_running and release the webcam hardware.
