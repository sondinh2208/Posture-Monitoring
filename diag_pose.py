"""Diagnostic: test webcam + MediaPipe Pose detection outside the Streamlit app.

Run:  python diag_pose.py
Prints camera read stats, frame brightness and MediaPipe pose detection rate
to isolate whether 'No pose detected' comes from the camera signal or the model.
"""
import sys
import time

import cv2
import mediapipe as mp

print('python:', sys.version.split()[0])
print('mediapipe:', mp.__version__)
print('opencv:', cv2.__version__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.5)

if not cap.isOpened():
    print('CAMERA FAILED TO OPEN (busy or no permission?)')
    sys.exit(1)

# --- warm-up reads ---
ok_count = fail_count = 0
frame = None
for _ in range(30):
    s, f = cap.read()
    if s:
        ok_count += 1
        frame = f
    else:
        fail_count += 1
    time.sleep(0.03)
print('warm-up read ok/fail:', ok_count, '/', fail_count)

if frame is None:
    print('NO FRAME AT ALL from camera')
    cap.release()
    sys.exit(1)

print('frame shape:', frame.shape, '| mean brightness:', round(float(frame.mean()), 1))

pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
res = pose.process(rgb)
print('single-frame pose_landmarks found:', bool(res.pose_landmarks))
if res.pose_landmarks:
    lms = res.pose_landmarks.landmark
    print('n landmarks:', len(lms))
    for name in ['LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'NOSE']:
        lm = lms[mp.solutions.pose.PoseLandmark[name].value]
        print(' ', name, 'x=%.3f y=%.3f vis=%.3f' % (lm.x, lm.y, lm.visibility))

# --- detection rate over 20 frames ---
det = tot = 0
for _ in range(20):
    s, f = cap.read()
    if not s:
        continue
    tot += 1
    r = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if r.pose_landmarks:
        det += 1
    time.sleep(0.03)
print('detection rate: %d/%d' % (det, tot))

pose.close()
cap.release()
cv2.destroyAllWindows()
print('DONE')
