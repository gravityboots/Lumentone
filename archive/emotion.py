import math
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# -------------------------
# Utility functions (from provided emotion_detect.py)
# -------------------------
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye):
    v1 = euclidean(eye[1], eye[5])
    v2 = euclidean(eye[2], eye[4])
    h = euclidean(eye[0], eye[3])
    if h == 0:
        return 0
    return (v1 + v2) / (2.0 * h)


def mouth_aspect_ratio(m):
    h = euclidean(m['left'], m['right'])
    v = euclidean(m['top'], m['bottom'])
    if h == 0:
        return 0
    return v / h


def lip_curvature(lm, left, right, top, bottom):
    left_y = lm[left][1]
    right_y = lm[right][1]
    top_y = lm[top][1]
    bottom_y = lm[bottom][1]
    corner_avg = (left_y + right_y) / 2
    mouth_center = (top_y + bottom_y) / 2
    return mouth_center - corner_avg


# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
BROW_L = 70
BROW_R = 300
CHIN = 152
FOREHEAD = 10


class FaceEmotionDetector:
    def __init__(self, width=960, height=720):
        self.width = width
        self.height = height
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.draw = mp.solutions.drawing_utils
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_mood = {
            "label": "Idle",
            "ear": None,
            "mar": None,
            "lip": None,
            "brow": None,
            "pitch": None,
            "timestamp": time.time(),
        }
        self.alpha = 0.35
        self.smooth_vals = {
            "ear": None,
            "mar": None,
            "lip": None,
            "brow": None,
            "pitch": None,
        }
        self.frame_queue = deque(maxlen=1)

    def start(self):
        if self.running:
            return True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        self.running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _smooth(self, key, new):
        old = self.smooth_vals[key]
        if old is None:
            self.smooth_vals[key] = new
        else:
            self.smooth_vals[key] = self.alpha * new + (1 - self.alpha) * old
        return self.smooth_vals[key]

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (self.width, self.height))
            processed_frame = self._process_frame(frame)
            with self.lock:
                self.latest_frame = processed_frame
            time.sleep(0.01)

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        mood_label = "Neutral"

        if result.multi_face_landmarks:
            lm = [(int(p.x * w), int(p.y * h)) for p in result.multi_face_landmarks[0].landmark]

            left_eye = [lm[i] for i in LEFT_EYE]
            right_eye = [lm[i] for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            mouth = {
                'left': lm[MOUTH_LEFT],
                'right': lm[MOUTH_RIGHT],
                'top': lm[MOUTH_TOP],
                'bottom': lm[MOUTH_BOTTOM],
            }
            mar = mouth_aspect_ratio(mouth)
            lipc = lip_curvature(lm, MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM)
            brow_dist = euclidean(lm[BROW_L], lm[BROW_R]) / w
            pitch = (lm[CHIN][1] - lm[FOREHEAD][1]) / h

            ear_s = self._smooth("ear", ear)
            mar_s = self._smooth("mar", mar)
            lip_s = self._smooth("lip", lipc)
            brow_s = self._smooth("brow", brow_dist)
            pitch_s = self._smooth("pitch", pitch)

            mood_label = self._mood_from_metrics(pitch_s, lip_s, brow_s, mar_s)

            self.latest_mood = {
                "label": mood_label,
                "ear": round(ear_s, 3) if ear_s is not None else None,
                "mar": round(mar_s, 3) if mar_s is not None else None,
                "lip": round(lip_s, 3) if lip_s is not None else None,
                "brow": round(brow_s, 3) if brow_s is not None else None,
                "pitch": round(pitch_s, 3) if pitch_s is not None else None,
                "timestamp": time.time(),
            }

            self.draw.draw_landmarks(
                frame,
                result.multi_face_landmarks[0],
                self.mp_face.FACEMESH_TESSELATION,
            )

            overlay_lines = [
                f"MOOD: {mood_label}",
                f"EAR: {ear_s:.3f}" if ear_s is not None else "EAR: --",
                f"MAR: {mar_s:.3f}" if mar_s is not None else "MAR: --",
                f"LIP: {lip_s:.2f}" if lip_s is not None else "LIP: --",
                f"BROW: {brow_s:.3f}" if brow_s is not None else "BROW: --",
                f"PITCH: {pitch_s:.3f}" if pitch_s is not None else "PITCH: --",
            ]
            for idx, text in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    text,
                    (10, 35 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if idx else 0.9,
                    (0, 255, 0) if idx == 0 else (255, 255, 0),
                    2,
                )
        else:
            cv2.putText(frame, "No face detected", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame

    @staticmethod
    def _mood_from_metrics(pitch_s, lip_s, brow_s, mar_s):
        mood = "Neutral"
        if pitch_s is None or lip_s is None or brow_s is None or mar_s is None:
            return mood
        if pitch_s > 0.22 and lip_s < -3:
            mood = "Depressed"
        elif brow_s < 0.080 and mar_s < 0.22 and lip_s < 0:
            mood = "Angry"
        elif mar_s > 0.40:
            mood = "Stressed / Anxious"
        elif lip_s > 3.5:
            mood = "Happy"
        elif lip_s < -2:
            mood = "Sad"
        else:
            mood = "Neutral"
        return mood

    def get_jpeg(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None:
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera stopped", (30, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = placeholder
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()

    def get_status(self):
        running = self.running
        mood = self.latest_mood
        return {"running": running, "mood": mood}
