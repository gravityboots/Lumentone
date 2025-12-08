import math
import threading
import time
from collections import deque
import statistics

import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
import io


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


def pos_signal(rgb_series):
    C = np.asarray(rgb_series, dtype=np.float32)
    Cn = C / (np.mean(C, axis=1, keepdims=True) + 1e-8)
    S = np.empty((2, Cn.shape[0]), dtype=np.float32)
    S[0, :] = Cn[:, 0] - Cn[:, 1]
    S[1, :] = Cn[:, 1] - Cn[:, 2]
    std0 = np.std(S[0]) + 1e-8
    std1 = np.std(S[1]) + 1e-8
    P = S[0] + (std0 / std1) * S[1]
    P = P - np.mean(P)
    return P


def bandpass(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    lown = low / nyq
    highn = high / nyq
    if lown <= 0 or highn >= 1 or lown >= highn:
        return None
    b, a = signal.butter(order, [lown, highn], btype='band')
    return signal.filtfilt(b, a, sig)


def estimate_bpm_welch_with_snr(sig, fs, low_hz, high_hz, snr_thresh):
    if len(sig) < 16:
        return None, None
    nperseg = min(len(sig), 256)
    noverlap = min(128, nperseg // 2)
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return None, None
    peak_idx = np.argmax(psd[band])
    peak_freq = freqs[band][peak_idx]
    peak_power = psd[band][peak_idx]
    median_power = np.median(psd[band]) + 1e-12
    snr = peak_power / median_power
    bpm = peak_freq * 60.0
    if snr < snr_thresh:
        return None, snr
    return bpm, snr


def estimate_bpm_peaks(sig, fs, hr_min=40, hr_max=180):
    if len(sig) < 10:
        return None
    # simple peak picking on bandpassed POS
    sig = sig - np.mean(sig)
    # dynamic threshold
    thr = np.percentile(sig, 85)
    distance = int(fs * 60.0 / hr_max)
    distance = max(distance, 1)
    peaks, _ = signal.find_peaks(sig, height=thr, distance=distance)
    if len(peaks) < 2:
        return None
    rr = np.diff(peaks) / fs  # seconds between peaks
    if len(rr) == 0:
        return None
    rr = rr[(rr > 60.0 / hr_max) & (rr < 60.0 / hr_min)]
    if len(rr) == 0:
        return None
    bpm = 60.0 / float(np.median(rr))
    if bpm < hr_min or bpm > hr_max:
        return None
    return bpm


def smooth_roi(prev_roi, new_roi, alpha):
    if prev_roi is None:
        return new_roi
    x1p, y1p, x2p, y2p = prev_roi
    x1n, y1n, x2n, y2n = new_roi
    x1 = int(alpha * x1n + (1 - alpha) * x1p)
    y1 = int(alpha * y1n + (1 - alpha) * y1p)
    x2 = int(alpha * x2n + (1 - alpha) * x2p)
    y2 = int(alpha * y2n + (1 - alpha) * y2p)
    return (x1, y1, x2, y2)


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


class AnalysisEngine:
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
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_status = {
            "mood": {"label": "Idle"},
            "hr_bpm": None,
            "snr": None,
            "timestamp": time.time(),
        }
        self.alpha = 0.35
        self.smooth_vals = {"ear": None, "mar": None, "lip": None, "brow": None, "pitch": None}
        self.frame_queue = deque(maxlen=1)
        self.times = deque()
        self.rgb_buffer = deque()
        self.hr_history = deque(maxlen=120)
        self.pos_history = deque(maxlen=600)
        self.last_estimate_time = 0.0
        self.snr_thresh = 6.0
        self.window_sec = 18.0
        self.bandpass_low = 0.7
        self.bandpass_high = 1.9
        self.hr_min, self.hr_max = 40, 180
        self.roi_smooth_alpha = 0.2
        self.smooth_roi_box = None
        self.last_sig_value = 0.0
        self.fps_fallback = 30.0
        self.last_hr_bpm = None
        self.last_hr_ts = 0.0
        self.camera_index = None
        self.last_pos_value = 0.0
        self.last_hr_method = "none"
        self.hr_smooth = deque(maxlen=5)
        self.last_mood_label = "Idle"

    @staticmethod
    def list_cameras(max_index=5):
        cams = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                cams.append({"index": idx, "label": f"Camera {idx}"})
                cap.release()
        return cams or [{"index": 0, "label": "Camera 0"}]

    def start(self, camera_index=0):
        if self.running and self.camera_index == int(camera_index):
            return True
        if self.running and self.camera_index != int(camera_index):
            self.stop()
        self.camera_index = int(camera_index)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        fps_fixed = self.cap.get(cv2.CAP_PROP_FPS)
        if fps_fixed and fps_fixed > 1e-3:
            self.fps_fallback = fps_fixed
        self.running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.smooth_roi_box = None
        self.times.clear()
        self.rgb_buffer.clear()
        self.hr_history.clear()
        self.pos_history.clear()
        self.last_hr_ts = 0.0
        self.last_pos_value = 0.0
        self.last_hr_method = "none"
        self.latest_frame = None
        with self.lock:
            # retain last mood/hr values; just mark not running
            self.latest_status.update({
                "running": False,
                "timestamp": time.time(),
            })

    def _smooth(self, key, new):
        old = self.smooth_vals[key]
        if old is None:
            self.smooth_vals[key] = new
        else:
            self.smooth_vals[key] = self.alpha * new + (1 - self.alpha) * old
        return self.smooth_vals[key]

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (self.width, self.height))
            processed_frame, status = self._process_frame(frame)
            with self.lock:
                self.latest_frame = processed_frame
                self.latest_status = status
            time.sleep(0.01)

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        mood_label = self.last_mood_label
        hr_bpm = None
        snr_val = None
        roi_box = None

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
            self.last_mood_label = mood_label

            xs = [p[0] for p in lm]
            ys = [p[1] for p in lm]
            x_min = max(0, min(xs))
            x_max = min(w - 1, max(xs))
            y_min_all = max(0, min(ys))
            y_max_all = min(h - 1, max(ys))
            width = x_max - x_min
            shrink = int(0.5 * 0.50 * width)
            x1 = x_min + shrink
            x2 = x_max - shrink
            y1 = int(y_min_all + 0.005 * (y_max_all - y_min_all))
            y2 = int(y_min_all + 0.105 * (y_max_all - y_min_all))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                roi_box = (x1, y1, x2, y2)

            if roi_box is not None:
                self.smooth_roi_box = smooth_roi(self.smooth_roi_box, roi_box, self.roi_smooth_alpha)
                x1, y1, x2, y2 = self.smooth_roi_box
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    mean_rgb = np.mean(roi_rgb.reshape(-1, 3), axis=0)
                    t_now = time.time()
                    self.times.append(t_now)
                    self.rgb_buffer.append(mean_rgb)
                    while self.times and (self.times[-1] - self.times[0]) > self.window_sec:
                        self.times.popleft()
                        self.rgb_buffer.popleft()

                    if len(self.rgb_buffer) >= 100 and (t_now - self.last_estimate_time) > 0.5:
                        self.last_estimate_time = t_now
                        fs = self.fps_fallback
                        sig_pos = pos_signal(np.array(self.rgb_buffer))
                        sig_filt = bandpass(sig_pos, fs, self.bandpass_low, self.bandpass_high)
                        if sig_filt is not None:
                            bpm_peak = estimate_bpm_peaks(sig_filt, fs, self.hr_min, self.hr_max)
                            bpm_welch, snr_val = estimate_bpm_welch_with_snr(sig_filt, fs, self.bandpass_low, self.bandpass_high, self.snr_thresh)
                            if bpm_peak:
                                hr_bpm = bpm_peak
                                self.last_hr_method = "peaks"
                            elif bpm_welch:
                                hr_bpm = bpm_welch
                                self.last_hr_method = "welch"
                            if hr_bpm and self.hr_min <= hr_bpm <= self.hr_max:
                                self.hr_history.append(hr_bpm)
                                self.hr_smooth.append(hr_bpm)
                                smoothed = sum(self.hr_smooth) / len(self.hr_smooth)
                                hr_bpm = smoothed
                                self.last_hr_bpm = hr_bpm
                                self.last_hr_ts = t_now
                            # keep short signal tail for plotting
                            take = min(len(sig_filt), 200)
                            if take > 0:
                                self.pos_history.extend(sig_filt[-take:])
                            if len(sig_filt) > 0:
                                self.last_sig_value = float(sig_filt[-1])
                                self.last_pos_value = self.last_sig_value

        else:
            self.smooth_roi_box = None

        overlay = frame.copy()
        if self.smooth_roi_box is not None:
            x1, y1, x2, y2 = self.smooth_roi_box
            # alpha scales with POS intensity magnitude
            intensity = min(1.5, abs(self.last_sig_value))
            alpha = 0.08 + 0.35 * intensity
            alpha = max(0.05, min(0.5, alpha))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=-1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), thickness=2)
        else:
            self.last_sig_value = 0.0

        status = {
            "running": self.running,
            "mood": {
                "label": mood_label,
            },
            "hr_bpm": hr_bpm if hr_bpm is not None else (self.last_hr_bpm if self.last_hr_bpm is not None else None),
            "snr": snr_val,
            "hr_mean": statistics.mean(self.hr_history) if self.hr_history else None,
            "hr_median": statistics.median(self.hr_history) if self.hr_history else None,
            "hr_mode": (statistics.mode(self.hr_history) if len(set(self.hr_history)) == 1 else None) if self.hr_history else None,
            "last_pos": self.last_pos_value,
            "hr_method": self.last_hr_method,
            "timestamp": time.time(),
        }

        hr_display = hr_bpm if hr_bpm is not None else self.last_hr_bpm
        if hr_display is not None:
            cv2.putText(frame, f"HR: {hr_display:.1f} bpm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "HR: --", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(frame, f"Mood: {self.last_mood_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return frame, status

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
        with self.lock:
            return dict(self.latest_status)

    def get_pos_history(self):
        with self.lock:
            return list(self.pos_history)
