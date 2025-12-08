#!/usr/bin/env python3
"""
Live POS rPPG (webcam) with ROI smoothing and SNR gate.

- Single forehead ROI (MediaPipe face mesh).
- ROI smoothing (exponential) to reduce box jitter.
- Welch PSD with SNR gate: accept BPM only if the peak is dominant.
- Bandpass 0.8–2.0 Hz, 15 s window.

Dependencies:
  pip install opencv-python numpy scipy mediapipe==0.10.21 matplotlib
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Parameters
BANDPASS_LOW = 0.8
BANDPASS_HIGH = 2.0
WINDOW_SEC = 16.0
MIN_SAMPLES = 100
HR_MIN, HR_MAX = 40, 180
PLOT_SIGNAL_LEN = 600
PLOT_HR_LEN = 300
TEXT_POS = (10, 30)

# ROI tuning
ROI_X_SHRINK = 0.50
ROI_Y_TOP = 0.005
ROI_Y_BOT = 0.105

# ROI smoothing
ROI_SMOOTH_ALPHA = 0.2  # 0=no smoothing, 1=no smoothing; smaller = more smoothing

# SNR gate
SNR_THRESH = 6.0  # peak_psd / median_band_psd must exceed this to accept BPM

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
        return None, snr  # reject if SNR too low
    return bpm, snr

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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Fixed FPS from camera (fallback 30)
    fps_fixed = cap.get(cv2.CAP_PROP_FPS)
    if fps_fixed is None or fps_fixed <= 1e-3:
        fps_fixed = 30.0

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    times = deque()
    rgb_buffer = deque()
    hr_history = deque(maxlen=PLOT_HR_LEN)
    sig_history = deque(maxlen=PLOT_SIGNAL_LEN)
    bpm_display = None

    # ROI smoothing state
    smooth_roi_box = None

    # Matplotlib setup
    plt.ion()
    fig, (ax_sig, ax_hr) = plt.subplots(2, 1, figsize=(8, 6))
    line_sig, = ax_sig.plot([], [], lw=1)
    line_hr, = ax_hr.plot([], [], lw=1, marker='o', ms=3)
    ax_sig.set_title("Filtered POS signal")
    ax_hr.set_title("HR (bpm)")
    ax_hr.set_ylim(40, 180)
    ax_hr.set_ylabel("bpm")
    ax_hr.grid(True)
    ax_sig.grid(True)

    last_estimate_time = 0.0
    ESTIMATE_EVERY = 0.5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_now = time.time()

            roi_box = None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)
            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                xs = [lm.x for lm in lms]
                ys = [lm.y for lm in lms]
                h, w, _ = frame.shape
                x_min = max(0, int(min(xs) * w))
                x_max = min(w - 1, int(max(xs) * w))
                y_min_all = max(0, int(min(ys) * h))
                y_max_all = min(h - 1, int(max(ys) * h))

                width = x_max - x_min
                shrink = int(0.5 * ROI_X_SHRINK * width)
                x1 = x_min + shrink
                x2 = x_max - shrink
                y1 = int(y_min_all + ROI_Y_TOP * (y_max_all - y_min_all))
                y2 = int(y_min_all + ROI_Y_BOT * (y_max_all - y_min_all))

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    roi_box = (x1, y1, x2, y2)

            if roi_box is not None:
                # Smooth ROI
                smooth_roi_box = smooth_roi(smooth_roi_box, roi_box, ROI_SMOOTH_ALPHA)
                x1, y1, x2, y2 = smooth_roi_box
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    mean_rgb = np.mean(rgb.reshape(-1, 3), axis=0)
                    times.append(t_now)
                    rgb_buffer.append(mean_rgb)
                    while times and (times[-1] - times[0]) > WINDOW_SEC:
                        times.popleft()
                        rgb_buffer.popleft()

                    if len(rgb_buffer) >= MIN_SAMPLES and (t_now - last_estimate_time) > ESTIMATE_EVERY:
                        last_estimate_time = t_now
                        fs = fps_fixed
                        sig_pos = pos_signal(np.array(rgb_buffer))
                        sig_filt = bandpass(sig_pos, fs, BANDPASS_LOW, BANDPASS_HIGH)
                        if sig_filt is not None:
                            bpm, snr = estimate_bpm_welch_with_snr(sig_filt, fs, BANDPASS_LOW, BANDPASS_HIGH, SNR_THRESH)
                            if bpm and HR_MIN <= bpm <= HR_MAX:
                                bpm_display = bpm
                                hr_history.append(bpm)
                            take = min(len(sig_filt), PLOT_SIGNAL_LEN)
                            sig_history.extend(sig_filt[-take:])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display BPM and average BPM
            if bpm_display is not None:
                cv2.putText(frame, f"HR: {bpm_display:.1f} bpm",
                            TEXT_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "HR: --", TEXT_POS,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if len(hr_history) >= 3:
                avg_hr = float(np.mean(hr_history))
                cv2.putText(frame, f"Avg: {avg_hr:.1f} bpm",
                            (TEXT_POS[0], TEXT_POS[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            cv2.imshow("POS rPPG (forehead ROI, live)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update plots
            if len(sig_history) > 2:
                line_sig.set_data(np.arange(len(sig_history)), list(sig_history))
                ax_sig.relim(); ax_sig.autoscale_view()
            if len(hr_history) > 0:
                line_hr.set_data(np.arange(len(hr_history)), list(hr_history))
                ax_hr.relim(); ax_hr.autoscale_view()
                ax_hr.set_ylim(40, 180)
            plt.pause(0.001)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()