import time
import io

from flask import Flask, jsonify, render_template, request, Response

from analysis import AnalysisEngine
from data import fetch_tracks, genres_for_state, valence_arousal
from matplotlib.figure import Figure

app = Flask(__name__)
engine = AnalysisEngine()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_capture', methods=['POST'])
def start_capture():
    payload = request.get_json(silent=True) or {}
    camera_index = payload.get("camera_index", 0)
    ok = engine.start(camera_index=camera_index)
    if not ok:
        return jsonify({"ok": False, "message": "Could not open camera"}), 500
    return jsonify({"ok": True})


@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    engine.stop()
    return jsonify({"ok": True})


@app.route('/cameras')
def cameras():
    cams = AnalysisEngine.list_cameras()
    return jsonify({"ok": True, "cameras": cams})


@app.route('/video_feed')
def video_feed():
    def gen():
        boundary = b'--frame\r\n'
        while True:
            frame = engine.get_jpeg()
            if frame is None:
                break
            yield boundary
            yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.02)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pos_plot')
def pos_plot():
    data = engine.get_pos_history()
    fig = Figure()
    ax = fig.subplots()
    if data:
        ax.plot(data, color='green', linewidth=1)
    ax.set_title('POS signal (rPPG)')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/status')
def status():
    heart_rate = request.args.get('heart_rate', type=int)
    state = engine.get_status()
    mood_label = state.get("mood", {}).get("label", "Neutral")
    hr_for_valence = state.get("hr_bpm") or heart_rate
    valence, arousal = valence_arousal(mood_label, hr_for_valence)
    genres = genres_for_state(valence, arousal)
    state.update({
        "valence": valence,
        "arousal": arousal,
        "genres": genres,
    })
    return jsonify(state)


@app.route('/playlist', methods=['POST'])
def playlist():
    payload = request.get_json(force=True)
    heart_rate = payload.get('heart_rate')
    measured = engine.get_status().get("hr_bpm")
    hr_for_valence = measured or heart_rate
    mode = payload.get('playlist_mode', 'medium')
    import random
    mode_ranges = {
        "short": (3, 5),
        "medium": (6, 10),
        "long": (8, 20),
    }
    lo, hi = mode_ranges.get(mode, mode_ranges["medium"])
    playlist_size = random.randint(lo, hi)
    mood_label = engine.get_status().get("mood", {}).get("label", "Neutral")
    valence, arousal = valence_arousal(mood_label, hr_for_valence)
    genres = genres_for_state(valence, arousal)
    print("[playlist] mood", mood_label, "valence", valence, "arousal", arousal, "genres", genres, "size", playlist_size)
    try:
        tracks = fetch_tracks(genres, size=playlist_size, valence=valence, arousal=arousal)
    except Exception as exc:  # fail loudly for visibility
        print("[playlist-error]", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({
        "mood": mood_label,
        "valence": valence,
        "arousal": arousal,
        "genres": genres,
        "tracks": tracks,
        "ok": True,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
