# Lumentone

Mood-driven music web app that reads facial cues, estimates mood/valence/arousal, and builds an iTunes-powered playlist with embedded previews.

## Setup
- Install Python 3.10+.
- Create a virtualenv and install deps: `python -m venv .venv` then `./.venv/Scripts/activate` and `pip install -r requirements.txt`.
- Ensure a webcam is available for face mesh + rPPG heart-rate estimation.

## Run
- Start the server: `python server.py` (defaults to http://localhost:5000).
- Allow camera access in the browser, then click "Start Capture" and generate a playlist.

## Notes
- Tracks are fetched from iTunes Search; no API keys required.
- Camera dropdown queries available devices; rPPG heart rate is derived from the forehead ROI.
- Spotify env vars in `data.py` are unused in the current iTunes flow.
