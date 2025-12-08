import os
import time
import urllib.parse
import random
from typing import Dict, List

import requests


def valence_arousal(mood_label, heart_rate):
    hr = heart_rate or 75
    if hr >= 105:
        arousal = "high"
    elif hr >= 85:
        arousal = "medium"
    else:
        arousal = "low"

    positive = {"Happy"}
    negative = {"Sad", "Depressed", "Angry"}
    ambiguous = {"Neutral", "Stressed / Anxious"}
    if mood_label in positive:
        valence = "positive"
    elif mood_label in negative:
        valence = "negative"
    else:
        valence = "neutral"

    return valence, arousal


# Curated Spotify-friendly seed genres per valence/arousal (all valid per /available-genre-seeds)
MOOD_GENRES: Dict = {
    ("positive", "high"): ["dance", "edm", "pop", "party", "latin"],
    ("positive", "medium"): ["indie-pop", "funk", "disco", "soul", "r-n-b"],
    ("positive", "low"): ["acoustic", "folk", "chill", "singer-songwriter", "ambient"],
    ("neutral", "high"): ["electronic", "synth-pop", "house", "alternative", "hip-hop"],
    ("neutral", "medium"): ["indie", "rock", "indie-pop", "r-n-b", "pop"],
    ("neutral", "low"): ["jazz", "chill", "ambient", "piano", "sleep", "lo-fi"],
    ("negative", "high"): ["metal", "hard-rock", "punk", "heavy-metal", "industrial"],
    ("negative", "medium"): ["trip-hop", "emo", "grunge", "goth", "alternative"],
    ("negative", "low"): ["sad", "sleep", "piano", "acoustic", "ambient", "lo-fi"],
}


# # Example tracks (ids) per genre for documentation/future seeding; not all used in API call
# GENRE_SAMPLE_TRACKS: Dict[str, List[str]] = {
#     "pop": [
#         "0VjIjW4GlUZAMYd2vXMi3b",  # Blinding Lights
#         "463CkQjx2Zk1yXoBuierM9",  # Levitating
#         "7qiZfU4dY1lWllzX7mPBI3",  # Shape of You
#         "1dGr1c8CrMLDpV6mPbImSI",  # As It Was
#         "6UelLqGlWMcVH1E5c4H7lY",  # Watermelon Sugar
#     ],
#     "dance": [
#         "4jPy3l0RUwlUI9T5X2r0XH",  # Head & Heart
#         "7dSJdRtipx6xbD6VucczKQ",  # Rain On Me
#         "2wrJq5XKLnmhRXHiaAXg3G",  # Don't Start Now
#         "5KawlOMHjWeUjQtnuRs22c",  # One Kiss
#         "7EfI2Numy2I8aGJ3nTt9lp",  # Cold Heart
#     ],
#     "edm": [
#         "7JJmb5XwzOO8jgpou264Ml",  # Clarity
#         "0ct6r3EGTcMLPtrXHDvVjc",  # The Nights
#         "12b9Vi25DANhKp4jQSQgz6",  # Levels
#         "0CJIVlq17pIFv4IG9G10wl",  # Don't You Worry Child
#         "1HNkqx9Ahdgi1Ixy2xkKkL",  # Wake Me Up
#     ],
#     "latin": [
#         "0F7FA14euOIX8K0f7nkor6",  # Tusa
#         "2wVDWtLKXunswWecARNILj",  # Con Altura
#         "3AzjcOeAmA57TIOr9zF1ZW",  # Despacito
#         "2rPE9A1vEgShuZxxzR2tZH",  # Mi Gente
#         "4sPmO7WMQUAf45kwMOtONw",  # Hawái
#     ],
#     "indie-pop": [
#         "2XW4DbS6NddZxRP2Jd0r7d",  # Ribs
#         "0a4ZhgDltJr6ODc2n7R3Cy",  # Space Song
#         "1o22pU4jbDbpjs8vQR0f63",  # Alaska
#         "5hc71nKsUgtwQ3z52KEKQk",  # Somebody Else
#         "0r7CVbZTWZgbTCYdfa2P31",  # Electric Love
#     ],
#     "funk": [
#         "5mjTr8yl8lIBdn8fKMjsyO",  # Uptown Funk
#         "7dS5EaCoMnN7DzlpT6aRn2",  # Get Lucky
#         "2DwUdMJ5uxdS0L5t3zDhNe",  # Treasure
#         "0jXXNGljqupsJaZsgSbMZV",  # Take On Me (funky)
#         "4pbJqGIASGPr0ZpGpnWkDn",  # Feel It Still
#     ],
#     "disco": [
#         "0qOnSQQF0yzuPWsXrQ9paz",  # Dancing Queen
#         "5uEYRdEIh9Bo4fpjDd4Na9",  # Le Freak
#         "1qfiD1RIvxgJdQkqdh1zlC",  # Stayin' Alive
#         "1fF8iSDuKuy8WGqwomPlkB",  # I Will Survive
#         "2aTHAfMZ1cK0tBAJmS0HVb",  # September
#     ],
#     "soul": [
#         "1JSTJqkT5qHq8MDJnJbRE1",  # Valerie
#         "4U45aEWtQhrm8A5mxPaFZ7",  # Rehab
#         "4RVwu0g32PAqgUiJoXsdF8",  # Redbone
#         "1P17dC1amhFzptugyAO7Il",  # Leave The Door Open
#         "3zBhihYUHBmGd2bcQIobrF",  # Lovely Day
#     ],
#     "r-n-b": [
#         "5aAx2yezTd8zXrkmtKl66Z",  # Blame Game
#         "3Qm86XLflmIXVm1wcwkgDK",  # Crew
#         "1IICgXArEULZ42ZHvbspeK",  # Thinkin Bout You
#         "6habFhsOp2NvshLv26DqMb",  # Love On Top
#         "5yXQ8wvFqZ4v2fU8X0f2CF",  # No Guidance
#     ],
#     "acoustic": [
#         "1diX6i4LgUKR9qMRrAeGLi",  # Tenerife Sea
#         "1Pw5C4N6Bn5xwBF0Zzq2tn",  # Skinny Love
#         "0d2iYfpKoM0QCKvcLCkBao",  # Happier Than Ever
#         "2LAl8O6nDqYvrZ6V3yPyjC",  # Cherry Wine
#         "3XVBdLihbNbxUwZosxcGuJ",  # Heartbeats
#     ],
#     "folk": [
#         "6Vc5wAMmXdKIAM7WUoEb7N",  # Ho Hey
#         "0lTjl8pUQxSJKYyZ6N7Y0E",  # Holocene
#         "5aqjYtvhqWb2Fefj24C0EA",  # Ophelia
#         "3KkXRkHbMCARz0aVfEt68P",  # All Too Well
#         "2PpruBYCo4H7WOBJ7Q2EwM",  # Hey Ya! (folk pop)
#     ],
#     "chill": [
#         "0ggxFFfcIEUfX30uQbH64q",  # sunsets
#         "2X485T9Z5Ly0xyaghN73ed",  # Borderline
#         "2Fxmhks0bxGSBdJ92vM42m",  # Bad Guy (chillish)
#         "5CtI0qwDJkDQGwXD1H1cLb",  # Slow Dancing in the Dark
#         "0RlUqIlh5gV9lFeoW0NJqk",  # Coffee
#     ],
#     "ambient": [
#         "5Hfbag0Ss7VECYvE5Yh3dG",
#         "0OZ0B70JOfUozfvZL9Gvkx",
#         "7ky9TDb4djOetd9i6b5e9b",
#         "0R7jJuuD2yMQBQX6HJ9KMe",
#         "1yJ5WZatSd2Z1sr1MSQFct",
#     ],
#     "electronic": [
#         "2X485T9Z5Ly0xyaghN73ed",
#         "0tBbt8CrmxbjRP0pueQkyU",
#         "3sNVsP50132BTNlImLx70i",
#         "1BxfuPKGuaTgP7aM0Bbdwr",
#         "0e7ipj03S05BNilyu5bRzt",
#     ],
#     "house": [
#         "4h8VwCb1MTGoLKueQ1WgbD",
#         "3E7dfMvvCLUddWissuqMwr",
#         "5KawlOMHjWeUjQtnuRs22c",
#         "5wbJc4tFAYu84P4F0j35tJ",
#         "0pqnGHJpmpxLKifKRmU6WP",
#     ],
#     "hip-hop": [
#         "3AEZUABDXNtecAOSC1qTfo",  # Sicko Mode
#         "4Oun2ylbjFKMPTiaSbbCih",  # HUMBLE.
#         "2b8fOow8UzyDFAE27YhOZM",  # God's Plan
#         "3t6A2SxvqJc0T42V1Q1VZK",  # Lose Yourself
#         "5Z8EDau8uNcP1E8JvmfkZe",  # DNA.
#     ],
#     "indie": [
#         "4M6hHq0hJ9fXbJWnKtM9F2",  # Seventeen Going Under
#         "5hc71nKsUgtwQ3z52KEKQk",  # Somebody Else
#         "0a4ZhgDltJr6ODc2n7R3Cy",  # Space Song
#         "2XW4DbS6NddZxRP2Jd0r7d",  # Ribs
#         "0TnOYISbd1XYRBk9myaseg",  # Mr. Brightside
#     ],
#     "rock": [
#         "0tBbt8CrmxbjRP0pueQkyU",  # Seven Nation Army
#         "7oK9VyNzrYvRFo7nQEYkWN",  # Mr Brightside (rock pop)
#         "3ZOEytgrvLwQaqXreDs2Jx",  # Hotel California
#         "5ghIJDpPoe3CfHMGu71E6T",  # Sweet Child O' Mine
#         "5W3cjX2J3tjhG8zb6u0qHn",  # Smells Like Teen Spirit
#     ],
#     "jazz": [
#         "3E7fzvH1X8rK2hxGqg5F1p",  # Take Five
#         "6EJKtMcq7qDdyRplhl3iIW",  # So What
#         "0lI3oFfY0e6MN2NfmWgW8x",  # Blue in Green
#         "5cXM2Q3VcLAr72m6K8Qzh1",  # Feeling Good
#         "4eWQlBRaTjPPUlzacqEeoQ",  # Fly Me To The Moon
#     ],
#     "metal": [
#         "4l4u9e9D6hWq1is5Tne19U",  # Enter Sandman
#         "2S5hlvw4CMtMGswCcXNIcI",  # The Trooper
#         "6EPRKhUOdiFSQwGBRBbvsZ",  # Chop Suey!
#         "3BGQ0PZwt3kC9zpKj32cYk",  # Du Hast
#         "2d6gLIPdCIRcQ1BZ0PQ2cF",  # Painkiller
#     ],
#     "hard-rock": [
#         "6jA8HL9i4QZzxWTnYzJ9R4",  # Back In Black
#         "4pbJqGIASGPr0ZpGpnWkDn",  # Feel It Still (rockish)
#         "6D9urNKpua3YWxk9l8Z8mz",  # Highway to Hell
#         "7xYcJgKrVsJ8dYYGsgym10",  # Thunderstruck
#         "5CQ30WqJwcep0pYcV4AMNc",  # Bohemian Rhapsody
#     ],
#     "punk": [
#         "1oYiJtgFAxL2oiwrqGxD2P",  # American Idiot
#         "0mOn6sEBUa3KqZ3B0XNRPR",  # Basket Case
#         "5CnpZV3q5BcESefcB3WJmz",  # I Write Sins Not Tragedies
#         "5ghIJDpPoe3CfHMGu71E6T",  # Sweet Child O' Mine (rock/punk vibes)
#         "2bP9AIf1LxPXN2Cgg1v26q",  # Fat Lip
#     ],
#     "trip-hop": [
#         "6QPKYGnAW9QozVz2d1k6V9",  # Teardrop
#         "2lH4z3bYE5iymAJ7aS3XCE",  # Glory Box
#         "0Puj4YlTm6xNzDDADXHMI9",  # All I Need
#         "1w5Kfo2jwwIPruYS2UWh56",  # Porcelain
#         "6Xf8xFbFMmKIyJ28bWQ39m",  # Angel
#     ],
#     "emo": [
#         "3K4HG9evC7dg3N0R9cYqk4",  # Welcome to the Black Parade
#         "2ziB7fzrXBoh1HUPS6sVFn",  # I Miss You
#         "3Z3Wjj8q8uyFvXYO2Yq7GS",  # Helena
#         "0r0dir9c7HkKzGbpF1DEbN",  # The Middle
#         "3Fd9W7x1zY1f46dP5E4mYU",  # Thnks fr th Mmrs
#     ],
#     "sad": [
#         "4TnjEaWOeW0eKTKIEvJyCa",  # Drivers License
#         "6PERP62TejQjgHu81OHxgM",  # Someone Like You
#         "3XVBdLihbNbxUwZosxcGuJ",  # Heartbeats
#         "4RvWPyQ5RL0ao9LPZeSouE",  # lovely
#         "5KawlOMHjWeUjQtnuRs22c",  # One Kiss (less sad but mellow)
#     ],
#     "rain": [
#         "0R7jJuuD2yMQBQX6HJ9KMe",
#         "0lI3oFfY0e6MN2NfmWgW8x",
#         "5Y9ljw4nwfGqfOsnQyaiqG",
#         "6dGnYIeXmHdcikdzNNDMm2",
#         "1o4eM4FZW0YJYcs3cD5t9J",
#     ],
# }


def genres_for_state(valence, arousal):
    return MOOD_GENRES.get((valence, arousal), ["indie", "pop", "electronic"])


# _token_cache = {"token": None, "exp": 0}  # retained for compatibility; not used in iTunes flow


# def _get_spotify_credentials():
#     cid = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIFY-CLIENT-ID")
#     secret = os.getenv("SPOTIFY_CLIENT_SECRET") or os.getenv("SPOTIFY-CLIENT-SECRET")
#     return cid, secret


# def _get_spotify_token():
#     now = time.time()
#     if _token_cache["token"] and now < _token_cache["exp"] - 30:
#         return _token_cache["token"]

#     client_id, client_secret = _get_spotify_credentials()
#     if not client_id or not client_secret:
#         raise RuntimeError("Spotify credentials missing. Set SPOTIFY_CLIENT_ID/SECRET env vars.")

#     resp = requests.post(
#         "https://accounts.spotify.com/api/token",
#         data={"grant_type": "client_credentials"},
#         auth=(client_id, client_secret),
#         timeout=10,
#     )
#     print("[spotify-token] status", resp.status_code)
#     print("[spotify-token] body", resp.text[:300])
#     resp.raise_for_status()
#     payload = resp.json()
#     access_token = payload.get("access_token")
#     expires_in = payload.get("expires_in", 3600)
#     _token_cache["token"] = access_token
#     _token_cache["exp"] = now + expires_in
#     return access_token



# def _build_audio_targets(valence: str, arousal: str):
#     valence_target = {"positive": 0.8, "neutral": 0.55, "negative": 0.25}.get(valence, 0.5)
#     energy_target = {"high": 0.8, "medium": 0.6, "low": 0.35}.get(arousal, 0.5)
#     return valence_target, energy_target


def fetch_tracks(genres, size=5, valence="neutral", arousal="medium"):
    tracks = []
    # Limit per-genre fetch to distribute roughly evenly, but we'll shuffle later
    per_genre = max(2, size // max(1, len(genres)))
    for genre in genres:
        try:
            params = {
                "term": genre,
                "entity": "musicTrack",
                "limit": per_genre,
                "country": "US",
            }
            resp = requests.get("https://itunes.apple.com/search", params=params, timeout=8)
            print("[itunes] status", resp.status_code, "genre", genre, "url", resp.url)
            resp.raise_for_status()
            data = resp.json().get("results", [])
            for item in data:
                title = item.get("trackName") or item.get("collectionName")
                artist = item.get("artistName", "")
                if not title:
                    continue
                query = urllib.parse.quote_plus(f"{title} {artist} {genre}")
                tracks.append({
                    "title": title,
                    "artist": artist,
                    "genre": genre,
                    "preview": item.get("previewUrl"),
                    "source": item.get("trackViewUrl") or item.get("collectionViewUrl"),
                    "spotify": f"https://open.spotify.com/search/{query}",
                    "youtube": f"https://www.youtube.com/results?search_query={query}",
                })
        except Exception as exc:
            print("[itunes-error]", genre, exc)
            continue
    random.shuffle(tracks)
    if len(tracks) > size:
        return tracks[:size]
    return tracks
