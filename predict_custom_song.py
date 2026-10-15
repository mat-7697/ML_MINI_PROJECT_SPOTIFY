r"""Predict the most likely playlists for one manually specified song.

Usage from the project root (Windows PowerShell):
    .\.venv\Scripts\python.exe .\predict_custom_song.py

Prerequisite: run the combined notebook through its final export cell once so
artifacts/champion_histgradientboosting.joblib exists.
"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH_CANDIDATES = [ROOT / "data" / "spotify_songs.csv", ROOT / "spotify_songs.csv"]
DATA_PATH = next((p for p in DATA_PATH_CANDIDATES if p.exists()), None)
MODEL_PATH = ROOT / "artifacts" / "champion_histgradientboosting.joblib"

# Edit these values to try a different song.
# Values are illustrative; replace them with real track features when available.
CUSTOM_SONG = {
    "danceability": 0.68,
    "energy": 0.74,
    "key": 2,
    "loudness": -6.2,
    "mode": 1,
    "speechiness": 0.04,
    "acousticness": 0.21,
    "instrumentalness": 0.03,
    "liveness": 0.17,
    "valence": 0.56,
    "tempo": 112.4,
    "duration_ms": 198000,
    "track_popularity": 0,
    "release_year": 2025,
}

SEED = 42
N_PLAYLISTS = 13
MIN_TRACKS = 100
EXTENDED_NUMERIC = [
    "danceability", "energy", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "loudness",
    "log_duration_ms", "track_popularity", "release_year",
    "release_decade", "is_recent_2019", "acoustic_energy_diff",
    "energy_loudness_ratio", "valence_energy_prod", "dance_tempo_ratio",
]
FEATURE_COLS = EXTENDED_NUMERIC + ["key"]


def make_feature_row(song: dict) -> pd.DataFrame:
    """Create the same raw feature columns and engineered values used in training."""
    required = [
        "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo",
        "duration_ms", "track_popularity", "release_year",
    ]
    missing = [k for k in required if k not in song]
    if missing:
        raise ValueError(f"Missing input values: {', '.join(missing)}")

    for col in [
        "danceability", "energy", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence",
    ]:
        if not 0.0 <= float(song[col]) <= 1.0:
            raise ValueError(f"{col} must be between 0 and 1.")
    if int(song["key"]) not in range(-1, 12):
        raise ValueError("key must be -1 (unknown) or an integer from 0 to 11.")
    if int(song["mode"]) not in (0, 1):
        raise ValueError("mode must be 0 or 1.")
    if float(song["tempo"]) <= 0 or float(song["duration_ms"]) <= 0:
        raise ValueError("tempo and duration_ms must be greater than zero.")
    if not 0 <= float(song["track_popularity"]) <= 100:
        raise ValueError("track_popularity must be between 0 and 100.")

    year = float(song["release_year"])
    loudness = float(song["loudness"])
    energy = float(song["energy"])
    danceability = float(song["danceability"])
    acousticness = float(song["acousticness"])
    valence = float(song["valence"])
    tempo = float(song["tempo"])

    row = dict(song)
    row.update({
        "log_duration_ms": float(np.log1p(float(song["duration_ms"]))),
        "release_year": year,
        "release_decade": float((year // 10) * 10),
        "is_recent_2019": float(year >= 2019),
        "acoustic_energy_diff": acousticness - energy,
        "energy_loudness_ratio": energy / (abs(loudness) + 1e-4),
        "valence_energy_prod": valence * energy,
        "dance_tempo_ratio": danceability / (tempo + 1e-4),
    })
    return pd.DataFrame([row])[FEATURE_COLS]


def reconstruct_label_mapping(data_path: Path):
    """Reproduce notebook playlist selection so numeric model classes map to names."""
    raw = pd.read_csv(data_path)
    counts = raw.groupby("playlist_id")["track_id"].count()
    eligible = np.sort(counts[counts >= MIN_TRACKS].index.values)
    selected_ids = np.random.default_rng(SEED).choice(
        eligible, size=N_PLAYLISTS, replace=False
    )

    cleaned = raw.dropna(subset=["track_name", "track_artist"]).copy()
    cleaned = cleaned[cleaned["playlist_id"].isin(selected_ids)].copy()
    counts_by_track = cleaned.groupby("track_id")["playlist_id"].nunique()
    ambiguous = counts_by_track[counts_by_track > 1].index
    cleaned = cleaned[~cleaned["track_id"].isin(ambiguous)].copy()
    cleaned = cleaned.drop_duplicates(subset=["track_id"]).copy()

    if cleaned["playlist_id"].nunique() != N_PLAYLISTS:
        raise RuntimeError(
            "Could not reconstruct the notebook's 13 playlist classes. "
            "Check that data/spotify_songs.csv is the same CSV used to train the model."
        )

    encoder = LabelEncoder()
    encoder.fit(cleaned["playlist_id"])
    id_to_name = cleaned.groupby("playlist_id")["playlist_name"].first().to_dict()
    return encoder, id_to_name


def main():
    if DATA_PATH is None:
        print("ERROR: Could not find spotify_songs.csv.")
        print("Expected data/spotify_songs.csv or spotify_songs.csv beside this script.")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print("ERROR: Trained model artifact not found:", MODEL_PATH)
        print("Open the combined notebook and run all cells through the final export cell first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    encoder, id_to_name = reconstruct_label_mapping(DATA_PATH)
    X_custom = make_feature_row(CUSTOM_SONG)

    probabilities = model.predict_proba(X_custom)[0]
    predicted_class_numbers = model.classes_
    order = np.argsort(probabilities)[::-1][:3]

    print("\nCUSTOM SONG INPUT")
    for k, v in CUSTOM_SONG.items():
        print(f"  {k}: {v}")

    print("\nTOP 3 PREDICTED PLAYLISTS")
    for rank, i in enumerate(order, start=1):
        encoded_class = int(predicted_class_numbers[i])
        playlist_id = encoder.inverse_transform([encoded_class])[0]
        playlist_name = id_to_name.get(playlist_id, str(playlist_id))
        print(f"{rank}. {playlist_name} — {probabilities[i]:.2%} model probability")

    best_index = int(order[0])
    best_class = int(predicted_class_numbers[best_index])
    best_playlist_id = encoder.inverse_transform([best_class])[0]
    print("\nPredicted playlist:", id_to_name.get(best_playlist_id, str(best_playlist_id)))
    print("Note: probabilities are model scores, not guaranteed measures of user preference.")


if __name__ == "__main__":
    main()
