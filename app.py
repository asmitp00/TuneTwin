"""Local Flask app for TuneTwin AI's genre-aware Spotify recommender."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import secrets
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_PATH = Path(os.getenv("TUNETWIN_DATA_PATH", ROOT / "data" / "data.csv"))
GENRE_PATH = Path(os.getenv("TUNETWIN_GENRE_PATH", ROOT / "data" / "extracted" / "data" / "data_w_genres.csv"))
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
# The Web Playback SDK validates all three browser-identifying scopes in
# addition to `streaming`; without the email/private pair it returns the
# unhelpful client-side error: "Invalid token scopes."
SCOPES = "user-read-playback-state user-modify-playback-state streaming user-read-currently-playing user-read-recently-played user-read-email user-read-private playlist-read-private playlist-modify-private playlist-modify-public"
PLAYBACK_SCOPES = {"user-read-playback-state", "user-modify-playback-state", "user-read-currently-playing", "streaming", "user-read-email", "user-read-private"}
LINUCB_PATH = ROOT / "models" / "linucb_live.json"
RECOMMENDER_ARTIFACT_PATH = ROOT / "models" / "tunetwin_recommender.joblib"
LINUCB_ALPHA = 0.20
LINUCB_BLEND = 0.25
CLASSIFIER_BLEND = 0.25
MIN_CLASSIFIER_EVENTS = 2
PCA_VARIANCE_TARGET = 0.90
KMEANS_SEED = 42
MAX_TRACKS_PER_ARTIST = 3
MIN_DISTINCT_LEAD_ARTISTS = 3
BASE_FEATURE_CANDIDATES = [
    "valence", "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "liveness", "loudness", "speechiness", "tempo",
]
DERIVED_FEATURE_ORDER = [
    "energy_valence", "dance_energy", "acoustic_instrumental",
    "tempo_deviation", "log_duration",
]

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_urlsafe(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    TEMPLATES_AUTO_RELOAD=True,
    SEND_FILE_MAX_AGE_DEFAULT=0,
)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Werkzeug's default access log includes callback query parameters, which can
# contain one-time OAuth codes. Keep request URLs out of local logs.
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def parse_list(value) -> list[str]:
    """Read list-like CSV cells without assuming their serialization format."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parts = list(value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "[]"}:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            parts = list(parsed)
        elif isinstance(parsed, str):
            parts = [parsed]
        else:
            parts = re.split(r"\s*[;,|]\s*", text.strip("[]"))
    return [str(part).strip().strip("'\"") for part in parts if str(part).strip()]


def normalize_song_title(value: str) -> str:
    """Normalize harmless title differences so catalog copies count as one song."""
    title = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").casefold()
    title = re.sub(r"\s*[\(\[]?\s*(?:feat\.?|ft\.?|featuring)\s+.*?[\)\]]?$", "", title)
    title = re.sub(
        r"\s*[\(\[][^)\]]*(?:remaster(?:ed)?|radio edit|single version|album version|live version|live at|mono version|stereo version)[^)\]]*[\)\]]",
        " ", title,
    )
    title = re.sub(
        r"\s*[-–—]\s*(?:(?:\d{4}\s*)?remaster(?:ed)?(?:\s+\d+)?|radio edit|single version|album version|live version)$",
        " ", title,
    )
    return re.sub(r"[^a-z0-9]+", "", title)


def normalize_artist_name(value: str) -> str:
    artist = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", artist)


def load_recommender_artifact() -> dict | None:
    """Load the notebook-trained preprocessing, PCA/K-Means and classifier."""
    try:
        artifact = joblib.load(RECOMMENDER_ARTIFACT_PATH)
    except FileNotFoundError:
        logging.warning("Recommender artifact not found at %s; using the built-in fallback pipeline.", RECOMMENDER_ARTIFACT_PATH)
        return None
    except Exception as exc:
        logging.warning("Could not load recommender artifact (%s); using the fallback pipeline.", type(exc).__name__)
        return None
    required = {
        "base_features", "feature_order", "feature_min", "feature_max",
        "clip_lower", "clip_upper", "imputer", "scaler", "pca", "kmeans",
        "classifier", "classifier_features",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        logging.warning("Recommender artifact is missing %s; using the fallback pipeline.", ", ".join(missing))
        return None
    return artifact


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the same interaction features used by the tuning notebook."""
    out = frame.copy()
    if {"energy", "valence"}.issubset(out.columns):
        out["energy_valence"] = out["energy"] * out["valence"]
    if {"danceability", "energy"}.issubset(out.columns):
        out["dance_energy"] = out["danceability"] * out["energy"]
    if {"acousticness", "instrumentalness"}.issubset(out.columns):
        out["acoustic_instrumental"] = out["acousticness"] * out["instrumentalness"]
    if "tempo" in out.columns:
        out["tempo_deviation"] = (out["tempo"] - out["tempo"].median()).abs()
    if "duration_ms" in out.columns:
        out["log_duration"] = np.log1p(out["duration_ms"].clip(lower=0))
    return out


def load_catalog() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    tracks = pd.read_csv(DATA_PATH, low_memory=False)
    name_col = next((c for c in ("name", "track_name", "track") if c in tracks.columns), None)
    artist_col = next((c for c in ("artists", "artist", "artist_name") if c in tracks.columns), None)
    if not name_col or not artist_col:
        raise ValueError("The track CSV needs a track name column and an artist column.")

    artifact = RECOMMENDER_ARTIFACT
    if artifact:
        feature_cols = [c for c in artifact["base_features"] if c in tracks.columns]
        missing = sorted(set(artifact["base_features"]).difference(tracks.columns))
        if missing:
            raise ValueError(f"Dataset is missing artifact feature columns: {', '.join(missing)}")
    else:
        feature_cols = [c for c in BASE_FEATURE_CANDIDATES if c in tracks.columns]
    if not feature_cols:
        raise ValueError("The track CSV has none of the supported audio feature columns.")

    tracks = tracks.dropna(subset=[name_col]).drop_duplicates().reset_index(drop=True)
    tracks["_name_col"] = name_col
    tracks["_artist_col"] = artist_col
    tracks["_name"] = tracks[name_col].astype(str).str.strip()
    tracks["_artists"] = tracks[artist_col].map(parse_list)
    tracks["_artist_display"] = tracks["_artists"].map(lambda xs: ", ".join(xs) if xs else "Unknown artist")
    tracks["_artist_keys"] = tracks["_artists"].map(lambda xs: {x.casefold().strip() for x in xs if x.strip()})
    tracks["_artist_identity_keys"] = tracks["_artists"].map(
        lambda xs: {normalize_artist_name(x) for x in xs if x.strip()}
    )
    tracks["_title_key"] = tracks["_name"].map(normalize_song_title)
    tracks["_song_key"] = tracks["_artists"].map(
        lambda artists: normalize_artist_name(artists[0]) if artists else ""
    )

    artist_genres: dict[str, set[str]] = {}
    if GENRE_PATH.exists():
        genre_table = pd.read_csv(GENRE_PATH, low_memory=False)
        g_artist = next((c for c in ("artists", "artist") if c in genre_table.columns), None)
        g_tags = next((c for c in genre_table.columns if "genre" in str(c).casefold()), None)
        if g_artist and g_tags:
            for artist, tags in genre_table[[g_artist, g_tags]].itertuples(index=False, name=None):
                parsed_tags = {x.casefold().strip() for x in parse_list(tags) if x.strip()}
                for name in parse_list(artist):
                    artist_genres.setdefault(name.casefold().strip(), set()).update(parsed_tags)
    tracks["_genres"] = tracks["_artists"].map(
        lambda artists: set().union(*(artist_genres.get(a.casefold().strip(), set()) for a in artists)) if artists else set()
    )

    numeric = tracks[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if artifact:
        lower = pd.Series(artifact["clip_lower"], index=artifact["base_features"])
        upper = pd.Series(artifact["clip_upper"], index=artifact["base_features"])
        numeric = numeric.clip(lower=lower, upper=upper, axis="columns")
        imputed = pd.DataFrame(
            artifact["imputer"].transform(numeric),
            columns=artifact["base_features"],
            index=tracks.index,
        )
    else:
        imputed = numeric.fillna(numeric.median()).fillna(0.0)
        imputed = pd.DataFrame(imputed, columns=feature_cols, index=tracks.index)
    imputed = add_derived_features(imputed)
    if artifact:
        feature_order = list(artifact["feature_order"])
        imputed = imputed.reindex(columns=feature_order)
        scaler = artifact["scaler"]
    else:
        feature_order = list(imputed.columns)
        scaler = StandardScaler()
    matrix = imputed.to_numpy(dtype=float)
    if artifact:
        scaled = scaler.transform(matrix).astype(np.float32)
        feature_min = np.asarray(artifact["feature_min"], dtype=np.float32)
        feature_max = np.asarray(artifact["feature_max"], dtype=np.float32)
    else:
        scaled = scaler.fit_transform(matrix).astype(np.float32)
        feature_min = imputed.min().to_numpy(dtype=np.float32)
        feature_max = imputed.max().to_numpy(dtype=np.float32)
    denominator = np.where(feature_max > feature_min, feature_max - feature_min, 1.0)
    unit = np.clip((matrix - feature_min) / denominator, 0.0, 1.0).astype(np.float32)
    return (
        tracks,
        feature_order,
        normalize(scaled, norm="l2").astype(np.float32),
        scaled,
        unit,
        feature_min,
        feature_max,
    )


def fit_catalog_models(scaled_features: np.ndarray, artifact: dict | None = None) -> tuple[PCA, np.ndarray, KMeans, np.ndarray, NearestNeighbors, dict[int, float]]:
    """Use the notebook's fitted PCA/K-Means, or fit them if no artifact exists."""
    if artifact:
        pca = artifact["pca"]
        # The notebook's PCA/K-Means were fitted on float64 arrays. Keep that
        # dtype through prediction, then store float32 vectors for the index.
        pca_vectors = pca.transform(scaled_features).astype(np.float64)
        kmeans = artifact["kmeans"]
        cluster_labels = kmeans.predict(pca_vectors)
        pca_vectors = pca_vectors.astype(np.float32)
        scores = {int(k): float(v) for k, v in artifact.get("kmeans_silhouettes", {}).items()}
        neighbors = NearestNeighbors(metric="euclidean", algorithm="auto", n_jobs=-1)
        neighbors.fit(pca_vectors)
        return pca, pca_vectors, kmeans, cluster_labels, neighbors, scores

    pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full", random_state=KMEANS_SEED)
    pca_vectors = pca.fit_transform(scaled_features).astype(np.float32)

    row_count = len(pca_vectors)
    if row_count < 3:
        raise ValueError("At least three usable tracks are required for K-Means clustering.")
    rng = np.random.default_rng(KMEANS_SEED)
    sample_indices = rng.choice(row_count, size=min(row_count, 3000), replace=False)
    sample_vectors = pca_vectors[sample_indices]
    scores: dict[int, float] = {}
    for k in range(2, min(10, row_count - 1) + 1):
        candidate = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=10)
        labels = candidate.fit_predict(sample_vectors)
        scores[k] = float(silhouette_score(sample_vectors, labels))
    selected_k = max(scores, key=scores.get)

    # Fit on the same reproducible sample used to compare K, then assign every
    # catalog track to its nearest learned cluster center.
    kmeans = KMeans(n_clusters=selected_k, random_state=KMEANS_SEED, n_init=10)
    kmeans.fit(sample_vectors)
    cluster_labels = kmeans.predict(pca_vectors)

    neighbors = NearestNeighbors(metric="euclidean", algorithm="auto", n_jobs=-1)
    neighbors.fit(pca_vectors)
    return pca, pca_vectors, kmeans, cluster_labels, neighbors, scores


RECOMMENDER_ARTIFACT = load_recommender_artifact()
if RECOMMENDER_ARTIFACT:
    CLASSIFIER_BLEND = float(RECOMMENDER_ARTIFACT.get("classifier_blend", CLASSIFIER_BLEND))
    LINUCB_ALPHA = float(RECOMMENDER_ARTIFACT.get("linucb_alpha", LINUCB_ALPHA))
    LINUCB_BLEND = float(RECOMMENDER_ARTIFACT.get("linucb_blend", LINUCB_BLEND))
(
    TRACKS,
    AUDIO_FEATURES,
    AUDIO_VECTORS,
    STANDARDIZED_FEATURES,
    UNIT_FEATURES,
    FEATURE_MIN,
    FEATURE_MAX,
) = load_catalog()
PCA_MODEL, PCA_VECTORS, KMEANS_MODEL, CLUSTER_LABELS, PCA_KNN, KMEANS_SILHOUETTES = fit_catalog_models(
    STANDARDIZED_FEATURES, RECOMMENDER_ARTIFACT
)
SELECTED_K = KMEANS_MODEL.n_clusters
PCA_CONTEXT = normalize(PCA_VECTORS, norm="l2").astype(np.float32)
CLUSTER_CONTEXT = np.eye(SELECTED_K, dtype=np.float32)[CLUSTER_LABELS]
MODEL_CONTEXTS = normalize(np.hstack([AUDIO_VECTORS, PCA_CONTEXT, CLUSTER_CONTEXT]), norm="l2").astype(np.float32)
NAME_COL = TRACKS["_name_col"].iloc[0]
ARTIST_COL = TRACKS["_artist_col"].iloc[0]
PRIMARY_ARTISTS = TRACKS["_artists"].map(lambda artists: artists[0].casefold().strip() if artists else "")
PRIMARY_ARTIST_KEYS = PRIMARY_ARTISTS.to_numpy()
TITLE_KEYS = TRACKS["_title_key"].to_numpy()
ARTIST_KEYS = TRACKS["_artist_keys"].to_numpy()
ARTIST_IDENTITY_KEYS = TRACKS["_artist_identity_keys"].to_numpy()
SONG_KEYS = list(zip(TRACKS["_title_key"], TRACKS["_song_key"]))
# Mood anchors should be real songs, not hour-long noise loops or interludes.
POPULARITY = TRACKS["popularity"].fillna(0).to_numpy(dtype=np.float32) / 100.0 if "popularity" in TRACKS else np.zeros(len(TRACKS), dtype=np.float32)
_DURATION_MS = TRACKS["duration_ms"].fillna(0).to_numpy(dtype=np.float32) if "duration_ms" in TRACKS else np.zeros(len(TRACKS), dtype=np.float32)
NOISE_GENRES = {
    "white noise", "brown noise", "pink noise", "sleep", "environmental",
    "water", "sound", "background music", "rain",
}
MOOD_ELIGIBLE = (
    (_DURATION_MS >= 60_000)
    & (_DURATION_MS <= 600_000)
    & ~TRACKS["_name"].str.casefold().str.contains("noise|binaural|asmr|loopable", regex=True).to_numpy()
    & ~TRACKS["_genres"].map(lambda genres: bool(genres & NOISE_GENRES)).to_numpy()
)
GENRE_ENCODER = MultiLabelBinarizer(sparse_output=True)
GENRE_MATRIX = GENRE_ENCODER.fit_transform(TRACKS["_genres"]).tocsr()
GENRE_COUNTS = np.asarray(GENRE_MATRIX.sum(axis=1)).ravel()
GENRE_INDEX = {name: idx for idx, name in enumerate(GENRE_ENCODER.classes_)}
ARTIST_ROWS: dict[str, list[int]] = {}
for _row_idx, _artist_keys in enumerate(TRACKS["_artist_keys"]):
    for _artist_key in _artist_keys:
        ARTIST_ROWS.setdefault(_artist_key, []).append(_row_idx)
TITLE_ROWS: dict[str, list[int]] = {}
for _row_idx, _title_key in enumerate(TRACKS["_title_key"]):
    if _title_key:
        TITLE_ROWS.setdefault(_title_key, []).append(_row_idx)

MOOD_PROFILES = {
    "Upbeat": {"valence": 1.0, "energy": 0.8, "danceability": 0.7},
    "Chill": {"acousticness": 0.8, "energy": -0.9, "speechiness": -0.4},
    "Focus": {"instrumentalness": 1.0, "speechiness": -0.8, "energy": -0.2, "acousticness": 0.2},
    "Workout": {"energy": 1.0, "danceability": 0.8, "tempo": 0.4, "valence": 0.3},
}
MOOD_MIX_COPY = {
    "Focus": ("Focus Mix", "Instrumental-leaning tracks for deep work, built from the model's audio-feature profile."),
    "Chill": ("Unwind Mix", "Softer, lower-energy picks for slowing down."),
    "Upbeat": ("Lift Me Up Mix", "Bright, high-valence tracks chosen by the recommender."),
    "Workout": ("Workout Mix", "High-energy, danceable tracks selected for training and movement."),
}


def load_bandit_state() -> dict:
    dimension = MODEL_CONTEXTS.shape[1]
    try:
        state = json.loads(LINUCB_PATH.read_text(encoding="utf-8"))
        matrix = np.asarray(state["A"], dtype=float)
        vector = np.asarray(state["b"], dtype=float)
        if matrix.shape == (dimension, dimension) and vector.shape == (dimension,):
            return {"A": matrix, "b": vector, "updates": int(state.get("updates", 0)), "events": state.get("events", [])}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    return {"A": np.eye(dimension), "b": np.zeros(dimension), "updates": 0, "events": []}


def save_bandit_state(state: dict) -> None:
    LINUCB_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "A": np.asarray(state["A"], dtype=float).tolist(),
        "b": np.asarray(state["b"], dtype=float).tolist(),
        "updates": int(state["updates"]),
        "events": state.get("events", [])[-50:],
    }
    temporary = LINUCB_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(serializable, separators=(",", ":")), encoding="utf-8")
    temporary.replace(LINUCB_PATH)


def update_bandit(track_idx: int, feedback: str) -> tuple[int, float]:
    """Apply one explicit like/skip signal to a shared linear contextual bandit."""
    rewards = {"like": 2.0, "skip": -1.0}
    if feedback not in rewards:
        raise ValueError("Feedback must be like or skip.")
    context = MODEL_CONTEXTS[track_idx].astype(float)
    reward = rewards[feedback]
    state = load_bandit_state()
    state["A"] += np.outer(context, context)
    state["b"] += reward * context
    state["updates"] += 1
    state["events"].append({
        "track_idx": int(track_idx),
        "track": TRACKS.iloc[track_idx]["_name"],
        "artist": TRACKS.iloc[track_idx]["_artist_display"],
        "feedback": feedback,
        "reward": reward,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    save_bandit_state(state)
    return state["updates"], reward


def valid_feedback_events(state: dict | None = None) -> list[dict]:
    """Feedback rows that still point at a usable catalog track."""
    state = state or load_bandit_state()
    return [
        event for event in state.get("events", [])
        if isinstance(event.get("track_idx"), int)
        and 0 <= event["track_idx"] < len(TRACKS)
        and event.get("feedback") in {"like", "skip"}
    ]


def feedback_preference_profile(state: dict | None = None) -> np.ndarray | None:
    """Estimate a 0..1 listener profile from real Like/Skip feedback."""
    events = valid_feedback_events(state)
    if not events:
        return None
    liked_rows = [event["track_idx"] for event in events if event["feedback"] == "like"]
    skipped_rows = [event["track_idx"] for event in events if event["feedback"] == "skip"]
    median = np.median(UNIT_FEATURES, axis=0)
    liked_mean = UNIT_FEATURES[liked_rows].mean(axis=0) if liked_rows else median
    skipped_mean = UNIT_FEATURES[skipped_rows].mean(axis=0) if skipped_rows else median
    # Start from liked tracks, then move away from skipped tracks relative to
    # the catalog median. The result uses the same 0..1 space as the notebook.
    profile = liked_mean - 0.5 * (skipped_mean - median)
    return np.clip(profile, 0.0, 1.0).astype(np.float32)


def classifier_probabilities(candidate_pool: np.ndarray, state: dict | None = None) -> np.ndarray | None:
    """Predict Like probability with the tuned notebook classifier.

    The classifier is only enabled after real feedback contains both classes.
    Cold start stays content-based and lets LinUCB explore.
    """
    artifact = RECOMMENDER_ARTIFACT
    if not artifact:
        return None
    events = valid_feedback_events(state)
    if len(events) < MIN_CLASSIFIER_EVENTS:
        return None
    if len({event["feedback"] for event in events}) < 2:
        return None
    profile = feedback_preference_profile(state)
    if profile is None:
        return None
    scaler = artifact["scaler"]
    preference = (
        profile * (FEATURE_MAX - FEATURE_MIN) + FEATURE_MIN - scaler.mean_
    ) / scaler.scale_
    track_block = STANDARDIZED_FEATURES[candidate_pool]
    classifier_features = list(artifact["classifier_features"])
    frame = pd.DataFrame(
        np.column_stack([track_block, np.tile(preference, (len(candidate_pool), 1))]),
        columns=classifier_features,
    )
    model = artifact["classifier"]
    probabilities = model.predict_proba(frame)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    like_column = int(np.flatnonzero(classes == 1)[0]) if np.any(classes == 1) else probabilities.shape[1] - 1
    return np.asarray(probabilities[:, like_column], dtype=np.float32)


def artist_main(row: pd.Series) -> str:
    artists = row["_artists"]
    return artists[0] if artists else ""


def get_user_oauth(state: str | None = None, scopes: str | None = None) -> SpotifyOAuth:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to the project .env file.")
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        scope=scopes or SCOPES,
        state=state,
        cache_path=str(ROOT / ".spotify_token_cache"),
        open_browser=False,
        show_dialog=True,
    )


def user_access_token() -> dict:
    """Return the cached OAuth token without starting an interactive flow.

    Spotipy's auth managers validate the cached scopes and, when they do not
    match the current SCOPES string, print an authorization URL and wait for
    console input. That is fine for a CLI, but it would stall a web request.
    """
    oauth = get_user_oauth()
    token = oauth.cache_handler.get_cached_token()
    if not token:
        raise RuntimeError("Connect Spotify first.")
    if oauth.is_token_expired(token):
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("Spotify session expired. Reconnect Spotify.")
        token = oauth.refresh_access_token(refresh_token)
    return token


def get_user_client():
    # Use the same grant as the browser player. Asking Spotipy for a smaller
    # scope set here can return a cached token without `streaming`, which the
    # Web Playback SDK rejects as an invalid token scope.
    return spotipy.Spotify(auth=user_access_token()["access_token"])


def granted_user_scopes() -> set[str]:
    try:
        return set(str(user_access_token().get("scope", "")).split())
    except Exception:
        return set()


@lru_cache(maxsize=1)
def get_catalog_client():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=manager)


def spotify_connected() -> bool:
    try:
        oauth = get_user_oauth()
        token = oauth.cache_handler.get_cached_token()
        if not token:
            return False
        if oauth.is_token_expired(token):
            if not token.get("refresh_token"):
                return False
            try:
                token = oauth.refresh_access_token(token["refresh_token"])
            except Exception:
                return False
            if not token or oauth.is_token_expired(token):
                return False
        granted_scopes = set(str(token.get("scope", "")).split())
        # Keep playback usable with an older cached grant. Playlist and history
        # features report their own missing-scope reconnect prompt when needed.
        return PLAYBACK_SCOPES.issubset(granted_scopes)
    except Exception:
        return False


SPOTIFY_ID_COL = next((c for c in TRACKS.columns if str(c).casefold() == "id"), None)


def spotify_id_for(track_index: int) -> str | None:
    """The dataset already carries Spotify track IDs, so use them directly."""
    if SPOTIFY_ID_COL is None:
        return None
    value = TRACKS.iloc[int(track_index)].get(SPOTIFY_ID_COL)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9]{22}", value.strip()):
        return value.strip()
    return None


@lru_cache(maxsize=8192)
def spotify_art_for(spotify_id: str) -> str | None:
    """Album art via Spotify's public oEmbed endpoint (no auth, reliable)."""
    url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/track/{spotify_id}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TuneTwin/1.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("thumbnail_url")
    except Exception:
        return None


@lru_cache(maxsize=4096)
def spotify_match(track_index: int) -> dict | None:
    """Direct Spotify track for a dataset row, falling back to a search match."""
    row = TRACKS.iloc[int(track_index)]
    spotify_id = spotify_id_for(int(track_index))
    if spotify_id:
        # The dataset ID is authoritative, so playback never depends on search.
        return {
            "id": spotify_id,
            "uri": f"spotify:track:{spotify_id}",
            "name": row["_name"],
            "artist": row["_artist_display"],
            "url": f"https://open.spotify.com/track/{spotify_id}",
            "image": spotify_art_for(spotify_id),
            "confidence": 1.0,
        }

    client = get_catalog_client()
    if client is None:
        return None
    title = row["_name"]
    artist = artist_main(row)
    query = f'track:"{title}" artist:"{artist}"' if artist else f'track:"{title}"'
    try:
        items = client.search(q=query, type="track", limit=10).get("tracks", {}).get("items", [])
    except Exception:
        return None
    norm = lambda value: re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    best = None
    for item in items:
        candidate_title = item.get("name", "")
        candidate_artist = ", ".join(a.get("name", "") for a in item.get("artists", []))
        title_score = SequenceMatcher(None, norm(title), norm(candidate_title)).ratio()
        artist_score = SequenceMatcher(None, norm(artist), norm(candidate_artist)).ratio() if artist else 0.5
        confidence = 0.65 * title_score + 0.35 * artist_score
        if best is None or confidence > best["confidence"]:
            images = (item.get("album") or {}).get("images") or []
            # Prefer a mid-size image (~300px) for thumbnails; fall back to any.
            image = None
            if images:
                image = images[1].get("url") if len(images) > 1 else images[0].get("url")
            best = {
                "id": item.get("id"),
                "uri": item.get("uri"),
                "name": candidate_title,
                "artist": candidate_artist,
                "url": item.get("external_urls", {}).get("spotify"),
                "image": image,
                "confidence": confidence,
            }
    return best if best and best["confidence"] >= 0.55 else None


def catalog_index_for_spotify_track(item: dict) -> int | None:
    """Map a Spotify history/current track to its closest dataset row."""
    title_key = normalize_song_title(item.get("name", ""))
    candidates = TITLE_ROWS.get(title_key, [])
    if not candidates:
        return None
    spotify_artists = {
        normalize_artist_name(artist.get("name", ""))
        for artist in item.get("artists", []) if artist.get("name")
    }
    if not spotify_artists:
        return candidates[0]
    best_idx, best_score = None, -1.0
    for idx in candidates:
        artist_keys = TRACKS.iloc[idx]["_artist_identity_keys"]
        score = 1.0 if artist_keys & spotify_artists else max(
            (SequenceMatcher(None, expected, actual).ratio()
             for expected in artist_keys for actual in spotify_artists),
            default=0.0,
        )
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx if best_score >= 0.55 else None


def queue_catalog_tracks(client, row_indices: list[int], device_id: str | None) -> dict[str, int]:
    """Append distinct catalog matches to Spotify's existing queue."""
    result = {"queued": 0, "already_queued": 0, "missing": 0, "failed": 0}
    queued_uris: set[str] = set()
    try:
        current_queue = client.queue()
        current_item = current_queue.get("currently_playing") or {}
        if current_item.get("uri"):
            queued_uris.add(current_item["uri"])
        queued_uris.update(item.get("uri") for item in current_queue.get("queue", []) if item.get("uri"))
    except Exception:
        pass
    for idx in row_indices:
        match = spotify_match(int(idx))
        if not match or not match.get("uri"):
            result["missing"] += 1
            continue
        if match["uri"] in queued_uris:
            result["already_queued"] += 1
            continue
        try:
            client.add_to_queue(match["uri"], device_id=device_id)
            queued_uris.add(match["uri"])
            result["queued"] += 1
        except Exception:
            result["failed"] += 1
    return result


def find_or_create_private_playlist(client, name: str, description: str) -> dict:
    """Find this app's private playlist, or create it for the current user."""
    page = client.current_user_playlists(limit=50)
    while page:
        for playlist in page.get("items", []):
            if playlist.get("name") == name and playlist.get("id"):
                return playlist
        if not page.get("next"):
            break
        page = client.next(page)
    return client._post("me/playlists", payload={
        "name": name,
        "description": description,
        "public": False,
    })


def playlist_track_uris(client, playlist_id: str) -> set[str]:
    uris: set[str] = set()
    page = client._get(f"playlists/{playlist_id}/items", args={"limit": 100})
    while page:
        for entry in page.get("items", []):
            item = entry.get("item") or entry.get("track") or {}
            if item.get("uri"):
                uris.add(item["uri"])
        if not page.get("next"):
            break
        page = client.next(page)
    return uris


def add_liked_track_to_playlist(client, uri: str) -> tuple[dict, bool]:
    playlist = find_or_create_private_playlist(
        client,
        "TuneTwin Liked Songs",
        "Tracks explicitly liked in TuneTwin.",
    )
    if uri in playlist_track_uris(client, playlist["id"]):
        return playlist, False
    client._post(f"playlists/{playlist['id']}/items", payload={"uris": [uri]})
    return playlist, True


_HISTORY_TTL_SECONDS = 60
_history_cache: dict = {"at": 0.0, "rows": ()}


def recent_history_rows(client) -> tuple[int, ...]:
    """Recently played Spotify tracks mapped to catalog rows (briefly cached)."""
    now = time.monotonic()
    if now - _history_cache["at"] < _HISTORY_TTL_SECONDS:
        return _history_cache["rows"]
    try:
        history = client.current_user_recently_played(limit=50)
    except Exception:
        try:
            history = client.current_user_recently_played(limit=50)
        except Exception:
            return tuple(_history_cache["rows"])
    rows: list[int] = []
    seen: set[int] = set()
    for entry in history.get("items", []):
        item = entry.get("track") or {}
        idx = catalog_index_for_spotify_track(item)
        if idx is not None and idx not in seen:
            seen.add(idx)
            rows.append(idx)
    _history_cache["at"] = now
    _history_cache["rows"] = tuple(rows)
    return _history_cache["rows"]


def mood_weight_vector(profile: dict) -> np.ndarray:
    """Turn a mood profile into a weight vector over the catalog's audio features."""
    positions = {name: AUDIO_FEATURES.index(name) for name in AUDIO_FEATURES}
    weights = np.zeros(len(AUDIO_FEATURES), dtype=np.float32)
    for feature, weight in profile.items():
        if feature in positions:
            weights[positions[feature]] = weight
    return weights


def rank_history_candidates(scores: np.ndarray, recent_rows: tuple[int, ...]) -> list[int]:
    """Eligible recent tracks ordered by how well they fit one mood."""
    candidates = [int(idx) for idx in recent_rows if MOOD_ELIGIBLE[int(idx)]]
    candidates.sort(key=lambda idx: float(scores[idx]), reverse=True)
    return candidates


def pick_mood_anchors(
    scores: np.ndarray,
    used_artists: set[str],
    count: int = 3,
    excluded: set[int] | None = None,
) -> list[int]:
    """Pick recognizable, high-scoring catalogue seeds for remaining slots."""
    excluded = excluded or set()
    eligible = np.flatnonzero(MOOD_ELIGIBLE)
    if not eligible.size:
        return []
    # Mood fit decides the candidate pool; popularity chooses the familiar
    # faces inside that pool instead of letting an obscure outlier lead.
    shortlist = eligible[np.argsort(-scores[eligible])[:2000]]
    ranked = shortlist[np.lexsort((-scores[shortlist], -POPULARITY[shortlist]))]
    anchors: list[int] = []
    artists: set[str] = set(used_artists)
    clusters: set[int] = set()
    # Prefer a different K-Means cluster per anchor, then relax that constraint
    # if the catalogue cannot supply enough distinct clusters.
    for require_cluster in (True, False):
        for raw_idx in ranked:
            idx = int(raw_idx)
            if idx in excluded:
                continue
            artist = PRIMARY_ARTIST_KEYS[idx]
            cluster = int(CLUSTER_LABELS[idx])
            if artist in artists:
                continue
            if require_cluster and cluster in clusters:
                continue
            anchors.append(idx)
            artists.add(artist)
            clusters.add(cluster)
            if len(anchors) == count:
                return anchors
    return anchors


@lru_cache(maxsize=8)
def catalog_mood_mixes(updates: int, recent_rows: tuple[int, ...] = ()) -> tuple[dict, ...]:
    """Build four mood playlists from recent listening plus the recommender.

    `updates` and `recent_rows` are cache keys: mixes rebuild when Like/Dislike
    learning changes or the listening history changes.
    """
    mixes: list[dict] = []
    used_artists: set[str] = set()
    mood_scores = {
        mood: STANDARDIZED_FEATURES @ mood_weight_vector(profile)
        for mood, profile in MOOD_PROFILES.items()
    }
    ranked_history = {
        mood: rank_history_candidates(mood_scores[mood], recent_rows)
        for mood in MOOD_PROFILES
    }
    history_assignments: dict[str, list[int]] = {mood: [] for mood in MOOD_PROFILES}
    used_history_tracks: set[int] = set()
    used_history_artists: set[str] = set()
    # Give every mood one history seed before any mood gets a second, so the
    # recent listening history is represented across all four mixes.
    for _ in range(3):
        for mood in MOOD_PROFILES:
            for idx in ranked_history[mood]:
                artist = PRIMARY_ARTIST_KEYS[idx]
                if idx in used_history_tracks or artist in used_history_artists:
                    continue
                history_assignments[mood].append(idx)
                used_history_tracks.add(idx)
                used_history_artists.add(artist)
                break
    for mood, profile in MOOD_PROFILES.items():
        scores = mood_scores[mood]
        history_anchors = history_assignments[mood]
        own_history = set(history_anchors)
        anchors = list(history_anchors)
        if len(anchors) < 3:
            anchors += pick_mood_anchors(
                scores,
                used_artists | {PRIMARY_ARTIST_KEYS[anchor] for anchor in anchors},
                count=3 - len(anchors),
                excluded=used_history_tracks,
            )
        if not anchors:
            continue
        used_artists.update(PRIMARY_ARTIST_KEYS[anchor] for anchor in anchors)
        # `with_art=False` avoids ~24 oEmbed requests per mood; mix cards use
        # their generated mosaic and playback fetches art on demand.
        rec_lists = []
        for anchor in anchors:
            candidates = [(anchor, 1.0)] + [
                (int(rec["index"]), float(rec.get("score", 0.0)))
                for rec in recommend(anchor, limit=12, with_art=False)
            ]
            # Content-similar tracks can still clash with the mix's intended
            # mood. Rank each anchor's suggestions by that same mood profile
            # before the lists are interleaved below.
            candidates.sort(key=lambda item: (float(scores[item[0]]), item[1]), reverse=True)
            rec_lists.append([anchor] + [idx for idx, _score in candidates if idx != anchor])
        ordered: list[int] = []
        seen: set[int] = set()
        artist_counts: dict[str, int] = {}
        for slot in range(max(len(recs) for recs in rec_lists)):
            for recs in rec_lists:
                if slot >= len(recs):
                    continue
                idx = int(recs[slot])
                if (
                    idx in seen
                    or (idx in used_history_tracks and idx not in own_history)
                    or not MOOD_ELIGIBLE[idx]
                ):
                    continue
                artist = PRIMARY_ARTIST_KEYS[idx]
                if artist_counts.get(artist, 0) >= 2:
                    continue
                seen.add(idx)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
                ordered.append(idx)
                if len(ordered) == 24:
                    break
            if len(ordered) == 24:
                break
        if not ordered:
            continue
        label, description = MOOD_MIX_COPY[mood]
        if history_anchors:
            description += " Seeded by your recent listening."
        tracks = []
        for idx in ordered:
            row = TRACKS.iloc[idx]
            spotify_id = spotify_id_for(idx)
            tracks.append({
                "index": idx,
                "name": row["_name"],
                "artist": row["_artist_display"],
                "uri": f"spotify:track:{spotify_id}" if spotify_id else None,
                "anchor": idx in anchors,
            })
        lead = TRACKS.iloc[anchors[0]]
        mixes.append({
            "mood": mood,
            "name": label,
            "description": description,
            "image": f"/static/moods/{mood.lower()}.svg",
            "seed_index": anchors[0],
            "seed_name": lead["_name"],
            "seed_artist": lead["_artist_display"],
            "anchors": [int(anchor) for anchor in anchors],
            "history_seeded": bool(history_anchors),
            "tracks": tracks,
        })
    return tuple(mixes)


def recommend(seed_idx: int, limit: int = 8, with_art: bool = True) -> list[dict]:
    seed = TRACKS.iloc[seed_idx]
    audio_scores = np.clip((AUDIO_VECTORS @ AUDIO_VECTORS[seed_idx] + 1.0) / 2.0, 0.0, 1.0)
    neighbor_count = min(5001, len(TRACKS))
    neighbor_distances, neighbor_indices = PCA_KNN.kneighbors(
        PCA_VECTORS[seed_idx].reshape(1, -1), n_neighbors=neighbor_count
    )
    neighbor_distances = neighbor_distances[0]
    neighbor_indices = neighbor_indices[0]
    distance_scale = max(float(neighbor_distances[-1]), 1e-8)
    knn_scores = np.zeros(len(TRACKS), dtype=np.float32)
    knn_scores[neighbor_indices] = np.clip(1.0 - neighbor_distances / distance_scale, 0.0, 1.0)
    knn_scores[seed_idx] = 0.0
    cluster_scores = (CLUSTER_LABELS == CLUSTER_LABELS[seed_idx]).astype(np.float32)
    seed_genres = seed["_genres"]
    seed_artists = seed["_artist_keys"]
    genre_scores = np.zeros(len(TRACKS), dtype=np.float32)
    if seed_genres:
        genre_columns = [GENRE_INDEX[name] for name in seed_genres if name in GENRE_INDEX]
        if genre_columns:
            seed_genre_vector = np.zeros(len(GENRE_ENCODER.classes_), dtype=np.float32)
            seed_genre_vector[genre_columns] = 1.0
            intersections = np.asarray(GENRE_MATRIX @ seed_genre_vector).ravel()
            unions = GENRE_COUNTS + len(genre_columns) - intersections
            genre_scores = np.divide(intersections, unions, out=genre_scores, where=unions > 0)
    artist_scores = np.zeros(len(TRACKS), dtype=np.float32)
    for key in seed_artists:
        artist_scores[ARTIST_ROWS.get(key, [])] = 1.0
    weights = (0.40, 0.25, 0.10, 0.15 if seed_genres else 0.0, 0.10 if seed_artists else 0.0)
    weight_sum = sum(weights)
    content_scores = (
        weights[0] * audio_scores
        + weights[1] * knn_scores
        + weights[2] * cluster_scores
        + weights[3] * genre_scores
        + weights[4] * artist_scores
    ) / weight_sum
    content_scores[seed_idx] = -1

    # Build a Top-50 content pool from a wider nearest-neighbor shortlist. The
    # pool starts with distinct lead artists, then may trade a weak slot for a
    # near-cutoff repeat (maximum three per lead artist).
    seed_title_key = seed["_title_key"]
    seed_artist_keys = seed["_artist_identity_keys"]
    source_pool_list: list[int] = []
    seen_song_keys = {(seed_title_key, seed["_song_key"])}
    for raw_idx in neighbor_indices:
        idx = int(raw_idx)
        # Exclude alternate catalog rows of the selected song, including
        # case, punctuation, featuring-credit, and common remaster variants.
        if TITLE_KEYS[idx] == seed_title_key and (ARTIST_IDENTITY_KEYS[idx] & seed_artist_keys):
            continue
        song_key = SONG_KEYS[idx]
        if not song_key[0] or song_key in seen_song_keys:
            continue
        seen_song_keys.add(song_key)
        source_pool_list.append(idx)
    source_pool = np.asarray(source_pool_list, dtype=int)
    source_pool = source_pool[np.argsort(content_scores[source_pool])[::-1]]
    pool_list: list[int] = []
    pool_artist_counts: dict[str, int] = {}
    for raw_idx in source_pool:
        idx = int(raw_idx)
        key = PRIMARY_ARTIST_KEYS[idx]
        if pool_artist_counts.get(key, 0) != 0:
            continue
        pool_list.append(idx)
        pool_artist_counts[key] = 1
        if len(pool_list) == 50:
            break
    if len(pool_list) < min(50, len(TRACKS) - 1):
        for raw_idx in source_pool:
            idx = int(raw_idx)
            key = PRIMARY_ARTIST_KEYS[idx]
            if pool_artist_counts.get(key, 0) >= MAX_TRACKS_PER_ARTIST:
                continue
            pool_list.append(idx)
            pool_artist_counts[key] = pool_artist_counts.get(key, 0) + 1
            if len(pool_list) == min(50, len(TRACKS) - 1):
                break
    if len(pool_list) == 50:
        pool_cutoff = float(content_scores[pool_list[-1]])
        for raw_idx in source_pool:
            idx = int(raw_idx)
            if idx in pool_list:
                continue
            key = PRIMARY_ARTIST_KEYS[idx]
            if pool_artist_counts.get(key, 0) >= MAX_TRACKS_PER_ARTIST:
                continue
            if content_scores[idx] < pool_cutoff - 0.04:
                break
            replace_at = min(
                (pos for pos, selected_idx in enumerate(pool_list) if PRIMARY_ARTIST_KEYS[selected_idx] != key),
                key=lambda pos: content_scores[pool_list[pos]],
                default=None,
            )
            if replace_at is None:
                continue
            removed_idx = pool_list[replace_at]
            if content_scores[idx] < content_scores[removed_idx] - 0.04:
                break
            removed_key = PRIMARY_ARTIST_KEYS[removed_idx]
            pool_artist_counts[removed_key] -= 1
            pool_list[replace_at] = idx
            pool_artist_counts[key] = pool_artist_counts.get(key, 0) + 1
    candidate_pool = np.asarray(pool_list, dtype=int)

    # Use the tuned notebook classifier once real feedback contains both Like
    # and Skip. Before that, cold start relies on content signals and LinUCB.
    state = load_bandit_state()
    classifier_scores = classifier_probabilities(candidate_pool, state)

    # Keep the small online KNN as a fallback when the shared artifact is
    # unavailable, so the live demo still learns from explicit feedback.
    knn_preference_scores: np.ndarray | None = None
    if classifier_scores is None:
        feedback_events = valid_feedback_events(state)
        feedback_rows = np.asarray([event["track_idx"] for event in feedback_events], dtype=int)
        feedback_labels = np.asarray([event["feedback"] == "like" for event in feedback_events], dtype=int)
        if len(feedback_rows) >= 2 and len(np.unique(feedback_labels)) == 2:
            preference_model = KNeighborsClassifier(
                n_neighbors=min(5, len(feedback_rows)), weights="distance"
            )
            preference_model.fit(MODEL_CONTEXTS[feedback_rows], feedback_labels)
            like_column = int(np.flatnonzero(preference_model.classes_ == 1)[0])
            knn_preference_scores = preference_model.predict_proba(MODEL_CONTEXTS[candidate_pool])[:, like_column]

    # LinUCB reranks the diversified Top-50 using the same PCA/cluster context.
    inverse = np.linalg.inv(state["A"])
    theta = inverse @ state["b"]
    contexts = MODEL_CONTEXTS[candidate_pool].astype(float)
    predicted_reward = contexts @ theta
    uncertainty = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", contexts, inverse, contexts)))
    ucb = predicted_reward + LINUCB_ALPHA * uncertainty
    bandit_scores = np.clip((np.tanh(ucb) + 1.0) / 2.0, 0.0, 1.0)
    ranking_scores = content_scores.copy()
    feedback_scores = content_scores[candidate_pool]
    if classifier_scores is not None:
        feedback_scores = (1.0 - CLASSIFIER_BLEND) * feedback_scores + CLASSIFIER_BLEND * classifier_scores
    elif knn_preference_scores is not None:
        feedback_scores = 0.80 * feedback_scores + 0.20 * knn_preference_scores
    ranking_scores[candidate_pool] = (1.0 - LINUCB_BLEND) * feedback_scores + LINUCB_BLEND * bandit_scores
    ranking_scores[seed_idx] = -1
    order = np.lexsort((candidate_pool, -ranking_scores[candidate_pool]))
    ranked = candidate_pool[order]
    selected: list[int] = []
    artist_counts: dict[str, int] = {}
    selected_songs: set[tuple[str, str]] = set()
    # First pass favors discovery: fill the list with one track per artist.
    for raw_idx in ranked:
        idx = int(raw_idx)
        if SONG_KEYS[idx] in selected_songs:
            continue
        keys = ARTIST_KEYS[idx]
        if any(artist_counts.get(key, 0) >= 1 for key in keys):
            continue
        selected.append(idx)
        selected_songs.add(SONG_KEYS[idx])
        for key in keys:
            artist_counts[key] = artist_counts.get(key, 0) + 1
        if len(selected) == limit:
            break

    # Allow a close-scoring second track from an artist, while keeping at least
    # preserve a variety of lead artists while keeping a hard cap of three.
    repeat_margin = 0.04
    if len(selected) == limit and limit >= 4:
        for raw_idx in ranked:
            idx = int(raw_idx)
            if idx in selected or SONG_KEYS[idx] in selected_songs:
                continue
            keys = ARTIST_KEYS[idx]
            if not any(artist_counts.get(key, 0) >= 1 for key in keys):
                continue
            if any(artist_counts.get(key, 0) >= MAX_TRACKS_PER_ARTIST for key in keys):
                continue
            weakest_position = min(range(len(selected)), key=lambda pos: ranking_scores[selected[pos]])
            weakest_idx = selected[weakest_position]
            if ranking_scores[idx] < ranking_scores[weakest_idx] - repeat_margin:
                break
            removed_keys = TRACKS.iloc[weakest_idx]["_artist_keys"]
            if any(key in removed_keys and artist_counts.get(key, 0) == 1 for key in keys):
                continue
            trial = selected.copy()
            trial[weakest_position] = idx
            primary_artist_count = len({PRIMARY_ARTIST_KEYS[item] for item in trial})
            minimum_distinct = min(
                len(selected),
                max(MIN_DISTINCT_LEAD_ARTISTS, limit - (MAX_TRACKS_PER_ARTIST - 1)),
            )
            if primary_artist_count < minimum_distinct:
                continue
            for key in removed_keys:
                artist_counts[key] -= 1
            if any(artist_counts.get(key, 0) >= MAX_TRACKS_PER_ARTIST for key in keys):
                for key in removed_keys:
                    artist_counts[key] += 1
                continue
            selected[weakest_position] = idx
            selected_songs.discard(SONG_KEYS[weakest_idx])
            selected_songs.add(SONG_KEYS[idx])
            for key in keys:
                artist_counts[key] = artist_counts.get(key, 0) + 1
            selected.sort(key=lambda idx: ranking_scores[idx], reverse=True)

    results = []
    result_song_keys: set[tuple[str, str]] = set()
    for idx in selected:
        if SONG_KEYS[idx] in result_song_keys:
            continue
        result_song_keys.add(SONG_KEYS[idx])
        row = TRACKS.iloc[idx]
        match = spotify_match(idx) if with_art else None
        popularity = row.get("popularity")
        results.append({
            "index": idx,
            "name": row["_name"],
            "artist": row["_artist_display"],
            "genres": sorted(row["_genres"])[:5],
            "score": float(ranking_scores[idx]),
            "similarity": float(content_scores[idx]),
            "bandit_score": float(bandit_scores[np.where(candidate_pool == idx)[0][0]]),
            "classifier_score": (
                float(classifier_scores[np.where(candidate_pool == idx)[0][0]])
                if classifier_scores is not None else None
            ),
            "knn_preference": (
                float(knn_preference_scores[np.where(candidate_pool == idx)[0][0]])
                if knn_preference_scores is not None else None
            ),
            "match": match,
            "popularity": int(popularity) if pd.notna(popularity) else None,
            "reason": recommendation_reason(seed_idx, idx, audio_scores[idx], knn_scores[idx], cluster_scores[idx], genre_scores[idx], artist_scores[idx]),
        })
    return results


def recommendation_reason(seed_idx: int, idx: int, audio: float, knn: float, cluster: float, genre: float, artist: float) -> str:
    """Human-readable explanation from the same signals used for ranking."""
    reasons = [(float(audio), "similar audio profile")]
    if knn > 0:
        reasons.append((float(knn), "nearby PCA neighbors"))
    if cluster > 0:
        reasons.append((float(cluster), "same K-Means cluster"))
    if genre > 0:
        reasons.append((float(genre), "shared artist-genre tags"))
    if artist > 0:
        reasons.append((float(artist), "same artist"))
    reasons.sort(reverse=True)
    return " · ".join(label for _, label in reasons[:2])


def csrf_value() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


@app.before_request
def check_csrf():
    if request.method == "POST":
        expected = session.get("_csrf", "")
        submitted = request.form.get("csrf_token", "") or request.headers.get("X-CSRF-Token", "")
        if not expected or not secrets.compare_digest(expected, submitted):
            abort(400, "Invalid form token. Refresh the page and try again.")


@app.context_processor
def inject_template_helpers():
    return {
        "csrf_token": csrf_value,
        "spotify_connected": spotify_connected,
        "bandit_update_count": load_bandit_state()["updates"],
    }


def search_tracks(query: str, limit: int = 12) -> list[dict]:
    query = query.strip().casefold()
    if len(query) < 2:
        return []
    title_match = TRACKS["_name"].str.casefold().str.contains(re.escape(query), regex=True, na=False)
    artist_match = TRACKS["_artist_display"].str.casefold().str.contains(re.escape(query), regex=True, na=False)
    indices = np.flatnonzero((title_match | artist_match).to_numpy())
    if not len(indices):
        return []
    starts = TRACKS.iloc[indices]["_name"].str.casefold().str.startswith(query).to_numpy()
    popularity = pd.to_numeric(TRACKS.iloc[indices].get("popularity", pd.Series(0, index=indices)), errors="coerce").fillna(0).to_numpy()
    order = np.lexsort((-popularity, ~starts))[:limit]
    results = []
    for local_idx in order:
        idx = int(indices[local_idx])
        row = TRACKS.iloc[idx]
        results.append({"index": idx, "name": row["_name"], "artist": row["_artist_display"]})
    return results


def render_main(seed_idx: int | None = None):
    seed = None
    suggestions = []
    if seed_idx is not None:
        if seed_idx < 0 or seed_idx >= len(TRACKS):
            abort(404, "Song not found in the local catalog.")
        row = TRACKS.iloc[seed_idx]
        seed = {"index": seed_idx, "name": row["_name"], "artist": row["_artist_display"], "genres": sorted(row["_genres"])[:8]}
        suggestions = recommend(seed_idx)
    return render_template(
        "index.html",
        seed=seed,
        recommendations=suggestions,
        features=AUDIO_FEATURES,
        pca_components=PCA_VECTORS.shape[1],
        selected_k=SELECTED_K,
        best_silhouette=KMEANS_SILHOUETTES.get(SELECTED_K, 0.0),
        song_count=len(TRACKS),
        connected=spotify_connected(),
    )


@app.get("/")
def home():
    seed_idx = request.args.get("seed", type=int)
    return render_main(seed_idx)


@app.get("/api/search")
def api_search():
    return {"results": search_tracks(request.args.get("q", ""))}


@app.get("/api/spotify/token")
def spotify_sdk_token():
    """Expose a short-lived access token to the browser SDK only after OAuth."""
    if not spotify_connected():
        return {"error": "Connect Spotify first."}, 401
    try:
        return {"access_token": user_access_token()["access_token"]}
    except Exception:
        return {"error": "Spotify token expired. Reconnect Spotify."}, 401


def spotify_client_or_error():
    if not spotify_connected():
        return None, ({"error": "Connect Spotify first."}, 401)
    try:
        return get_user_client(), None
    except Exception as exc:
        return None, ({"error": f"Spotify request failed ({type(exc).__name__})."}, 502)


@app.get("/api/spotify/state")
def spotify_state():
    client, error = spotify_client_or_error()
    if error:
        return error
    try:
        playback = client.current_playback() or {}
        item = playback.get("item") or {}
        album = item.get("album") or {}
        images = album.get("images") or []
        return {
            "is_playing": bool(playback.get("is_playing")),
            "progress_ms": playback.get("progress_ms") or 0,
            "duration_ms": item.get("duration_ms") or 0,
            "track": item.get("name"),
            "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])),
            "image": images[0].get("url") if images else None,
            "uri": item.get("uri"),
            "index": catalog_index_for_spotify_track(item) if item else None,
            "device_id": (playback.get("device") or {}).get("id"),
            "device_name": (playback.get("device") or {}).get("name"),
            "shuffle": playback.get("shuffle_state"),
            "repeat": playback.get("repeat_state"),
        }
    except Exception:
        return {"error": "Spotify playback status is unavailable right now."}, 502


@app.get("/api/spotify/devices")
def spotify_devices():
    client, error = spotify_client_or_error()
    if error:
        return error
    try:
        return {"devices": [{"id": d.get("id"), "name": d.get("name"), "type": d.get("type"), "active": d.get("is_active"), "restricted": d.get("is_restricted")} for d in client.devices().get("devices", [])]}
    except Exception:
        return {"error": "Spotify devices are unavailable right now."}, 502


def playlist_response(payload: dict) -> dict:
    return {
        "name": payload["name"],
        "description": payload["description"],
        "playlist_id": payload["playlist_id"],
        "snapshot_id": payload["snapshot_id"],
        "image": payload["image"],
        "total_tracks": payload["total_tracks"],
        "tracks": payload["tracks"],
        "recommendations": cached_playlist_recommendations(payload),
    }


@app.get("/api/spotify/playlists")
def spotify_playlists():
    client, error = spotify_client_or_error()
    if error:
        return error
    try:
        playlists = []
        page = client.current_user_playlists(limit=50)
        while page and len(playlists) < 50:
            for item in page.get("items", []):
                if not item or not item.get("id"):
                    continue
                images = item.get("images") or []
                counts = item.get("items") or item.get("tracks") or {}
                playlists.append({
                    "id": item["id"],
                    "name": item.get("name") or "Untitled playlist",
                    "tracks": counts.get("total", 0),
                    "image": images[0]["url"] if images else None,
                    "description": item.get("description") or "",
                    "owner": (item.get("owner") or {}).get("display_name") or "",
                })
            if not page.get("next"):
                break
            page = client.next(page)
        return {
            "playlists": playlists,
            "can_edit_public": "playlist-modify-public" in granted_user_scopes(),
        }
    except Exception:
        return {"error": "Could not load your Spotify playlists right now."}, 502


@app.get("/api/playlist/<playlist_id>/recommendations")
def playlist_recommendations_route(playlist_id: str):
    """Auto-load one playlist with model picks to add, no manual import needed."""
    client, error = spotify_client_or_error()
    if error:
        return error
    resolved = resolve_playlist_id(playlist_id)
    if not resolved:
        return {"error": "That playlist link is not valid."}, 400
    try:
        payload = load_playlist_payload(client, resolved)
    except Exception:
        return {"error": "That Spotify playlist could not be opened."}, 404
    return playlist_response(payload)


@app.post("/api/playlist/recommend")
def playlist_recommend():
    """Resolve a pasted playlist link and return it with model recommendations."""
    client, error = spotify_client_or_error()
    if error:
        return error
    playlist_id = resolve_playlist_id((request.get_json(silent=True) or {}).get("playlist", ""))
    if not playlist_id:
        return {"error": "Paste a Spotify playlist link or ID."}, 400
    try:
        payload = load_playlist_payload(client, playlist_id)
    except Exception:
        return {"error": "That Spotify playlist could not be opened."}, 404
    if not payload["seed_rows"]:
        return {"error": "None of this playlist's tracks are in the TuneTwin catalogue."}, 404
    return playlist_response(payload)


@app.post("/api/playlist/add")
def playlist_add():
    """Add one recommended track to a playlist the user owns or collaborates on."""
    client, error = spotify_client_or_error()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    playlist_id = resolve_playlist_id(data.get("playlist_id", ""))
    uri = data.get("uri")
    if not playlist_id or not isinstance(uri, str) or not uri.startswith("spotify:track:"):
        return {"error": "Choose a valid playlist and track."}, 400
    try:
        client.playlist_add_items(playlist_id, [uri])
    except Exception as exc:
        if getattr(exc, "http_status", None) == 403:
            return {"error": "Spotify blocked this edit. Reconnect Spotify (new permission) and use a playlist you own."}, 403
        return {"error": f"Could not add that song to the playlist ({type(exc).__name__})."}, 502
    invalidate_playlist_cache(playlist_id)
    return {"ok": True, "message": "Added to playlist."}


@app.get("/api/spotify/queue")
def spotify_queue():
    client, error = spotify_client_or_error()
    if error:
        return error
    try:
        queue = client.queue()
        def brief(item):
            if not item:
                return None
            return {"name": item.get("name"), "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])), "uri": item.get("uri"), "url": (item.get("external_urls") or {}).get("spotify")}
        current = brief(queue.get("currently_playing"))
        items = [brief(item) for item in queue.get("queue", [])[:50]]
        # The Web API may report the same URI many times after repeated Play
        # clicks. Compress adjacent duplicates for a legible, truthful view.
        compact = []
        for item in items:
            if not item:
                continue
            if compact and item.get("uri") and compact[-1]["item"].get("uri") == item["uri"]:
                compact[-1]["count"] += 1
            else:
                compact.append({"item": item, "count": 1})
        return {
            "current": current,
            "items": [{**entry["item"], "repeat_count": entry["count"]} for entry in compact],
            "raw_count": len(items),
        }
    except Exception as exc:
        return {"error": f"Queue unavailable ({type(exc).__name__})."}, 502


@app.get("/api/mood-mixes")
def model_mood_mixes():
    """Four playlists seeded by recent listening and refreshed after feedback."""
    updates = int(load_bandit_state().get("updates", 0))
    recent_rows: tuple[int, ...] = ()
    if spotify_connected():
        try:
            recent_rows = recent_history_rows(get_user_client())
        except Exception:
            recent_rows = ()
    return {"mixes": list(catalog_mood_mixes(updates, recent_rows))}


@app.post("/spotify/recommend-current")
def recommend_current_track():
    client, error = spotify_client_or_error()
    if error:
        flash(error[0].get("error", "Connect Spotify first."), "error")
        return redirect(url_for("home"))
    try:
        playback = client.current_playback() or {}
        item = playback.get("item") or {}
        seed_idx = catalog_index_for_spotify_track(item)
        if seed_idx is None:
            flash("The currently playing Spotify track was not found in this dataset. Try the manual catalog search below.", "error")
            return redirect(url_for("home"))
        device = playback.get("device") or {}
        if not device.get("id"):
            device = next((d for d in client.devices().get("devices", []) if d.get("is_active")), {})
        if not device.get("id"):
            flash("Open Spotify on a device and start playback, then try again.", "error")
            return redirect(url_for("home", seed=seed_idx))
        rows = [rec["index"] for rec in recommend(seed_idx, limit=8)]
        summary = queue_catalog_tracks(client, rows, device["id"])
        flash(f"Using {item.get('name')} as your seed. Queued {summary['queued']} similar songs in Spotify.", "success")
        if summary["already_queued"]:
            flash(f"Skipped {summary['already_queued']} songs already in your queue.", "success")
        if summary["missing"] or summary["failed"]:
            flash(f"{summary['missing']} songs had no Spotify match; {summary['failed']} queue requests failed.", "error")
        return redirect(url_for("home", seed=seed_idx))
    except Exception as exc:
        flash(f"Couldn't read your current Spotify track ({type(exc).__name__}).", "error")
        return redirect(url_for("home"))


@app.post("/spotify/create-mood-playlist")
def create_mood_playlist():
    mood = request.form.get("mood", "")
    if mood not in MOOD_PROFILES:
        flash("That mood mix could not be exported.", "error")
        return redirect(url_for("home"))
    client, error = spotify_client_or_error()
    if error:
        flash(error[0].get("error", "Connect Spotify first."), "error")
        return redirect(url_for("home"))
    try:
        updates = int(load_bandit_state().get("updates", 0))
        recent_rows: tuple[int, ...] = ()
        try:
            recent_rows = recent_history_rows(client)
        except Exception:
            recent_rows = ()
        mix = next(
            (item for item in catalog_mood_mixes(updates, recent_rows) if item["mood"] == mood),
            None,
        )
        if not mix:
            flash("The model could not build that mix.", "error")
            return redirect(url_for("home"))
        title = f"TuneTwin {mix['name']}"
        uris = []
        missing = 0
        for track in mix["tracks"]:
            match = spotify_match(int(track["index"]))
            if match and match.get("uri") and match["uri"] not in uris:
                uris.append(match["uri"])
            else:
                missing += 1
        if not uris:
            flash("Spotify could not match tracks for this mix.", "error")
            return redirect(url_for("home"))
        playlist = find_or_create_private_playlist(client, title, mix["description"])
        client._put(f"playlists/{playlist['id']}/items", payload={"uris": uris})
        url = (playlist.get("external_urls") or {}).get("spotify", "")
        flash(f"Exported {title} with {len(uris)} tracks. {url}", "success")
        if missing:
            flash(f"Skipped {missing} tracks without a confident Spotify match.", "error")
        return redirect(url_for("home"))
    except Exception as exc:
        flash(f"Couldn't create the Spotify playlist ({type(exc).__name__}). Reconnect Spotify and approve private playlist access.", "error")
        return redirect(url_for("home"))


@app.post("/api/spotify/control/<action>")
def spotify_control(action):
    client, error = spotify_client_or_error()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    try:
        if action == "play": client.start_playback(device_id=device_id)
        elif action == "pause": client.pause_playback(device_id=device_id)
        elif action == "next": client.next_track(device_id=device_id)
        elif action == "previous": client.previous_track(device_id=device_id)
        elif action == "seek": client.seek_track(max(0, int(data.get("position_ms", 0))), device_id=device_id)
        elif action == "volume": client.volume(max(0, min(100, int(data.get("volume", 50)))), device_id=device_id)
        elif action == "transfer":
            target = data.get("device_id")
            if not target: return {"error": "Choose a device."}, 400
            client.transfer_playback(device_id=target, force_play=False)
        elif action == "shuffle": client.shuffle(bool(data.get("enabled")), device_id=device_id)
        elif action == "repeat": client.repeat(data.get("state", "off"), device_id=device_id)
        else: return {"error": "Unknown playback action."}, 404
        return {"ok": True}
    except Exception as exc:
        return {"error": f"Spotify control failed ({type(exc).__name__}). Check Premium, device status, and permissions."}, 502


@app.post("/api/spotify/player-feedback")
def spotify_player_feedback():
    client, error = spotify_client_or_error()
    if error:
        return error
    feedback = (request.get_json(silent=True) or {}).get("feedback")
    if feedback not in {"like", "skip"}:
        return {"error": "Choose Like or Skip."}, 400
    try:
        playback = client.current_playback() or {}
        item = playback.get("item") or {}
        if not item.get("uri"):
            return {"error": "There is no current Spotify song to rate."}, 409
    except Exception as exc:
        return {"error": f"Could not read the current Spotify song ({type(exc).__name__})."}, 502

    # Save learning first and report it independently from Spotify playback or
    # playlist operations. Spotify may reject next_track when the active device
    # is unavailable, but that must not roll back the user's explicit feedback.
    updates = None
    track_idx = catalog_index_for_spotify_track(item)
    try:
        if track_idx is not None:
            updates, _ = update_bandit(track_idx, feedback)
    except OSError:
        return {"error": "Could not save adaptive learning. Check that the models folder is writable."}, 500

    message = "Liked this song" if feedback == "like" else "Skip feedback saved"
    if updates is not None:
        message += " · adaptive learning updated"
    else:
        message += " · song not found in the local dataset, so learning was unchanged"
    result = {"ok": True, "feedback": feedback, "message": message,
              "learning_updated": updates is not None, "updates": updates}

    if feedback == "like":
        try:
            playlist, newly_liked = add_liked_track_to_playlist(client, item["uri"])
            result["playlist_url"] = (playlist.get("external_urls") or {}).get("spotify")
            result["message"] += " and saved in TuneTwin Liked Songs" if newly_liked else " · already in TuneTwin Liked Songs"
        except Exception as exc:
            result["message"] += f" · learning saved; reconnect Spotify to allow private playlist access ({type(exc).__name__})"
    else:
        try:
            client.next_track(device_id=(playback.get("device") or {}).get("id"))
            result["advanced"] = True
            result["message"] = result["message"].replace("Skip feedback saved", "Skipped this song")
        except Exception as exc:
            result["advanced"] = False
            result["message"] += f" · Spotify could not advance ({type(exc).__name__}); use an active device"
    return result


@app.get("/api/learning")
def learning_stats():
    state = load_bandit_state()
    events = list(reversed(state.get("events", [])[-20:]))
    return {"updates": state["updates"], "likes": sum(e.get("feedback") == "like" for e in state.get("events", [])), "skips": sum(e.get("feedback") == "skip" for e in state.get("events", [])), "recent": events}


@app.post("/api/learning/reset")
def reset_learning():
    save_bandit_state({"A": np.eye(MODEL_CONTEXTS.shape[1]), "b": np.zeros(MODEL_CONTEXTS.shape[1]), "updates": 0, "events": []})
    return {"ok": True, "updates": 0}


@app.post("/recommend")
def recommend_route():
    seed_idx = request.form.get("track_index", type=int)
    if seed_idx is None or not 0 <= seed_idx < len(TRACKS):
        flash("Choose a song from the search results first.", "error")
        return redirect(url_for("home"))
    return redirect(url_for("home", seed=seed_idx))


@app.get("/spotify/connect")
def spotify_connect():
    try:
        # Force Spotify to show the consent screen so a previously cached
        # grant is replaced with the Web Playback SDK scopes above.
        (ROOT / ".spotify_token_cache").unlink(missing_ok=True)
        state = secrets.token_urlsafe(24)
        session["spotify_oauth_state"] = state
        return redirect(get_user_oauth(state=state).get_authorize_url())
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("home"))


@app.get("/callback")
def spotify_callback():
    expected = session.pop("spotify_oauth_state", None)
    returned = request.args.get("state")
    if not expected or not returned or not secrets.compare_digest(expected, returned):
        flash("Spotify authorization could not be verified. Please connect again.", "error")
        return redirect(url_for("home"))
    if request.args.get("error"):
        flash("Spotify authorization was cancelled or denied.", "error")
        return redirect(url_for("home"))
    code = request.args.get("code")
    if not code:
        flash("Spotify did not return an authorization code.", "error")
        return redirect(url_for("home"))
    try:
        get_user_oauth().get_access_token(code, as_dict=True, check_cache=False)
        flash("Spotify connected. Choose a song and press Play when you are ready.", "success")
    except Exception as exc:
        flash(f"Spotify authorization failed: {type(exc).__name__}. Check the redirect URI and app access.", "error")
    return redirect(url_for("home"))


def catalog_queue_entry(index: int) -> dict | None:
    match = spotify_match(index)
    if not match or not match.get("uri"):
        return None
    row = TRACKS.iloc[index]
    return {
        "index": index,
        "name": row["_name"],
        "artist": row["_artist_display"],
        "uri": match["uri"],
        "art": match.get("image"),
    }


AUTO_QUEUE_SIZE = 8  # upcoming recommendations kept after the current song


def build_queue_entries(seed_idx: int, include_recommendations: bool = True) -> list[dict]:
    """Build a queue without repeating the seed or alternate copies of a song."""
    entries: list[dict] = []
    seen_uris: set[str] = set()
    seen_songs: set[tuple[str, str]] = set()

    def add(index: int) -> None:
        entry = catalog_queue_entry(index)
        if not entry or entry["uri"] in seen_uris:
            return
        song_key = SONG_KEYS[index]
        if song_key in seen_songs:
            return
        seen_uris.add(entry["uri"])
        seen_songs.add(song_key)
        entries.append(entry)

    add(seed_idx)
    if include_recommendations:
        # Fetch a wider pool so alternate copies filtered below never leave the
        # queue short of its eight recommendation slots.
        for rec in recommend(seed_idx, limit=24):
            add(rec["index"])
            if len(entries) > AUTO_QUEUE_SIZE:
                break
    return entries


def resolve_playlist_id(value: str) -> str | None:
    """Accept a playlist URL, URI, or bare Spotify ID."""
    match = re.search(r"playlist[:/]([A-Za-z0-9]{22})", str(value))
    if match:
        return match.group(1)
    candidate = str(value).strip()
    return candidate if re.fullmatch(r"[A-Za-z0-9]{22}", candidate) else None


PLAYLIST_CACHE_TTL = 300  # seconds
_PLAYLIST_PAYLOAD_CACHE: dict[str, tuple[float, dict]] = {}
_PLAYLIST_REC_CACHE: dict[tuple[str, str, int], tuple[float, list[dict]]] = {}


def invalidate_playlist_cache(playlist_id: str) -> None:
    _PLAYLIST_PAYLOAD_CACHE.pop(playlist_id, None)
    for key in [key for key in _PLAYLIST_REC_CACHE if key[0] == playlist_id]:
        _PLAYLIST_REC_CACHE.pop(key, None)


def load_playlist_payload(client, playlist_id: str, max_tracks: int = 200) -> dict:
    """Fetch a playlist once and normalise the old and new Spotify shapes."""
    cached = _PLAYLIST_PAYLOAD_CACHE.get(playlist_id)
    if cached and time.time() - cached[0] < PLAYLIST_CACHE_TTL:
        return cached[1]

    playlist = client.playlist(playlist_id)
    page = client.playlist_items(playlist_id, limit=100, additional_types=("track",))
    total_tracks = page.get("total") if page else 0
    tracks: list[dict] = []
    seed_rows: list[int] = []
    while page:
        for entry in page.get("items", []) or []:
            item = entry.get("track") or entry.get("item") or {}
            if not item or not item.get("uri") or item.get("type") == "episode":
                continue
            images = (item.get("album") or {}).get("images") or []
            idx = catalog_index_for_spotify_track(item)
            if idx is not None:
                seed_rows.append(idx)
            tracks.append({
                "index": idx,
                "name": item.get("name") or "Unknown track",
                "artist": ", ".join(artist.get("name", "") for artist in item.get("artists", [])),
                "uri": item["uri"],
                "art": images[0]["url"] if images else None,
            })
            if len(tracks) >= max_tracks:
                break
        if len(tracks) >= max_tracks or not page.get("next"):
            break
        page = client.next(page)

    images = playlist.get("images") or []
    payload = {
        "playlist_id": playlist_id,
        "name": playlist.get("name") or "Imported playlist",
        "description": playlist.get("description") or "",
        "snapshot_id": playlist.get("snapshot_id") or "",
        "image": images[0]["url"] if images else None,
        "total_tracks": int(total_tracks or len(tracks)),
        "tracks": tracks,
        "seed_rows": seed_rows,
    }
    _PLAYLIST_PAYLOAD_CACHE[playlist_id] = (time.time(), payload)
    return payload


def cached_playlist_recommendations(payload: dict, limit: int = 20) -> list[dict]:
    """Cache the expensive model pass per playlist version and learning state."""
    if not payload["seed_rows"]:
        return []
    updates = int(load_bandit_state().get("updates", 0))
    key = (payload["playlist_id"], payload["snapshot_id"], updates)
    cached = _PLAYLIST_REC_CACHE.get(key)
    if cached and time.time() - cached[0] < PLAYLIST_CACHE_TTL:
        return cached[1]
    recommendations = playlist_recommendations(payload["seed_rows"], limit=limit)
    _PLAYLIST_REC_CACHE[key] = (time.time(), recommendations)
    return recommendations


def playlist_recommendations(seed_rows: list[int], limit: int = 20) -> list[dict]:
    """Blend recommender output from several playlist seeds into one list."""
    seeds = list(dict.fromkeys(int(row) for row in seed_rows))[:5]
    playlist_songs = {SONG_KEYS[idx] for idx in seed_rows}

    def seed_recommendations(seed: int) -> list[int]:
        return [rec["index"] for rec in recommend(seed, limit=8, with_art=False)]

    # Each recommend() pass is independent and read-only; running the seeds in
    # parallel keeps the first playlist open from feeling stuck.
    with ThreadPoolExecutor(max_workers=min(4, len(seeds) or 1)) as pool:
        rec_lists = list(pool.map(seed_recommendations, seeds))
    ordered: list[int] = []
    seen: set[int] = set()
    artist_counts: dict[str, int] = {}
    for slot in range(max((len(recs) for recs in rec_lists), default=0)):
        for recs in rec_lists:
            if slot >= len(recs):
                continue
            idx = int(recs[slot])
            if idx in seen or idx in seeds or SONG_KEYS[idx] in playlist_songs:
                continue
            artist = PRIMARY_ARTIST_KEYS[idx]
            if artist_counts.get(artist, 0) >= 2:
                continue
            seen.add(idx)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            ordered.append(idx)
            if len(ordered) == limit:
                break
        if len(ordered) == limit:
            break
    results = []
    for idx in ordered:
        entry = catalog_queue_entry(idx)
        if entry:
            results.append(entry)
    return results


@app.post("/api/play")
def api_play():
    """Play a catalogue row (plus its recommendation context) as an app-managed queue.

    Uses a Spotify *context* (uris + offset) instead of add-to-queue, so the app's
    queue is its own, can be edited (removed), and never pollutes Spotify's queue.
    """
    data = request.get_json(silent=True) or {}
    if not spotify_connected():
        return {"error": "Connect Spotify first."}, 401
    try:
        client = get_user_client()
        devices = client.devices().get("devices", [])
        if not devices:
            return {"error": "No Spotify device is active. Open Spotify on a device."}, 409
        requested = data.get("device_id")
        device = next((d for d in devices if d.get("id") == requested), None) if requested else None
        if device is None:
            # The Web Playback SDK can silently pause after a few seconds on
            # some browsers/accounts. Prefer a normal Spotify Connect device;
            # users can still explicitly select another device in the UI.
            real_devices = [d for d in devices if d.get("name") != "TuneTwin browser player"]
            active_device = next((d for d in real_devices if d.get("is_active")), None)
            device = active_device or (real_devices[0] if real_devices else None) or next((d for d in devices if d.get("is_active")), None) or devices[0]
        if not device.get("is_active"):
            try:
                client.transfer_playback(device_id=device["id"], force_play=False)
            except Exception:
                pass

        # Re-play a queue the client edited (remove / reorder).
        uris = data.get("uris")
        if isinstance(uris, list) and uris:
            uris = [u for u in uris if isinstance(u, str) and u.startswith("spotify:track:")]
            if not uris:
                return {"error": "Nothing to play."}, 400
            play_args = {"device_id": device["id"], "uris": uris}
            offset = data.get("offset")
            if isinstance(offset, int) and 0 <= offset < len(uris):
                play_args["offset"] = {"position": offset}
            position_ms = data.get("position_ms")
            if isinstance(position_ms, int) and position_ms > 0:
                play_args["position_ms"] = position_ms
            client.start_playback(**play_args)
            return {"ok": True, "message": f"Playing on {device.get('name', 'your Spotify device')}."}

        try:
            row_idx = int(data.get("row_index"))
        except (TypeError, ValueError):
            row_idx = None
        if row_idx is None or not 0 <= row_idx < len(TRACKS):
            return {"error": "That song is not in the local catalog."}, 400

        try:
            seed_idx = int(data.get("seed_index"))
        except (TypeError, ValueError):
            seed_idx = row_idx
        if seed_idx is None or not 0 <= seed_idx < len(TRACKS):
            seed_idx = row_idx
        include_recs = bool(data.get("include_recommendations"))
        entries = build_queue_entries(
            seed_idx if include_recs else row_idx,
            include_recommendations=include_recs,
        )

        if not entries:
            return {"error": "Spotify could not match this song."}, 404

        offset = next((i for i, entry in enumerate(entries) if entry["index"] == row_idx), 0)
        play_args = {"device_id": device["id"], "uris": [entry["uri"] for entry in entries]}
        if offset:
            play_args["offset"] = {"position": offset}
        position_ms = data.get("position_ms")
        if isinstance(position_ms, int) and position_ms > 0:
            play_args["position_ms"] = position_ms
        client.start_playback(**play_args)
        return {
            "ok": True,
            "message": f"Playing on {device.get('name', 'your Spotify device')}.",
            "queue": entries,
            "offset": offset,
        }
    except Exception as exc:
        return {"error": f"Spotify could not start playback ({type(exc).__name__})."}, 502


def queue_track_key(name: str, artist: str) -> str:
    return f"{normalize_song_title(str(name or ''))}|{normalize_artist_name(str(artist or ''))}"


@app.post("/api/queue/refill")
def queue_refill():
    """Return fresh recommendations, excluding everything already queued.

    The client asks for one replacement each time a song ends so the queue keeps
    exactly eight recommendation slots ahead of the current track.
    """
    client, error = spotify_client_or_error()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    try:
        count = max(1, min(int(data.get("count") or 1), AUTO_QUEUE_SIZE))
    except (TypeError, ValueError):
        count = 1

    seed_idx = data.get("seed_index")
    if not isinstance(seed_idx, int) or not 0 <= seed_idx < len(TRACKS):
        try:
            playback = client.current_playback() or {}
            seed_idx = catalog_index_for_spotify_track(playback.get("item") or {})
        except Exception:
            seed_idx = None
    if seed_idx is None:
        return {"entries": []}

    exclude_uris = {str(uri) for uri in (data.get("uris") or []) if isinstance(uri, str)}
    exclude_keys = {str(key) for key in (data.get("keys") or []) if isinstance(key, str)}
    entries: list[dict] = []
    try:
        for rec in recommend(seed_idx, limit=60):
            idx = int(rec["index"])
            if idx == seed_idx:
                continue
            entry = catalog_queue_entry(idx)
            if not entry:
                continue
            key = queue_track_key(entry["name"], entry["artist"])
            if entry["uri"] in exclude_uris or key in exclude_keys:
                continue
            exclude_uris.add(entry["uri"])
            exclude_keys.add(key)
            entries.append(entry)
            if len(entries) >= count:
                break
    except Exception as exc:
        return {"error": f"Could not build the next recommendation ({type(exc).__name__})."}, 502
    return {"entries": entries}


@app.post("/api/catalog/art")
def catalog_art():
    """Resolve cover art for catalogue rows that arrived without an image.

    Mix queues and older cached queue entries carry only a catalogue index, so
    the client asks for those covers lazily instead of blocking queue creation.
    """
    data = request.get_json(silent=True) or {}
    raw = data.get("indices") or []
    indices: list[int] = []
    for value in raw[:60]:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(TRACKS) and index not in indices:
            indices.append(index)
    if not indices:
        return {"art": {}}

    def resolve(index: int) -> tuple[int, str | None]:
        spotify_id = spotify_id_for(index)
        return index, spotify_art_for(spotify_id) if spotify_id else None

    art: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(indices))) as pool:
        for index, image in pool.map(resolve, indices):
            if image:
                art[str(index)] = image
    return {"art": art}


@app.post("/api/queue/current")
def queue_current():
    """Return the current song plus recommendations without starting playback."""
    client, error = spotify_client_or_error()
    if error:
        return error
    try:
        playback = client.current_playback() or {}
        item = playback.get("item") or {}
        if not item.get("uri"):
            return {"entries": []}
        album = item.get("album") or {}
        images = album.get("images") or []
        seed_idx = catalog_index_for_spotify_track(item)
        current = {
            "index": seed_idx,
            "name": item.get("name") or "Unknown track",
            "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])),
            "uri": item["uri"],
            "art": images[0].get("url") if images else None,
        }
        if seed_idx is None:
            return {"entries": [current]}
        entries = build_queue_entries(seed_idx, include_recommendations=True)
        entries = [current, *[entry for entry in entries if entry["uri"] != current["uri"]]]
        return {"entries": entries}
    except Exception as exc:
        return {"error": f"Could not refresh the queue ({type(exc).__name__})."}, 502


@app.post("/play")
def play_track():
    row_idx = request.form.get("row_index", type=int)
    seed_idx = request.form.get("seed_index", type=int)
    if row_idx is None or not 0 <= row_idx < len(TRACKS):
        flash("That song is not in the local catalog.", "error")
        return redirect(url_for("home"))
    if not spotify_connected():
        flash("Connect Spotify first, then press Play again.", "error")
        return redirect(url_for("home", seed=seed_idx) if seed_idx is not None else url_for("home"))
    try:
        row = TRACKS.iloc[row_idx]
        match = spotify_match(row_idx)
        if not match:
            flash("Spotify did not return a confident match for this song; playback was not started.", "error")
        else:
            client = get_user_client()
            devices = client.devices().get("devices", [])
            if not devices:
                flash("No Spotify Connect device is active. Open Spotify on a device and try again (playback requires Premium).", "error")
            else:
                requested_device = request.form.get("device_id")
                device = next((d for d in devices if d.get("id") == requested_device), None) if requested_device else None
                if device is None:
                    browser_device = next((d for d in devices if d.get("name") == "TuneTwin browser player"), None)
                    active_device = next((d for d in devices if d.get("is_active")), None)
                    real_device = next((d for d in devices if d.get("name") != "TuneTwin browser player"), None)
                    device = browser_device or active_device or real_device or devices[0]
                if not device.get("is_active"):
                    try:
                        client.transfer_playback(device_id=device["id"], force_play=False)
                    except Exception:
                        pass
                client.start_playback(device_id=device["id"], uris=[match["uri"]])
                queue_rows = []
                if seed_idx is not None and 0 <= seed_idx < len(TRACKS):
                    suggested = recommend(seed_idx)
                    suggested_indices = [item["index"] for item in suggested]
                    if row_idx == seed_idx:
                        queue_rows = suggested_indices
                    elif row_idx in suggested_indices:
                        queue_rows = suggested_indices[suggested_indices.index(row_idx) + 1:]
                    else:
                        queue_rows = suggested_indices
                queued = 0
                already_queued = 0
                missing = 0
                failed = 0
                # Spotify does not offer a queue-clear endpoint. Avoid adding
                # TuneTwin tracks that are already in the current queue, and
                # de-duplicate Spotify URI matches within this enqueue pass.
                queued_uris = set()
                try:
                    current_queue = client.queue()
                    current_item = current_queue.get("currently_playing") or {}
                    if current_item.get("uri"):
                        queued_uris.add(current_item["uri"])
                    queued_uris.update(
                        item.get("uri") for item in current_queue.get("queue", []) if item.get("uri")
                    )
                except Exception:
                    pass
                # Add sequentially, after starting the selected song. Spotify appends
                # items to its existing queue; it does not clear older queue entries.
                for queue_idx in queue_rows:
                    queue_match = spotify_match(queue_idx)
                    if not queue_match or not queue_match.get("uri"):
                        missing += 1
                        continue
                    if queue_match["uri"] in queued_uris:
                        already_queued += 1
                        continue
                    try:
                        client.add_to_queue(queue_match["uri"], device_id=device["id"])
                        queued += 1
                        queued_uris.add(queue_match["uri"])
                    except Exception:
                        failed += 1
                if queue_rows:
                    extra = f" Added {queued} following recommendations."
                    if already_queued:
                        extra += f" Skipped {already_queued} already in your Spotify queue."
                    extra += " Spotify appends songs after existing queue items."
                    if missing or failed:
                        extra += f" {missing} had no confident match; {failed} queue requests failed."
                else:
                    extra = " No further recommendations remained to queue."
                flash(f"Playing {match['name']} by {match['artist']} on {device.get('name', 'your Spotify device')}.{extra}", "success")
    except Exception as exc:
        flash(f"Spotify could not start playback ({type(exc).__name__}). Check Premium, device status, and app permissions.", "error")
    return redirect(url_for("home", seed=seed_idx) if seed_idx is not None else url_for("home"))


@app.post("/feedback")
def record_feedback():
    row_idx = request.form.get("row_index", type=int)
    seed_idx = request.form.get("seed_index", type=int)
    feedback = request.form.get("feedback", "")
    if row_idx is None or not 0 <= row_idx < len(TRACKS):
        flash("That recommendation is not in the local catalog.", "error")
        return redirect(url_for("home"))
    if feedback not in {"like", "skip"}:
        flash("Choose Like or Skip to give the model feedback.", "error")
        return redirect(url_for("home", seed=seed_idx) if seed_idx is not None else url_for("home"))
    try:
        updates, reward = update_bandit(row_idx, feedback)
    except OSError:
        flash("Could not save local LinUCB feedback. Check that the models folder is writable.", "error")
        return redirect(url_for("home", seed=seed_idx) if seed_idx is not None else url_for("home"))

    advanced = False
    liked_playlist_added = False
    liked_playlist_url = None
    if feedback == "like" and spotify_connected():
        try:
            match = spotify_match(row_idx)
            if match and match.get("uri"):
                playlist, liked_playlist_added = add_liked_track_to_playlist(get_user_client(), match["uri"])
                liked_playlist_url = (playlist.get("external_urls") or {}).get("spotify")
        except Exception:
            pass
    if feedback == "skip" and spotify_connected():
        try:
            client = get_user_client()
            current = client.current_playback()
            current_item = (current or {}).get("item") or {}
            current_uri = current_item.get("uri")
            matching_track = spotify_match(row_idx)
            if current_uri and matching_track and current_uri == matching_track.get("uri"):
                client.next_track()
                advanced = True
        except Exception:
            pass
    action = "Like" if feedback == "like" else "Skip"
    message = f"{action} feedback saved for {TRACKS.iloc[row_idx]['_name']} ({reward:+.0f} reward). LinUCB has learned from {updates} feedback events."
    feedback_labels = {event.get("feedback") for event in load_bandit_state().get("events", [])}
    if {"like", "skip"}.issubset(feedback_labels):
        message += " The KNN preference model can now learn from both feedback types."
    else:
        message += " Add at least one Like and one Skip to activate the KNN preference model."
    if advanced:
        message += " Spotify advanced to the next queued track."
    elif feedback == "like":
        message += " This updates TuneTwin only; it does not add the song to Spotify Liked Songs."
    if feedback == "like" and liked_playlist_added:
        message += f" Added to TuneTwin Liked Songs: {liked_playlist_url}"
    elif feedback == "like" and spotify_connected():
        message += " The TuneTwin Liked Songs playlist already has this track, or Spotify could not match it."
    flash(message, "success")
    return redirect(url_for("home", seed=seed_idx) if seed_idx is not None else url_for("home"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracks": len(TRACKS),
        "features": AUDIO_FEATURES,
        "pca_components": PCA_VECTORS.shape[1],
        "pca_explained_variance": float(PCA_MODEL.explained_variance_ratio_.sum()),
        "kmeans_clusters": SELECTED_K,
        "kmeans_silhouette": KMEANS_SILHOUETTES.get(SELECTED_K),
        "recommender_artifact": RECOMMENDER_ARTIFACT is not None,
        "classifier": (RECOMMENDER_ARTIFACT or {}).get("classifier_name", "Online KNN fallback"),
        "recommendation_methods": ["PCA", "K-Means", "PCA KNN", "Tuned classifier", "LinUCB"],
    }


if __name__ == "__main__":
    print(f"TuneTwin AI loaded {len(TRACKS):,} tracks using {len(AUDIO_FEATURES)} audio features.")
    print(f"Spotify callback must exactly match the developer dashboard URI: {REDIRECT_URI}")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8888")), debug=False)
