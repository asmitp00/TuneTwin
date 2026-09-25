"""Local Flask app for TuneTwin AI's genre-aware Spotify recommender."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import secrets
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_PATH = Path(os.getenv("TUNETWIN_DATA_PATH", ROOT / "data" / "data.csv"))
GENRE_PATH = Path(os.getenv("TUNETWIN_GENRE_PATH", ROOT / "data" / "extracted" / "data" / "data_w_genres.csv"))
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SCOPES = "user-read-playback-state user-modify-playback-state"
LINUCB_PATH = ROOT / "models" / "linucb_live.json"
LINUCB_ALPHA = 0.20
LINUCB_BLEND = 0.25
FEATURE_CANDIDATES = [
    "valence", "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "liveness", "loudness", "speechiness", "tempo",
]

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_urlsafe(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
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


def load_catalog() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    tracks = pd.read_csv(DATA_PATH, low_memory=False)
    name_col = next((c for c in ("name", "track_name", "track") if c in tracks.columns), None)
    artist_col = next((c for c in ("artists", "artist", "artist_name") if c in tracks.columns), None)
    if not name_col or not artist_col:
        raise ValueError("The track CSV needs a track name column and an artist column.")
    feature_cols = [c for c in FEATURE_CANDIDATES if c in tracks.columns]
    if not feature_cols:
        raise ValueError("The track CSV has none of the supported audio feature columns.")

    tracks = tracks.dropna(subset=[name_col]).drop_duplicates().reset_index(drop=True)
    tracks["_name_col"] = name_col
    tracks["_artist_col"] = artist_col
    tracks["_name"] = tracks[name_col].astype(str).str.strip()
    tracks["_artists"] = tracks[artist_col].map(parse_list)
    tracks["_artist_display"] = tracks["_artists"].map(lambda xs: ", ".join(xs) if xs else "Unknown artist")
    tracks["_artist_keys"] = tracks["_artists"].map(lambda xs: {x.casefold().strip() for x in xs if x.strip()})

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
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    scaled = StandardScaler().fit_transform(numeric)
    return tracks, feature_cols, normalize(scaled, norm="l2").astype(np.float32)


TRACKS, AUDIO_FEATURES, AUDIO_VECTORS = load_catalog()
NAME_COL = TRACKS["_name_col"].iloc[0]
ARTIST_COL = TRACKS["_artist_col"].iloc[0]
PRIMARY_ARTISTS = TRACKS["_artists"].map(lambda artists: artists[0].casefold().strip() if artists else "")
GENRE_ENCODER = MultiLabelBinarizer(sparse_output=True)
GENRE_MATRIX = GENRE_ENCODER.fit_transform(TRACKS["_genres"]).tocsr()
GENRE_COUNTS = np.asarray(GENRE_MATRIX.sum(axis=1)).ravel()
GENRE_INDEX = {name: idx for idx, name in enumerate(GENRE_ENCODER.classes_)}
ARTIST_ROWS: dict[str, list[int]] = {}
for _row_idx, _artist_keys in enumerate(TRACKS["_artist_keys"]):
    for _artist_key in _artist_keys:
        ARTIST_ROWS.setdefault(_artist_key, []).append(_row_idx)


def load_bandit_state() -> dict:
    dimension = len(AUDIO_FEATURES)
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
    context = AUDIO_VECTORS[track_idx].astype(float)
    reward = rewards[feedback]
    state = load_bandit_state()
    state["A"] += np.outer(context, context)
    state["b"] += reward * context
    state["updates"] += 1
    state["events"].append({
        "track": TRACKS.iloc[track_idx]["_name"],
        "artist": TRACKS.iloc[track_idx]["_artist_display"],
        "feedback": feedback,
        "reward": reward,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    save_bandit_state(state)
    return state["updates"], reward


def artist_main(row: pd.Series) -> str:
    artists = row["_artists"]
    return artists[0] if artists else ""


def get_user_oauth(state: str | None = None) -> SpotifyOAuth:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to the project .env file.")
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        state=state,
        cache_path=str(ROOT / ".spotify_token_cache"),
        open_browser=False,
    )


def get_user_client():
    return spotipy.Spotify(auth_manager=get_user_oauth())


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
        return get_user_oauth().cache_handler.get_cached_token() is not None
    except Exception:
        return False


@lru_cache(maxsize=4096)
def spotify_match(track_index: int) -> dict | None:
    """Find a catalog match using Spotify search, never trusting the CSV ID."""
    client = get_catalog_client()
    if client is None:
        return None
    row = TRACKS.iloc[int(track_index)]
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
            best = {
                "id": item.get("id"),
                "uri": item.get("uri"),
                "name": candidate_title,
                "artist": candidate_artist,
                "url": item.get("external_urls", {}).get("spotify"),
                "confidence": confidence,
            }
    return best if best and best["confidence"] >= 0.55 else None


def recommend(seed_idx: int, limit: int = 5) -> list[dict]:
    seed = TRACKS.iloc[seed_idx]
    audio_scores = np.clip((AUDIO_VECTORS @ AUDIO_VECTORS[seed_idx] + 1.0) / 2.0, 0.0, 1.0)
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
    weights = (0.60, 0.25 if seed_genres else 0.0, 0.15 if seed_artists else 0.0)
    weight_sum = sum(weights)
    content_scores = (weights[0] * audio_scores + weights[1] * genre_scores + weights[2] * artist_scores) / weight_sum
    content_scores[seed_idx] = -1

    # Build a Top-50 content pool from a wider nearest-neighbor shortlist. The
    # pool starts with distinct lead artists, then may trade a weak slot for a
    # near-cutoff repeat (maximum two per lead artist).
    content_rank = np.argsort(content_scores)[::-1]
    source_pool = content_rank[:min(5000, len(TRACKS) - 1)]
    pool_list: list[int] = []
    pool_artist_counts: dict[str, int] = {}
    for raw_idx in source_pool:
        idx = int(raw_idx)
        key = PRIMARY_ARTISTS.iloc[idx]
        if pool_artist_counts.get(key, 0) != 0:
            continue
        pool_list.append(idx)
        pool_artist_counts[key] = 1
        if len(pool_list) == 50:
            break
    if len(pool_list) < min(50, len(TRACKS) - 1):
        for raw_idx in source_pool:
            idx = int(raw_idx)
            key = PRIMARY_ARTISTS.iloc[idx]
            if pool_artist_counts.get(key, 0) >= 2:
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
            key = PRIMARY_ARTISTS.iloc[idx]
            if pool_artist_counts.get(key, 0) >= 2:
                continue
            if content_scores[idx] < pool_cutoff - 0.04:
                break
            replace_at = min(
                (pos for pos, selected_idx in enumerate(pool_list) if PRIMARY_ARTISTS.iloc[selected_idx] != key),
                key=lambda pos: content_scores[pool_list[pos]],
                default=None,
            )
            if replace_at is None:
                continue
            removed_idx = pool_list[replace_at]
            if content_scores[idx] < content_scores[removed_idx] - 0.04:
                break
            removed_key = PRIMARY_ARTISTS.iloc[removed_idx]
            pool_artist_counts[removed_key] -= 1
            pool_list[replace_at] = idx
            pool_artist_counts[key] = pool_artist_counts.get(key, 0) + 1
    candidate_pool = np.asarray(pool_list, dtype=int)

    # LinUCB reranks this diversified Top-50 pool using explicit user feedback.
    state = load_bandit_state()
    inverse = np.linalg.inv(state["A"])
    theta = inverse @ state["b"]
    contexts = AUDIO_VECTORS[candidate_pool].astype(float)
    predicted_reward = contexts @ theta
    uncertainty = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", contexts, inverse, contexts)))
    ucb = predicted_reward + LINUCB_ALPHA * uncertainty
    bandit_scores = np.clip((np.tanh(ucb) + 1.0) / 2.0, 0.0, 1.0)
    ranking_scores = content_scores.copy()
    ranking_scores[candidate_pool] = (1.0 - LINUCB_BLEND) * content_scores[candidate_pool] + LINUCB_BLEND * bandit_scores
    ranking_scores[seed_idx] = -1
    order = np.lexsort((candidate_pool, -ranking_scores[candidate_pool]))
    ranked = candidate_pool[order]
    selected: list[int] = []
    artist_counts: dict[str, int] = {}
    # First pass favors discovery: fill the list with one track per artist.
    for raw_idx in ranked:
        idx = int(raw_idx)
        keys = TRACKS.iloc[idx]["_artist_keys"]
        if any(artist_counts.get(key, 0) >= 1 for key in keys):
            continue
        selected.append(idx)
        for key in keys:
            artist_counts[key] = artist_counts.get(key, 0) + 1
        if len(selected) == limit:
            break

    # A repeat can replace the weakest unique pick only when it scores at least
    # as highly as the current cutoff. Keep a hard cap of two per artist.
    repeat_margin = 0.0
    if len(selected) == limit and limit >= 4:
        for raw_idx in ranked:
            idx = int(raw_idx)
            if idx in selected:
                continue
            keys = TRACKS.iloc[idx]["_artist_keys"]
            if not any(artist_counts.get(key, 0) >= 1 for key in keys):
                continue
            if any(artist_counts.get(key, 0) >= 2 for key in keys):
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
            primary_artist_count = len({artist_main(TRACKS.iloc[item]).casefold() for item in trial})
            if primary_artist_count < min(limit - 1, len(selected)):
                continue
            for key in removed_keys:
                artist_counts[key] -= 1
            if any(artist_counts.get(key, 0) >= 2 for key in keys):
                for key in removed_keys:
                    artist_counts[key] += 1
                continue
            selected[weakest_position] = idx
            for key in keys:
                artist_counts[key] = artist_counts.get(key, 0) + 1
            selected.sort(key=lambda idx: ranking_scores[idx], reverse=True)

    results = []
    for idx in selected:
        row = TRACKS.iloc[idx]
        match = spotify_match(idx)
        popularity = row.get("popularity")
        results.append({
            "index": idx,
            "name": row["_name"],
            "artist": row["_artist_display"],
            "genres": sorted(row["_genres"])[:5],
            "score": float(ranking_scores[idx]),
            "similarity": float(content_scores[idx]),
            "bandit_score": float(bandit_scores[np.where(candidate_pool == idx)[0][0]]),
            "match": match,
            "popularity": int(popularity) if pd.notna(popularity) else None,
        })
    return results


def csrf_value() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


@app.before_request
def check_csrf():
    if request.method == "POST":
        expected = session.get("_csrf", "")
        submitted = request.form.get("csrf_token", "")
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
                device = next((d for d in devices if d.get("is_active")), devices[0])
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
                missing = 0
                failed = 0
                # Add sequentially, after starting the selected song. Spotify appends
                # items to its existing queue; it does not clear older queue entries.
                for queue_idx in queue_rows:
                    queue_match = spotify_match(queue_idx)
                    if not queue_match or not queue_match.get("uri"):
                        missing += 1
                        continue
                    try:
                        client.add_to_queue(queue_match["uri"], device_id=device["id"])
                        queued += 1
                    except Exception:
                        failed += 1
                if queue_rows:
                    extra = f" Queued {queued} of {len(queue_rows)} following recommendations at the end of Spotify's current queue."
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
    if advanced:
        message += " Spotify advanced to the next queued track."
    elif feedback == "like":
        message += " This updates TuneTwin only; it does not add the song to Spotify Liked Songs."
    flash(message, "success")
    return redirect(url_for("home", seed=seed_idx) if seed_idx is not None else url_for("home"))


@app.get("/health")
def health():
    return {"status": "ok", "tracks": len(TRACKS), "features": AUDIO_FEATURES}


if __name__ == "__main__":
    print(f"TuneTwin AI loaded {len(TRACKS):,} tracks using {len(AUDIO_FEATURES)} audio features.")
    print(f"Spotify callback must exactly match the developer dashboard URI: {REDIRECT_URI}")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8888")), debug=False)
