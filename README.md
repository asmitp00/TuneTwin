# TuneTwin AI — Flask Demo

TuneTwin searches the provided Spotify track dataset and ranks eight recommendations with a combined ML pipeline. It standardizes audio features, applies PCA to build the reduced music representation, selects a K-Means cluster count using sampled silhouette scores, and finds seed-song neighbors in PCA space with KNN. Genre and artist signals add context. With at least one Like and one Skip, a KNN classifier learns from explicit feedback; LinUCB also reranks the Top-50 candidate pool. The app can play a selected seed or recommendation through Spotify Connect. It favors distinct artists but may include up to three tracks from one artist when the scores support them, while keeping at least six distinct lead artists in an eight-track list when the catalog permits.

## Run locally

1. Keep the Kaggle archive extracted under `data/`. The app expects `data/data.csv` and, for genre hints, `data/extracted/data/data_w_genres.csv`.
2. Keep Spotify credentials in the project `.env` file. Copy `.env.example` only if you need a template. In the Spotify Developer Dashboard, add this exact redirect URI: `http://127.0.0.1:8888/callback`.
3. Install requirements into the Python environment you will use to run Flask:

   ```powershell
   C:\Users\Asmit\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
   ```

4. Start the local app:

   ```powershell
   C:\Users\Asmit\AppData\Local\Programs\Python\Python312\python.exe app.py
   ```

5. Open `http://127.0.0.1:8888` and connect Spotify. Start a song, then choose **Start auto queue** once. TuneTwin detects manually changed tracks and queues eight matches. Choose **Stop auto queue** to turn this off; manual dataset search remains available as a fallback.

## Playback notes

Spotify login requests playback, listening-history, and private-playlist permissions. TuneTwin detects playback changes using Spotify's playback-state endpoint and uses the same recommender for each new seed. It skips auto-requeueing when a track appears to have ended naturally, so the current recommendation queue can continue. Spotify's Web API can add queue items but does not provide a clear-queue action; new suggestions append after existing items, and queue entries already present are skipped. The player Like button records positive feedback and saves the track to a private **TuneTwin Liked Songs** playlist. The player's forward button records Skip feedback before moving to the next song. These direct TuneTwin actions update the local adaptive model; external Spotify skips are not treated as feedback. Mood playlists use up to 50 recent plays to select a seed, then include the seed and eight recommendations from TuneTwin's recommender. Spotify playback needs an active device and may require Premium. Reconnect once to grant the playlist permissions.

## Live feedback / LinUCB demo

The Like and Skip buttons on each recommendation are explicit feedback for TuneTwin, not Spotify library actions. Like gives reward `+2`; Skip gives reward `-1`. Each click updates a KNN preference model and a linear contextual bandit using the track's audio features, PCA representation, and K-Means cluster. The next request rebuilds the Top-50 PCA-neighbor candidate pool, applies learned preference when both feedback classes are available, and lets LinUCB rerank candidates using its uncertainty bonus. The small local state is saved at `models/linucb_live.json` (ignored by Git). If the skipped song is currently playing on Spotify, the app also asks Spotify to advance to the next queued track; otherwise Skip only records model feedback. This makes the learning loop demonstrable without relying on automatic listening-event collection.

The learning panel displays recent Like/Skip signals, totals, and a reset button for the local demo model. Each recommendation also shows a brief explanation from its audio, genre, and artist matching signals. The genre tags are joined from the dataset's artist-to-genre table, so treat them as artist-level context rather than verified track genres. Similarity scores are ranking scores and are not evidence of real listener preference.
