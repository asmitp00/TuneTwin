# TuneTwin AI — Flask Demo

TuneTwin searches the provided Spotify track dataset, ranks five similar tracks using standardized audio features plus the dataset's artist-level genre tags, and can play a selected seed or recommendation through Spotify Connect. It favors distinct artists; a second track from an artist is allowed only when it scores at least as highly as the weakest selected track, with a hard cap of two per artist. For a five-track list, it preserves at least four distinct lead artists when the catalog permits.

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

5. Open `http://127.0.0.1:8888`, search for a dataset track, and choose **Find 5 matches**. Connect Spotify, then press **Play** on the seed or a recommendation.

## Playback notes

Spotify login requests only `user-read-playback-state` and `user-modify-playback-state`. The app uses Spotify search to match the dataset title and artist; it does not trust the dataset's IDs as current Spotify IDs. Playback and adding tracks to the queue need a Spotify Premium account, an active Spotify Connect device, and permission for the app to control playback. Playing the seed starts it, then appends matched recommendations to the end of Spotify's existing queue. Spotify does not provide this app a queue-clear operation, and its docs caution that ordering is not guaranteed across separate player API calls. OAuth tokens are saved locally in `.spotify_token_cache`, which is ignored by Git.

## Live feedback / LinUCB demo

The Like and Skip buttons on each recommendation are explicit feedback for TuneTwin, not Spotify library actions. Like gives reward `+2`; Skip gives reward `-1`. Each click updates a shared linear contextual bandit using that track's normalized audio features. The next request rebuilds the Top-50 similar candidate pool and lets LinUCB rerank it using learned preference plus its uncertainty bonus. The small local state is saved at `models/linucb_live.json` (ignored by Git). If the skipped song is currently playing on Spotify, the app also asks Spotify to advance to the next queued track; otherwise Skip only records model feedback. This makes the learning loop demonstrable without relying on automatic listening-event collection.

The genre tags are joined from the dataset's artist-to-genre table, so treat them as artist-level context rather than verified track genres. Similarity scores are ranking scores and are not evidence of real listener preference.
