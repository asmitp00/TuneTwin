/* ==========================================================================
   TuneTwin AI front-end
   MelodyMind-style UI: violet glass shell, hash-free client views, and one
   cover-art palette (--cover-a/b/c) that follows the whole app.
    Keeps Spotify Connect playback, search, queue, and feedback behaviour.
   ========================================================================== */

/* ---------------------------------------------------------------- Search */
const input = document.querySelector("#track-search");
const resultsBox = document.querySelector("#search-results");
let timer;
let controller;

function clearResults() {
  if (!resultsBox) return;
  resultsBox.replaceChildren();
  resultsBox.classList.remove("open");
}

function chooseTrack(track) {
  input.value = `${track.name} — ${track.artist}`;
  clearResults();
  // Picking a search result starts it right away; auto queue fills the rail.
  if (connected) playTrack(track.index, track.index);
  else toast("Connect Spotify to play this song.", true);
}

input?.addEventListener("input", () => {
  clearTimeout(timer);
  if (controller) controller.abort();
  const query = input.value.trim();
  if (query.length < 2) {
    clearResults();
    return;
  }
  timer = setTimeout(async () => {
    controller = new AbortController();
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: controller.signal });
      const data = await response.json();
      clearResults();
      if (!data.results.length) {
        const empty = document.createElement("div");
        empty.className = "search-empty";
        empty.textContent = "No matching tracks found";
        resultsBox.append(empty);
      } else {
        data.results.forEach((track) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "search-result";
          button.setAttribute("role", "option");
          const title = document.createElement("strong");
          title.textContent = track.name;
          const artist = document.createElement("span");
          artist.textContent = track.artist;
          button.append(title, artist);
          button.addEventListener("click", () => chooseTrack(track));
          resultsBox.append(button);
        });
      }
      resultsBox.classList.add("open");
    } catch (error) {
      if (error.name !== "AbortError") console.error("Track search failed", error);
    }
  }, 180);
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".search-wrap")) clearResults();
});

/* ---------------------------------------------------------------- Views */
function scrollMainTo(element, offset = 16) {
  const main = document.querySelector("#main");
  if (!main || !element) return;
  const top = element.getBoundingClientRect().top - main.getBoundingClientRect().top + main.scrollTop;
  main.scrollTo({ top: Math.max(0, top - offset), behavior: "smooth" });
}

function showView(name) {
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("is-active", view.id === `view-${name}`));
  document.querySelectorAll(".nav-link, .nav-tab, .side-link").forEach((btn) => {
    const active = btn.dataset.view === name;
    btn.classList.toggle("active", active);
    if (active) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });
  const scroller = document.querySelector("#main");
  scroller?.scrollTo({ top: 0, behavior: "instant" });
  window.scrollTo({ top: 0, behavior: "instant" });
}
document.querySelectorAll("[data-view]").forEach((el) => {
  el.addEventListener("click", () => showView(el.dataset.view));
});

/* ---------------------------------------------------------------- Player */
const csrf = document.querySelector("#player-panel")?.dataset.csrf || "";
const connected = document.querySelector("#player-panel")?.dataset.connected === "true";
const playerStatus = document.querySelector("#player-status");
let browserPlayer = null;
let browserDeviceId = null;
let browserReady = null;
let sdkTrackUri = "";
let latestState = null;
let seeking = false;

function iconUse(id) {
  return `<svg aria-hidden="true"><use href="#i-${id}"></use></svg>`;
}

let toastTimer;
function toast(message, isError = false) {
  const el = document.querySelector("#toast");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 4800);
}

function setPlayIcon(playing) {
  const button = document.querySelector("#toggle-play");
  if (!button) return;
  const icon = iconUse(playing ? "pause" : "play");
  button.innerHTML = icon;
  button.setAttribute("aria-label", playing ? "Pause" : "Play");
  const nowButton = document.querySelector("#now-toggle-play");
  if (nowButton) {
    nowButton.innerHTML = icon;
    nowButton.setAttribute("aria-label", playing ? "Pause" : "Play");
  }
  document.querySelector("#now-hero")?.classList.toggle("is-playing", playing);
  document.querySelector("#player-panel")?.classList.toggle("is-playing", playing);
}

/* -------------------------------------------------- Cover-art palette */
const appEl = document.querySelector("#app");
const appCoverStorageKey = "tunetwin-last-cover-art";

function setAppCoverBackground(image) {
  let backdrop = image || "";
  try {
    if (backdrop) localStorage.setItem(appCoverStorageKey, backdrop);
    else backdrop = localStorage.getItem(appCoverStorageKey) || "";
  } catch (error) { /* keep the live cover even when storage is unavailable */ }
  if (!appEl) return;
  if (backdrop) {
    appEl.style.setProperty("--app-cover-image", `url("${backdrop}")`);
    appEl.classList.add("has-cover-art");
  } else {
    appEl.style.removeProperty("--app-cover-image");
    appEl.classList.remove("has-cover-art");
  }
}

function rgbCss(c) {
  return `rgb(${Math.round(c[0])}, ${Math.round(c[1])}, ${Math.round(c[2])})`;
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return { h: (h / 6) * 360, s, l };
}

function resetPalette() {
  if (!appEl) return;
  const themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) themeColor.content = "#111018";
  [appEl, document.documentElement].forEach((el) => {
    el.style.removeProperty("--cover-a");
    el.style.removeProperty("--cover-b");
    el.style.removeProperty("--cover-c");
  });
}

function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360 / 360;
  if (s === 0) { const v = Math.round(l * 255); return [v, v, v]; }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hue = (t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [hue(h + 1 / 3) * 255, hue(h) * 255, hue(h - 1 / 3) * 255];
}

/* k-means over a downsampled cover; clusters keep their real hue so the theme
   matches the artwork instead of being pushed toward brand violet. */
function extractPalette(img) {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);

  const pixels = [];
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) continue;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const { s, l } = rgbToHsl(r, g, b);
    if (l < 0.06 || l > 0.96) continue;
    pixels.push([r, g, b, s]);
  }
  if (!pixels.length) return null;

  const k = 6;
  const step = Math.max(1, Math.floor(pixels.length / k));
  let centroids = [];
  for (let i = 0; i < k; i += 1) {
    const p = pixels[Math.min(pixels.length - 1, i * step)];
    centroids.push([p[0], p[1], p[2]]);
  }
  let clusters = [];
  for (let iter = 0; iter < 9; iter += 1) {
    clusters = Array.from({ length: k }, () => ({ r: 0, g: 0, b: 0, s: 0, n: 0 }));
    for (const p of pixels) {
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c += 1) {
        const dr = p[0] - centroids[c][0];
        const dg = p[1] - centroids[c][1];
        const db = p[2] - centroids[c][2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bestDist) { bestDist = d; best = c; }
      }
      const cl = clusters[best];
      cl.r += p[0]; cl.g += p[1]; cl.b += p[2]; cl.s += p[3]; cl.n += 1;
    }
    centroids = clusters.map((cl) => (cl.n ? [cl.r / cl.n, cl.g / cl.n, cl.b / cl.n] : [0, 0, 0]));
  }

  return clusters
    .filter((cl) => cl.n > pixels.length * 0.015)
    .map((cl) => {
      const r = cl.r / cl.n;
      const g = cl.g / cl.n;
      const b = cl.b / cl.n;
      const { h, s, l } = rgbToHsl(r, g, b);
      return { r, g, b, h, s, l, n: cl.n, vibrancy: s * (1 - Math.abs(l - 0.5)) };
    });
}

/* Keep the cover's real hue; only nudge saturation/lightness enough to stay legible. */
function usable(cluster) {
  return hslToRgb(cluster.h, Math.min(0.94, Math.max(0.28, cluster.s)), Math.min(0.78, Math.max(0.32, cluster.l)));
}

/* Background variant: same hue, kept dark so white text stays readable. */
function usableBg(cluster) {
  return hslToRgb(cluster.h, Math.min(0.82, Math.max(0.24, cluster.s)), Math.min(0.5, Math.max(0.16, cluster.l)));
}

function hueGap(a, b) {
  const d = Math.abs(a.h - b.h) % 360;
  return d > 180 ? 360 - d : d;
}

/* Accent = most vibrant hue, secondary = a different hue, tertiary = the artwork's
   dominant colour, so the page backdrop still reads like the cover. */
function chooseTheme(clusters) {
  const pool = clusters.filter((c) => c.s > 0.12);
  const source = pool.length ? pool : clusters;
  const byVibrancy = [...source].sort((a, b) => (b.vibrancy * b.n) - (a.vibrancy * a.n));
  const accent = byVibrancy[0];
  const secondary = byVibrancy.find((c) => c !== accent && hueGap(c, accent) > 26) || byVibrancy[1] || accent;
  const dominant = [...clusters].sort((a, b) => b.n - a.n)[0] || accent;
  const tertiary = hueGap(dominant, accent) > 22 ? dominant : (byVibrancy.find((c) => hueGap(c, accent) > 48) || secondary);
  return [accent, secondary, tertiary];
}

function setPalette(clusters) {
  if (!appEl || !clusters || !clusters.length) return;
  const [a, b, c] = chooseTheme(clusters);
  // --cover-b is the app's main accent, so map the most vivid cover colour there.
  const values = [rgbCss(usable(b)), rgbCss(usable(a)), rgbCss(usableBg(c))];
  const themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) themeColor.content = values[1];
  [appEl, document.documentElement].forEach((el) => {
    el.style.setProperty("--cover-a", values[0]);
    el.style.setProperty("--cover-b", values[1]);
    el.style.setProperty("--cover-c", values[2]);
  });
}

const playerArt = document.querySelector("#player-art");
let lastThemeSrc = "";

/* Theme source: the playing track's cover, otherwise the first recommendation
   cover. Locking to one source stops the palette flickering as covers load. */
function paletteSource() {
  const heroArt = document.querySelector("#now-art");
  if (heroArt && heroArt.getAttribute("src")) return heroArt;
  if (playerArt && playerArt.getAttribute("src")) return playerArt;
  return document.querySelector(".row-art img, .image img, .side-item-art img");
}

function extractAndApply(img) {
  try {
    const clusters = extractPalette(img);
    if (clusters && clusters.length) setPalette(clusters);
    else resetPalette();
  } catch (error) {
    resetPalette();
  }
}

function refreshTheme() {
  const img = paletteSource();
  if (!img || !img.src) { resetPalette(); lastThemeSrc = ""; return; }
  if (!(img.complete && img.naturalWidth)) {
    if (!img.dataset.themeHooked) {
      img.dataset.themeHooked = "1";
      img.addEventListener("load", refreshTheme);
    }
    return;
  }
  if (img.src === lastThemeSrc) return;
  lastThemeSrc = img.src;
  extractAndApply(img);
}

if (playerArt) {
  new MutationObserver(refreshTheme).observe(playerArt, { attributes: true, attributeFilter: ["src"] });
  playerArt.addEventListener("load", refreshTheme);
}
const nowArtObs = document.querySelector("#now-art");
if (nowArtObs) {
  new MutationObserver(refreshTheme).observe(nowArtObs, { attributes: true, attributeFilter: ["src"] });
  nowArtObs.addEventListener("load", refreshTheme);
}
document.querySelectorAll(".row-art img").forEach((img) => {
  if (img.complete && img.naturalWidth) return;
  img.addEventListener("load", refreshTheme, { once: true });
});
refreshTheme();

/* ------------------------------------------------------- Helpers */
function formatTime(ms = 0) {
  const seconds = Math.floor(Math.max(0, ms) / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

async function api(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { "X-CSRF-Token": csrf, ...(options.headers || {}) } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function playerAction(action, data = {}) {
  return api(`/api/spotify/control/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_id: browserDeviceId || undefined, ...data }),
  });
}

/* ------------------------------------------------------- Player state */
async function refreshPlayer() {
  if (!connected) return;
  try {
    latestState = await api("/api/spotify/state");
    if (latestState.track) {
      document.querySelector("#player-title").textContent = latestState.track;
      document.querySelector("#player-artist").textContent = latestState.artist || "";
      const art = document.querySelector("#player-art");
      const imageUrl = latestState.image || "";
      if (art.getAttribute("src") !== imageUrl) art.src = imageUrl;
      art.classList.toggle("visible", Boolean(latestState.image));
      setPlayIcon(Boolean(latestState.is_playing));
      const progress = document.querySelector("#progress");
      if (!seeking) progress.value = latestState.duration_ms ? Math.round(latestState.progress_ms / latestState.duration_ms * 1000) : 0;
      document.querySelector("#current-time").textContent = formatTime(latestState.progress_ms);
      document.querySelector("#duration").textContent = formatTime(latestState.duration_ms);
      const shuffleLabel = document.querySelector("#shuffle-label");
      const repeatLabel = document.querySelector("#repeat-label");
      if (shuffleLabel) shuffleLabel.textContent = `Shuffle: ${latestState.shuffle ? "on" : "off"}`;
      if (repeatLabel) repeatLabel.textContent = `Repeat: ${latestState.repeat || "off"}`;
       if (playerStatus) playerStatus.textContent = `${latestState.is_playing ? "Playing" : "Paused"} on ${latestState.device_name || "Spotify"}`;
      const sideDevice = document.querySelector("#side-device");
      if (sideDevice && latestState.device_name) sideDevice.textContent = `${latestState.is_playing ? "Playing" : "Paused"} on ${latestState.device_name}`;
    } else {
      const art = document.querySelector("#player-art");
      if (art) { art.removeAttribute("src"); art.classList.remove("visible"); }
      const title = document.querySelector("#player-title");
      const artist = document.querySelector("#player-artist");
      if (title) title.textContent = "Nothing playing";
      if (artist) artist.textContent = "Pick a song to start";
      if (playerStatus) playerStatus.textContent = "No active Spotify playback";
    }
    renderNowHero();
    await syncQueue();
  } catch (error) { if (playerStatus) playerStatus.textContent = error.message; }
}

/* ------------------------------------------------ Website-only Spotify player */
function activateBrowserAudio() {
  try { browserPlayer?.activateElement?.(); } catch (error) { /* SDK compatibility */ }
}

function initialiseBrowserPlayer() {
  if (!connected || !window.Spotify?.Player || browserPlayer) return browserReady;
  browserReady = new Promise((resolve, reject) => {
    browserPlayer = new Spotify.Player({
      name: "TuneTwin web player",
      volume: 0.5,
      getOAuthToken: async (callback) => {
        try { callback((await api("/api/spotify/token")).access_token); }
        catch (error) { reject(error); if (playerStatus) playerStatus.textContent = "Reconnect Spotify to play in this browser"; }
      },
    });
    browserPlayer.addListener("ready", ({ device_id }) => {
      browserDeviceId = device_id;
      if (playerStatus) playerStatus.textContent = "Web player ready";
      resolve(device_id);
    });
    browserPlayer.addListener("not_ready", () => {
      browserDeviceId = null;
      if (playerStatus) playerStatus.textContent = "Web player disconnected — refresh and try again";
    });
    browserPlayer.addListener("initialization_error", ({ message }) => { reject(new Error(message)); if (playerStatus) playerStatus.textContent = `Web player error: ${message}`; });
    browserPlayer.addListener("authentication_error", ({ message }) => { reject(new Error(message)); if (playerStatus) playerStatus.textContent = "Reconnect Spotify to play in this browser"; });
    browserPlayer.addListener("account_error", ({ message }) => { reject(new Error(message)); if (playerStatus) playerStatus.textContent = `Web playback needs Spotify Premium: ${message}`; });
    browserPlayer.addListener("playback_error", ({ message }) => { if (playerStatus) playerStatus.textContent = `Playback error: ${message}`; });
    browserPlayer.addListener("player_state_changed", (state) => {
      if (!state) return;
      setPlayIcon(!state.paused);
      if (playerStatus) playerStatus.textContent = state.paused ? "Paused in this browser" : "Playing in this browser";
      // Track changes are the fastest way to learn that a mix just ended, so
      // sync immediately instead of waiting for the next 4s poll.
      const uri = state.track_window?.current_track?.uri || "";
      if (uri && uri !== sdkTrackUri) {
        sdkTrackUri = uri;
        refreshPlayer();
      }
    });
    browserPlayer.connect().then((ok) => {
      if (!ok) reject(new Error("The web player could not connect."));
    }).catch(reject);
  });
  return browserReady;
}

window.onSpotifyWebPlaybackSDKReady = () => { initialiseBrowserPlayer().catch(() => {}); };

/* ------------------------------------------------------- Queue */
/* Queue display is app-managed now (see renderQueue / removeQueueItem). */

/* ------------------------------------------------------- Learning */
async function refreshLearning() {
  try {
    const stats = await api("/api/learning");
    document.querySelectorAll(".js-learn-updates").forEach((el) => { el.textContent = stats.updates; });
    const likes = document.querySelector("#learn-likes");
    const skips = document.querySelector("#learn-skips");
    if (likes) likes.textContent = stats.likes;
    if (skips) skips.textContent = stats.skips;
    const history = document.querySelector("#learning-history");
    if (history) {
      history.replaceChildren();
      stats.recent.slice(0, 8).forEach((event) => {
        const row = document.createElement("span"); row.className = `learning-event ${event.feedback}`;
        row.textContent = `${event.feedback === "like" ? "+2 Like" : "−1 Dislike"} · ${event.track} — ${event.artist}`; history.append(row);
      });
    }
  } catch (error) { console.warn("Could not refresh learning stats", error); }
}

document.querySelector("#toggle-play")?.addEventListener("click", async () => {
  try {
    activateBrowserAudio();
    await initialiseBrowserPlayer();
    await browserPlayer.togglePlay();
    setTimeout(refreshPlayer, 300);
  } catch (error) { if (playerStatus) playerStatus.textContent = error.message; }
});

async function sendPlayerFeedback(feedback) {
  const status = document.querySelector("#player-feedback-status");
  const button = document.querySelector(feedback === "skip" ? "#next-track" : "#like-current-track");
  if (button?.disabled) return;
  if (feedback === "skip") { clearTimeout(mixEndTimer); mixEndTimer = null; mixEnding = false; }
  if (button) button.disabled = true;
  if (status) status.textContent = feedback === "skip" ? "Saving dislike…" : "Saving like…";
  try {
    const result = await api("/api/spotify/player-feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feedback }),
    });
    if (status) {
      status.replaceChildren(document.createTextNode(result.message));
      if (result.playlist_url) {
        const link = document.createElement("a"); link.href = result.playlist_url; link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = " Open playlist"; status.append(link);
      }
    }
    await refreshLearning().catch(() => {});
    if (feedback === "skip") {
      await refreshPlayer();
    }
  } catch (error) { if (status) status.textContent = error.message; }
  finally { if (button) button.disabled = false; }
}

document.querySelector("#next-track")?.addEventListener("click", () => sendPlayerFeedback("skip"));
document.querySelector("#like-current-track")?.addEventListener("click", () => sendPlayerFeedback("like"));
document.querySelector("#previous-track")?.addEventListener("click", () => playerAction("previous").then(refreshPlayer).catch((e) => playerStatus.textContent = e.message));
document.querySelector("#progress")?.addEventListener("input", () => { seeking = true; });
document.querySelector("#progress")?.addEventListener("change", async (event) => {
  seeking = false;
  const duration = latestState?.duration_ms || 0;
  try { await playerAction("seek", { position_ms: Math.round(Number(event.target.value) / 1000 * duration) }); }
  catch (error) { playerStatus.textContent = error.message; }
});
let currentVol = 50;
let lastNonzeroVol = 50;
let volumeRange = document.querySelector("#volume");
let volumeToggle = document.querySelector("#volume-toggle");
const volumeActions = volumeToggle?.parentElement
  || document.querySelector(".player-right-actions")
  || document.querySelector(".player-right");
if (volumeActions && !volumeToggle) {
  volumeToggle = document.createElement("button");
  volumeToggle.type = "button";
  volumeToggle.id = "volume-toggle";
  volumeToggle.className = "player-action-btn icon-btn";
  volumeToggle.setAttribute("aria-label", "Mute or unmute");
  volumeToggle.title = "Mute or unmute";
  volumeToggle.innerHTML = iconUse("volume");
  volumeActions.prepend(volumeToggle);
}
if (volumeActions && !volumeRange) {
  volumeRange = document.createElement("input");
  volumeRange.type = "range";
  volumeRange.id = "volume";
  volumeRange.min = "0";
  volumeRange.max = "100";
  volumeRange.value = String(currentVol);
}
if (volumeRange && volumeActions) {
  volumeRange.classList.add("player-volume-range");
  volumeRange.setAttribute("aria-label", "Volume");
  volumeRange.title = "Adjust volume";
  if (volumeToggle) volumeActions.insertBefore(volumeRange, volumeToggle.nextSibling);
  else volumeActions.append(volumeRange);
  volumeRange.addEventListener("input", (event) => {
    currentVol = Number(event.target.value);
    if (currentVol > 0) lastNonzeroVol = currentVol;
    volumeToggle?.setAttribute("aria-pressed", String(currentVol === 0));
  });
  volumeRange.addEventListener("change", (event) => playerAction("volume", { volume: Number(event.target.value) }).catch((error) => { if (playerStatus) playerStatus.textContent = error.message; }));
}
volumeToggle?.addEventListener("click", () => {
  if (currentVol > 0) {
    lastNonzeroVol = currentVol;
    currentVol = 0;
  } else currentVol = lastNonzeroVol;
  if (volumeRange) volumeRange.value = String(currentVol);
  volumeToggle?.setAttribute("aria-pressed", String(currentVol === 0));
  playerAction("volume", { volume: currentVol }).then(() => toast(currentVol === 0 ? "Muted" : `Volume ${currentVol}%`)).catch(() => {});
});
document.querySelector("#shuffle-toggle")?.addEventListener("click", () => playerAction("shuffle", { enabled: !latestState?.shuffle }).then(refreshPlayer).catch((e) => playerStatus.textContent = e.message));
document.querySelector("#repeat-toggle")?.addEventListener("click", () => {
  const state = latestState?.repeat === "off" ? "context" : latestState?.repeat === "context" ? "track" : "off";
  playerAction("repeat", { state }).then(refreshPlayer).catch((e) => playerStatus.textContent = e.message);
});
document.querySelector("#refresh-queue")?.addEventListener("click", () => { refreshPlayer(); renderQueue(); });

// Keep the complete queue in the Up Next section instead of a second popover.
const queueList = document.querySelector("#queue-items");
const queueSection = document.querySelector(".now-side") || document.querySelector("#lyrics-unit");
const queuePreview = document.querySelector("#now-upnext");
const queueHeader = queueSection?.querySelector(".now-side-head");
const queueTools = queueSection?.querySelector(".now-queue-toolbar") || document.createElement("div");
queueTools.classList.add("now-queue-toolbar");
const queueSummary = document.querySelector("#queue-summary");
const queueRefresh = document.querySelector("#refresh-queue");
const autoQueueControl = document.querySelector("#auto-queue-toggle");
document.querySelector("#queue-toggle")?.remove();
document.querySelector("#now-upnext-count")?.remove();
if (queueSection && queueList) {
  queueSection.querySelector(".now-side-note")?.remove();
  if (queueHeader && queueSummary && !queueHeader.contains(queueSummary)) queueHeader.append(queueSummary);
  if (queueRefresh && !queueTools.contains(queueRefresh)) queueTools.append(queueRefresh);
  if (autoQueueControl && !queueTools.contains(autoQueueControl)) queueTools.append(autoQueueControl);
  if (!queueTools.parentElement) queueSection.insertBefore(queueTools, queuePreview || queueList);
  if (queuePreview && queuePreview !== queueList) queuePreview.replaceWith(queueList);
  else if (!queueSection.contains(queueList)) queueSection.append(queueList);
  queueList.classList.add("now-queue-list");
}
document.querySelector("#queue-popover")?.remove();
document.querySelector("#reset-learning")?.addEventListener("click", async () => {
  if (!confirm("Reset all saved TuneTwin Like/Dislike learning for this demo?")) return;
  try { await api("/api/learning/reset", { method: "POST" }); await refreshLearning(); location.reload(); }
  catch (error) { alert(error.message); }
});
document.querySelectorAll(".feedback-form").forEach((form) => form.addEventListener("submit", () => setTimeout(refreshLearning, 700)));

// Play without a page reload so the web player state stays intact.
const autoQueueKey = "tunetwin-auto-queue";
const queueStoreKey = "tunetwin-queue";
let playInFlight = false;
let appQueue = [];
let queueMode = "auto";
let mixEnding = false;
let mixEndTimer = null;
let queueSyncFor = "";
let queueSyncInFlight = false;
let autoQueueEnabled = true;
const AUTO_QUEUE_TARGET = 8;
let prefetchedTracks = [];
let prefetchInFlight = false;
let topUpInFlight = false;
let restartInFlight = false;
try { autoQueueEnabled = localStorage.getItem(autoQueueKey) !== "false"; }
catch (error) { /* storage unavailable */ }

function queueKey(track) {
  const norm = (value) => String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  return `${norm(track?.name)}|${norm(track?.artist)}`;
}

function paintAutoQueue() {
  document.querySelector("#auto-queue-toggle")?.setAttribute("aria-checked", String(autoQueueEnabled));
}

function saveQueue() {
  try {
    sessionStorage.setItem(queueStoreKey, JSON.stringify(appQueue));
    sessionStorage.setItem("tunetwin-queue-mode", queueMode);
  } catch (error) { /* storage unavailable */ }
}

function restoreQueue() {
  try {
    const stored = JSON.parse(sessionStorage.getItem(queueStoreKey) || "[]");
    if (Array.isArray(stored)) appQueue = stored.filter((track) => track && track.uri);
    const mode = sessionStorage.getItem("tunetwin-queue-mode");
    if (["auto", "track", "mix", "playlist"].includes(mode)) queueMode = mode;
  } catch (error) { /* ignore malformed storage */ }
}

async function playTrack(rowIndex, seedIndex, { positionMs = 0, silent = false } = {}) {
  if (playInFlight) return;
  clearTimeout(mixEndTimer); mixEndTimer = null;
  prefetchedTracks = [];
  playInFlight = true;
  activateBrowserAudio();
  try {
    await initialiseBrowserPlayer();
    if (!browserDeviceId) throw new Error("The web player is still connecting. Try again in a moment.");
    const data = await api("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        row_index: String(rowIndex),
        seed_index: String(seedIndex),
        include_recommendations: autoQueueEnabled,
        position_ms: positionMs,
        device_id: browserDeviceId,
      }),
    });
    if (Array.isArray(data.queue)) {
      appQueue = data.queue.slice(data.offset || 0);
      queueMode = autoQueueEnabled ? "auto" : "track";
      mixEnding = false;
      renderQueue();
    }
    // Ignore the previous track's state while Spotify switches to the new one.
    queueSyncFor = latestState?.uri || "";
    if (!silent) toast(data.message || "Playing on Spotify.");
    await refreshPlayer();
  } catch (error) {
    toast(error.message, true);
  } finally {
    playInFlight = false;
  }
}

/* The app queue is the app's own list (played as a Spotify context), so items can
   be removed - unlike Spotify's native queue, which its API cannot edit. */
function queueGroupLabel(text) {
  const el = document.createElement("p");
  el.className = "queue-group";
  el.textContent = text;
  return el;
}

function normalizeTrackKey(track) {
  return `${String(track.name || "").trim().toLowerCase()}|${String(track.artist || "").trim().toLowerCase()}`;
}

function cleanQueue() {
  const currentUri = latestState?.uri || "";
  const currentKey = latestState?.track
    ? normalizeTrackKey({ name: latestState.track, artist: latestState.artist || "" })
    : "";
  const seen = new Set();
  appQueue = appQueue.filter((track, index) => {
    if (!track?.uri) return false;
    const songKey = normalizeTrackKey(track);
    if (seen.has(track.uri) || seen.has(songKey)) return false;
    if (index > 0 && (track.uri === currentUri || (currentKey && songKey === currentKey))) return false;
    seen.add(track.uri);
    seen.add(songKey);
    return true;
  });
}

function buildQueueRow(track, index, currentIndex) {
  const row = document.createElement("div");
  row.className = `queue-item${index === currentIndex ? " is-current" : ""}`;

  const play = document.createElement("button");
  play.type = "button";
  play.className = "queue-play";
  play.setAttribute("aria-label", `Play ${track.name}`);
  play.title = `Play ${track.name}`;

  const art = document.createElement("div");
  art.className = "queue-art";
  if (track.art) {
    const img = document.createElement("img");
    img.src = track.art;
    img.alt = "";
    img.decoding = "async";
    img.crossOrigin = "anonymous";
    img.referrerPolicy = "no-referrer";
    img.addEventListener("error", () => img.remove());
    art.append(img);
  }
  const main = document.createElement("div");
  main.className = "queue-main";
  const name = document.createElement("strong");
  name.textContent = track.name;
  name.title = track.name;
  const artist = document.createElement("span");
  artist.textContent = track.artist || "";
  artist.title = track.artist || "";
  main.append(name, artist);
  play.append(art, main);
  play.addEventListener("click", () => playQueueAt(index));

  row.append(play);
  if (index !== currentIndex) {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "queue-remove";
    remove.setAttribute("aria-label", `Remove ${track.name} from the queue`);
    remove.title = "Remove from queue";
    remove.innerHTML = iconUse("trash");
    remove.addEventListener("click", () => removeQueueItem(index));
    row.append(remove);
  }
  return row;
}

const artRequested = new Set();

/* Queue and mix rows sometimes arrive without artwork. Ask the catalogue for
   the missing covers lazily, then repaint once they resolve. */
function ensureQueueArt() {
  const tracks = [...appQueue, ...prefetchedTracks];
  const need = tracks.filter(
    (track) => track && !track.art && Number.isInteger(track.index) && track.index >= 0 && !artRequested.has(track.index)
  );
  if (!need.length) return;
  const indices = [...new Set(need.map((track) => track.index))];
  indices.forEach((index) => artRequested.add(index));
  api("/api/catalog/art", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ indices }),
  }).then((data) => {
    const art = data.art || {};
    let changed = false;
    [appQueue, prefetchedTracks].forEach((list) => {
      list.forEach((track) => {
        if (track && !track.art && track.index != null && art[String(track.index)]) {
          track.art = art[String(track.index)];
          changed = true;
        }
      });
    });
    if (changed) {
      saveQueue();
      renderQueue();
    }
  }).catch(() => {
    indices.forEach((index) => artRequested.delete(index));
  });
}

function renderQueue() {
  const list = document.querySelector("#queue-items");
  const summary = document.querySelector("#queue-summary");
  if (!list) return;
  cleanQueue();
  setAppCoverBackground(latestState?.image || appQueue.find((track) => track?.art)?.art || "");
  saveQueue();
  const currentIndex = latestState?.uri
    ? appQueue.findIndex((track) => track.uri === latestState.uri)
    : -1;
  // Spotify's own state is the most reliable cover for the current row.
  if (currentIndex >= 0 && latestState?.image) appQueue[currentIndex].art = latestState.image;
  ensureQueueArt();
  list.replaceChildren();
  if (summary) {
    if (!appQueue.length) summary.textContent = "0 tracks";
    else if (currentIndex < 0 && (queueMode === "mix" || queueMode === "playlist")) {
      summary.textContent = `${appQueue.length} tracks · finished`;
    } else summary.textContent = `${appQueue.length} track${appQueue.length === 1 ? "" : "s"}`;
  }
  if (!appQueue.length) {
    const empty = document.createElement("div");
    empty.className = "queue-empty";
    empty.innerHTML = `${iconUse("note")}<strong>Start with a song</strong><span>Choose a track to build your queue.</span>`;
    list.append(empty);
    return;
  }
  appQueue.forEach((track, index) => {
    if (index === currentIndex) list.append(queueGroupLabel("Now playing"));
    else if (index === (currentIndex < 0 ? 0 : currentIndex + 1)) list.append(queueGroupLabel("Up next"));
    list.append(buildQueueRow(track, index, currentIndex));
  });
}

/* ------------------------------------------------------- Now playing stage */

function renderNowHero() {
  const hero = document.querySelector("#now-hero");
  if (!hero) return;
  const image = latestState?.image || "";
  const art = document.querySelector("#now-art");
  if (art) {
    if (image) {
      if (art.getAttribute("src") !== image) art.src = image;
      art.style.display = "block";
      art.classList.add("visible");
    } else {
      art.removeAttribute("src");
      art.style.display = "none";
      art.classList.remove("visible");
    }
  }
  document.querySelector("#now-art-box")?.classList.toggle("has-art", Boolean(image));
  const label = document.querySelector("#now-vinyl-label");
  if (label) label.style.backgroundImage = image ? `url("${image}")` : "";
  hero.style.setProperty("--hero-image", image ? `url("${image}")` : "none");
  setAppCoverBackground(image);
  hero.classList.toggle("has-image", Boolean(image));

  const hasTrack = Boolean(latestState?.track);
  const title = latestState?.track || "Nothing playing";
  const artist = latestState?.artist || "Pick a song to start";
  const titleEl = document.querySelector("#now-title");
  if (titleEl) titleEl.textContent = title;
  const artistEl = document.querySelector("#now-artist");
  if (artistEl) artistEl.textContent = artist;
  const albumEl = document.querySelector("#now-album");
  if (albumEl) albumEl.textContent = latestState?.album || "";
  const ghost = document.querySelector("#now-ghost");
  if (ghost) ghost.textContent = hasTrack ? title.toUpperCase() : "YOUR MUSIC";

  const kicker = document.querySelector("#now-kicker");
  if (kicker) { kicker.textContent = hasTrack ? "SQ" : "READY"; kicker.hidden = !hasTrack; }
  const source = document.querySelector("#now-source");
  if (source) {
    source.textContent = latestState?.device_name
      ? `Playing on ${latestState.device_name}`
      : "TuneTwin AI";
    source.hidden = !hasTrack;
  }
  const playingLabel = document.querySelector("#hero-playing-label");
  if (playingLabel) {
    playingLabel.textContent = latestState?.track
      ? (latestState.is_playing ? "Playing now" : "Paused")
      : "Ready when you are";
  }
  hero.classList.toggle("has-track", hasTrack);

  // Sidebar mini now-playing card mirrors the hero.
  const sideArt = document.querySelector("#side-now-art");
  if (sideArt) {
    if (image) {
      if (sideArt.getAttribute("src") !== image) sideArt.src = image;
      sideArt.style.display = "block";
      sideArt.classList.add("visible");
    } else {
      sideArt.removeAttribute("src");
      sideArt.style.display = "none";
      sideArt.classList.remove("visible");
    }
  }
  document.querySelector(".mini-player-cover")?.classList.toggle("has-art", Boolean(image));
  const sideTitle = document.querySelector("#side-now-title");
  if (sideTitle) sideTitle.textContent = title;
  const sideArtist = document.querySelector("#side-now-artist");
  if (sideArtist) sideArtist.textContent = artist;

  const duration = latestState?.duration_ms || 0;
  const progress = latestState?.progress_ms || 0;
  const nowBar = document.querySelector("#now-progress");
  if (nowBar && !seeking) nowBar.value = duration ? Math.round(progress / duration * 1000) : 0;
  const nowCurrent = document.querySelector("#now-current-time");
  if (nowCurrent) nowCurrent.textContent = formatTime(progress);
  const nowDuration = document.querySelector("#now-duration");
  if (nowDuration) nowDuration.textContent = formatTime(duration);
  hero.classList.toggle("is-playing", Boolean(latestState?.is_playing));
}

function focusNowHero() {
  const hero = document.querySelector("#now-hero");
  if (!hero) return;
  scrollMainTo(hero, 0);
}

document.querySelector("#side-now-card")?.addEventListener("click", focusNowHero);
document.querySelector("#side-like-btn")?.addEventListener("click", (event) => {
  event.stopPropagation();
  sendPlayerFeedback("like");
});
document.querySelector("#cover-btn-like")?.addEventListener("click", () => sendPlayerFeedback("like"));
document.querySelector("#cover-btn-share")?.addEventListener("click", () => {
  if (latestState?.uri) {
    const id = latestState.uri.replace("spotify:track:", "");
    window.open(`https://open.spotify.com/track/${id}`, "_blank", "noopener,noreferrer");
  } else {
    toast("No active track to open in Spotify.");
  }
});
document.querySelector("#cover-btn-folder")?.addEventListener("click", () => showView("playlists"));
document.querySelector("#cover-btn-comment")?.addEventListener("click", () => {
  scrollMainTo(document.querySelector(".comment-column"));
});
document.querySelector(".player-track")?.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  focusNowHero();
});
document.querySelector(".player-track")?.setAttribute("title", "Jump to now playing");

document.querySelector("#now-toggle-play")?.addEventListener("click", async () => {
  try {
    activateBrowserAudio();
    await initialiseBrowserPlayer();
    await browserPlayer.togglePlay();
    setTimeout(refreshPlayer, 300);
  } catch (error) {
    if (playerStatus) playerStatus.textContent = error.message;
  }
});
document.querySelector("#now-prev")?.addEventListener("click", () => playerAction("previous").then(refreshPlayer).catch((error) => toast(error.message, true)));
document.querySelector("#now-next")?.addEventListener("click", () => sendPlayerFeedback("skip"));
document.querySelector("#now-like")?.addEventListener("click", () => sendPlayerFeedback("like"));
document.querySelector("#now-progress")?.addEventListener("input", () => { seeking = true; });
document.querySelector("#now-progress")?.addEventListener("change", async (event) => {
  seeking = false;
  const duration = latestState?.duration_ms || 0;
  const position = Math.round(Number(event.target.value) / 1000 * duration);
  try {
    await playerAction("seek", { position_ms: position });
    const bottom = document.querySelector("#progress");
    if (bottom) bottom.value = event.target.value;
    document.querySelector("#current-time").textContent = formatTime(position);
    document.querySelector("#now-current-time").textContent = formatTime(position);
  } catch (error) {
    toast(error.message, true);
  }
});

async function playQueueAt(index) {
  if (playInFlight || index < 0 || index >= appQueue.length) return;
  if (!browserDeviceId) { toast("The web player is still connecting. Try again in a moment.", true); return; }
  clearTimeout(mixEndTimer); mixEndTimer = null;
  const target = appQueue[index];
  appQueue = appQueue.slice(index);
  mixEnding = false;
  renderQueue();
  playInFlight = true;
  activateBrowserAudio();
  try {
    if (queueMode === "auto") await topUpQueue();
    await api("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        uris: appQueue.map((track) => track.uri),
        offset: 0,
        device_id: browserDeviceId,
      }),
    });
    queueSyncFor = latestState?.uri || "";
    toast(`Playing ${target.name}.`);
    await refreshPlayer();
  } catch (error) {
    toast(error.message, true);
  } finally {
    playInFlight = false;
  }
}

/* Stop right at the end of a mix/playlist instead of letting Spotify's
   autoplay pick a new song, which used to wipe out the mix queue. */
function scheduleMixEndPause() {
  clearTimeout(mixEndTimer);
  mixEndTimer = null;
  if (!mixEnding || !latestState?.is_playing) return;
  if (queueMode !== "mix" && queueMode !== "playlist") return;
  const duration = Number(latestState.duration_ms) || 0;
  const remaining = duration - (Number(latestState.progress_ms) || 0) - 400;
  if (!duration || remaining <= 0 || remaining > 45000) return;
  mixEndTimer = setTimeout(() => playerAction("pause").catch(() => {}), remaining);
}

async function requestRefill(count, extraExcludes = []) {
  const excluded = [...appQueue, ...prefetchedTracks, ...extraExcludes];
  const data = await api("/api/queue/refill", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      seed_index: Number.isInteger(latestState?.index) ? latestState.index : null,
      uris: excluded.map((track) => track.uri),
      keys: excluded.map(queueKey),
      count,
    }),
  });
  return Array.isArray(data.entries) ? data.entries : [];
}

/* Pull the next recommendation before it is needed so a track change never
   waits on the model. Two spares cover quick skips. */
async function prefetchNextTrack() {
  if (queueMode !== "auto" || !autoQueueEnabled) return;
  if (prefetchInFlight || prefetchedTracks.length >= 2) return;
  if (!appQueue.length || latestState?.uri !== appQueue[0]?.uri) return;
  prefetchInFlight = true;
  try {
    const entries = await requestRefill(2 - prefetchedTracks.length);
    entries.forEach((entry) => {
      if (!appQueue.some((track) => track.uri === entry.uri)
          && !prefetchedTracks.some((track) => track.uri === entry.uri)) {
        prefetchedTracks.push(entry);
      }
    });
  } catch (error) { /* the current queue keeps playing */ }
  finally { prefetchInFlight = false; }
}

/* Keep eight recommendations after the current song. One song ends, one new
   pick joins the queue. */
async function topUpQueue() {
  if (queueMode !== "auto" || !autoQueueEnabled || topUpInFlight) return false;
  topUpInFlight = true;
  try {
    const upcoming = Math.max(0, appQueue.length - 1);
    if (upcoming >= AUTO_QUEUE_TARGET) return false;
    const needed = AUTO_QUEUE_TARGET - upcoming;
    const added = [];
    while (added.length < needed && prefetchedTracks.length) {
      const entry = prefetchedTracks.shift();
      if (!entry?.uri) continue;
      if (appQueue.some((track) => track.uri === entry.uri) || added.some((track) => track.uri === entry.uri)) continue;
      added.push(entry);
    }
    if (added.length < needed) {
      try {
        const fetched = await requestRefill(needed - added.length, added);
        fetched.forEach((entry) => {
          if (!entry?.uri) return;
          if (appQueue.some((track) => track.uri === entry.uri) || added.some((track) => track.uri === entry.uri)) return;
          added.push(entry);
        });
      } catch (error) { /* keep whatever is already queued */ }
    }
    if (!added.length) return false;
    appQueue = [...appQueue, ...added];
    renderQueue();
    prefetchNextTrack();
    return true;
  } finally {
    topUpInFlight = false;
  }
}

/* Rebuild the Spotify playback context so the newly appended recommendation
   is actually part of what plays next. */
async function restartAutoQueue() {
  if (queueMode !== "auto" || !browserDeviceId || !appQueue.length || playInFlight || restartInFlight) return;
  restartInFlight = true;
  try {
    await api("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        uris: appQueue.map((track) => track.uri),
        offset: 0,
        position_ms: Math.max(0, Math.round(latestState?.progress_ms || 0)),
        device_id: browserDeviceId,
      }),
    });
    queueSyncFor = latestState?.uri || "";
  } catch (error) { /* the existing playback context keeps going */ }
  finally { restartInFlight = false; }
}

async function syncQueue() {
  const uri = latestState?.uri;
  if (!uri || queueSyncInFlight) return;
  const before = appQueue.length;
  cleanQueue();
  const queueChanged = appQueue.length !== before;
  const index = appQueue.findIndex((track) => track.uri === uri);
  if (index >= 0) mixEnding = index === appQueue.length - 1;
  scheduleMixEndPause();
  if (index > 0) {
    appQueue = appQueue.slice(index);
    queueSyncFor = uri;
    const added = await topUpQueue();
    if (!added) renderQueue();
    if (added) await restartAutoQueue();
    return;
  }
  if (index === 0) {
    if (queueChanged) renderQueue();
    prefetchNextTrack();
    return;
  }
  if (queueMode === "mix" || queueMode === "playlist") {
    // The list has finished (or playback moved outside it). Keep the list
    // instead of replacing it with recommendations for the new song.
    if (mixEnding) {
      mixEnding = false;
      playerAction("pause").catch(() => {});
    }
    return;
  }
  if (queueMode === "track") {
    appQueue = [{
      index: null,
      name: latestState.track,
      artist: latestState.artist,
      uri,
      art: latestState.image,
    }];
    renderQueue();
    return;
  }
  if (queueSyncFor === uri) return;
  queueSyncFor = uri;
  if (!autoQueueEnabled) {
    appQueue = [{
      index: null,
      name: latestState.track,
      artist: latestState.artist,
      uri,
      art: latestState.image,
    }];
    renderQueue();
    return;
  }
  queueSyncInFlight = true;
  try {
    const data = await api("/api/queue/current", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (latestState?.uri !== uri) return;
    appQueue = Array.isArray(data.entries) ? data.entries : [];
    renderQueue();
  } catch (error) {
    // Keep the existing queue when Spotify is between tracks.
    queueSyncFor = "";
  } finally {
    queueSyncInFlight = false;
  }
}

async function removeQueueItem(index) {
  if (index < 0 || index >= appQueue.length) return;
  clearTimeout(mixEndTimer); mixEndTimer = null;
  appQueue.splice(index, 1);
  mixEnding = false;
  renderQueue();
  if (!appQueue.length) return;
  if (queueMode === "auto") await topUpQueue();
  const position = index === 0 ? 0 : (latestState?.progress_ms || 0);
  try {
    await api("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        uris: appQueue.map((track) => track.uri),
        offset: 0,
        position_ms: position,
        device_id: browserDeviceId,
      }),
    });
    queueSyncFor = latestState?.uri || "";
    toast("Removed from the queue.");
  } catch (error) {
    toast(error.message, true);
  }
}

document.querySelector("#auto-queue-toggle")?.addEventListener("click", async () => {
  autoQueueEnabled = !autoQueueEnabled;
  localStorage.setItem(autoQueueKey, String(autoQueueEnabled));
  clearTimeout(mixEndTimer); mixEndTimer = null;
  prefetchedTracks = [];
  queueMode = autoQueueEnabled ? "auto" : "track";
  mixEnding = false;
  paintAutoQueue();
  if (!autoQueueEnabled) {
    toast("Auto queue off. New songs play on their own.");
    return;
  }
  try {
    const data = await api("/api/queue/current", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const filled = Array.isArray(data.entries) && data.entries.length;
    if (filled) {
      appQueue = data.entries;
      renderQueue();
    }
    toast(filled ? "Auto queue on. Model picks fill the queue." : "Auto queue on. It will fill the next song you play.");
    if (filled && latestState?.uri) await restartAutoQueue();
  } catch (error) {
    toast(error.message, true);
  }
});

/* ------------------------------------------------------- Mixes */
const tileStyle = (seed) => {
  const n = Math.abs(Number(seed) || 0);
  const pairs = [
    ["var(--cover-a)", "var(--cover-c)"],
    ["var(--cover-b)", "var(--cover-a)"],
    ["var(--cover-c)", "var(--cover-b)"],
  ];
  const [first, second] = pairs[n % pairs.length];
  return `--g1:color-mix(in srgb, ${first} 68%, #4080ff);--g2:color-mix(in srgb, ${second} 58%, #081026);`;
};
let mixesLoaded = false;

function sideItem({ label, meta, image, seed, onClick }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "side-item";
  const art = document.createElement("span");
  art.className = "side-item-art";
  if (image) {
    art.style.backgroundImage = `url("${image}")`;
    art.classList.add("has-image");
  } else {
    art.setAttribute("style", tileStyle(seed));
  }
  const copy = document.createElement("span");
  copy.className = "side-item-copy";
  const strong = document.createElement("strong");
  strong.textContent = label;
  const small = document.createElement("small");
  small.textContent = meta;
  copy.append(strong, small);
  button.append(art, copy);
  button.addEventListener("click", onClick);
  return button;
}

const MOOD_MIX_NAMES = {
  Upbeat: "Lift Me Up Mix",
  Chill: "Unwind Mix",
  Focus: "Focus Mix",
  Workout: "Workout Mix",
};

function normalizeMoodMix(mix) {
  const tracks = Array.isArray(mix.tracks) ? mix.tracks : [];
  const lead = tracks[0];
  return {
    ...mix,
    // Keep the visible identity tied to the actual mood and first queued
    // suggestion, even when older cached API data is still in the browser.
    name: MOOD_MIX_NAMES[mix.mood] || mix.name || "Your Mix",
    seed_name: lead?.name || mix.seed_name || "Model picks",
    seed_artist: lead?.artist || mix.seed_artist || "",
  };
}

function renderSideMixes(mixes) {
  const box = document.querySelector("#side-mixes");
  const count = document.querySelector("#side-mix-count");
  if (!box) return;
  if (count) count.textContent = mixes.length ? String(mixes.length) : "";
  box.replaceChildren();
  if (!mixes.length) {
    const empty = document.createElement("p");
    empty.className = "side-placeholder";
    empty.textContent = "No mixes yet.";
    box.append(empty);
    return;
  }
  mixes.forEach((mix) => {
    box.append(sideItem({
      label: mix.name,
      meta: `${mix.mood} · ${mix.tracks.length} tracks`,
      image: mix.tracks?.find((track) => track.art)?.art || mix.image,
      onClick: () => openMixDetail(mix),
    }));
  });
}

function renderMoodMixes(mixes) {
  const container = document.querySelector("#mood-mixes");
  const empty = document.querySelector("#playlists-empty");
  if (!container) return;
  container.replaceChildren();
  if (empty) empty.classList.toggle("hidden", mixes.length > 0);
  mixes.forEach((mix) => {
    const card = document.createElement("article");
    card.className = "playlist-card mix-card";
    card.dataset.mood = mix.mood || "";

    const cover = document.createElement("div");
    cover.className = "mix-cover";
    if (mix.image) {
      cover.style.backgroundImage = `url("${mix.image}")`;
      cover.classList.add("has-image");
    }
    const moodTag = document.createElement("span");
    moodTag.className = "mix-mood";
    moodTag.textContent = mix.mood || "Mix";
    const coverMeta = document.createElement("span");
    coverMeta.className = "mix-cover-meta";
    coverMeta.textContent = `${mix.tracks.length} tracks`;
    cover.append(moodTag, coverMeta);

    const body = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = mix.name;
    const meta = document.createElement("div");
    meta.className = "meta";
    const lead = mix.seed_name ? `${mix.seed_name} — ${mix.seed_artist || ""}` : "model picks";
    meta.textContent = `${mix.tracks.length} tracks · led by ${lead}`;
    body.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "card-actions";

    const play = document.createElement("button");
    play.type = "button";
    play.className = "btn small primary";
    play.innerHTML = `${iconUse("play")}Play`;
    play.addEventListener("click", () => playMix(mix));

    const open = document.createElement("button");
    open.type = "button";
    open.className = "btn small";
    open.innerHTML = `${iconUse("listen")}Open`;
    open.addEventListener("click", () => openMixDetail(mix));

    const form = document.createElement("form");
    form.method = "post";
    form.action = "/spotify/create-mood-playlist";
    [["csrf_token", csrf], ["mood", mix.mood]].forEach(([fieldName, value]) => {
      const field = document.createElement("input");
      field.type = "hidden"; field.name = fieldName; field.value = value; form.append(field);
    });
    const exportBtn = document.createElement("button");
    exportBtn.type = "submit";
    exportBtn.className = "btn small";
    exportBtn.innerHTML = `${iconUse("spotify")}Export`;
    form.append(exportBtn);

    actions.append(play, open, form);
    card.append(cover, body, actions);
    card.addEventListener("click", (event) => {
      if (event.target.closest("button, form, input, select, a")) return;
      openMixDetail(mix);
    });
    container.append(card);
  });
  renderSideMixes(mixes);
  hydrateMoodMixCovers(mixes);
}

function hydrateMoodMixCovers(mixes) {
  const indices = [...new Set(mixes.flatMap((mix) => (mix.tracks || []).slice(0, 4)
    .map((track) => track.index).filter(Number.isInteger)))];
  if (!indices.length) return;
  api("/api/catalog/art", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ indices }),
  }).then((data) => {
    const art = data.art || {};
    mixes.forEach((mix) => {
      const tracks = (mix.tracks || []).slice(0, 4);
      const images = tracks.map((track) => {
        const image = art[String(track.index)];
        if (image) track.art = image;
        return image;
      }).filter(Boolean);
      const cover = document.querySelector(`.mix-card[data-mood="${mix.mood}"] .mix-cover`);
      if (!cover || !images.length) return;
      cover.querySelector(".mix-cover-mosaic")?.remove();
      const mosaic = document.createElement("div");
      mosaic.className = "mix-cover-mosaic";
      mosaic.setAttribute("aria-hidden", "true");
      mosaic.style.gridTemplateColumns = `repeat(${Math.min(2, images.length)}, minmax(0, 1fr))`;
      images.forEach((image) => {
        const tile = document.createElement("div");
        const img = document.createElement("img");
        img.src = image;
        img.alt = "";
        img.decoding = "async";
        img.referrerPolicy = "no-referrer";
        tile.append(img);
        mosaic.append(tile);
      });
      cover.prepend(mosaic);
    });
    renderSideMixes(mixes);
  }).catch(() => {});
}

async function playList(tracks, label, mode) {
  if (playInFlight) return;
  const playable = (tracks || []).filter((track) => track.uri);
  if (!playable.length) {
    toast("There are no playable Spotify matches in this list.", true);
    return;
  }
  clearTimeout(mixEndTimer); mixEndTimer = null;
  prefetchedTracks = [];
  playInFlight = true;
  activateBrowserAudio();
  try {
    await initialiseBrowserPlayer();
    if (!browserDeviceId) throw new Error("The web player is still connecting. Try again in a moment.");
    await api("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        uris: playable.map((track) => track.uri),
        offset: 0,
        device_id: browserDeviceId,
      }),
    });
    appQueue = playable.map((track) => ({ ...track }));
    queueMode = mode;
    mixEnding = false;
    renderQueue();
    queueSyncFor = latestState?.uri || "";
    toast(`Playing ${label}.`);
    await refreshPlayer();
  } catch (error) {
    toast(error.message, true);
  } finally {
    playInFlight = false;
  }
}

function playMix(mix, startIndex = 0) {
  return playList((mix.tracks || []).slice(startIndex), mix.name, "mix");
}

async function loadMixes(force = false) {
  if (mixesLoaded && !force) return;
  document.querySelector("#playlists-empty")?.classList.add("hidden");
  const status = document.querySelector("#mixes-status");
  if (status) status.textContent = "Building mixes from your listening...";
  try {
    const { mixes: responseMixes } = await api("/api/mood-mixes");
    const mixes = (responseMixes || []).map(normalizeMoodMix);
    mixesLoaded = true;
    renderMoodMixes(mixes);
    if (status) status.textContent = mixes.length
      ? `${mixes.length} mixes ready \u00b7 refreshed with your feedback`
      : "No mixes available.";
  } catch (error) {
    if (status) status.textContent = error.message;
  }
}

let detailPlayPrimary = null;
let detailPlaySecondary = null;
const detailArtRequested = new Set();

/* Detail lists (especially fixed mood mixes) can arrive without covers; fill
   them in from the catalogue once the list is already on screen. */
function hydrateDetailArt(container) {
  const targets = [...container.querySelectorAll(".row-art[data-art-index]")]
    .filter((el) => !el.querySelector("img") && !detailArtRequested.has(el.dataset.artIndex));
  if (!targets.length) return;
  const indices = [...new Set(targets.map((el) => Number(el.dataset.artIndex)))];
  indices.forEach((index) => detailArtRequested.add(String(index)));
  api("/api/catalog/art", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ indices }),
  }).then((data) => {
    const art = data.art || {};
    targets.forEach((el) => {
      if (el.querySelector("img")) return;
      const url = art[el.dataset.artIndex];
      if (!url) return;
      const img = document.createElement("img");
      img.src = url;
      img.alt = "";
      img.decoding = "async";
      img.referrerPolicy = "no-referrer";
      img.addEventListener("error", () => img.remove());
      el.append(img);
    });
  }).catch(() => {
    indices.forEach((index) => detailArtRequested.delete(String(index)));
  });
}

function renderDetailRows(container, tracks, tagFor, onPlay, onAdd = null) {
  if (!container) return;
  container.replaceChildren();
  tracks.forEach((track, index) => {
    const row = document.createElement("div");
    row.className = "row";
    const num = document.createElement("div");
    num.className = "row-num";
    num.textContent = String(index + 1).padStart(2, "0");
    const art = document.createElement("div");
    art.className = "row-art";
    art.setAttribute("style", tileStyle((track.index ?? index) + index));
    if (track.art) {
      const img = document.createElement("img");
      img.src = track.art;
      img.alt = "";
      img.decoding = "async";
      img.referrerPolicy = "no-referrer";
      img.addEventListener("error", () => img.remove());
      art.append(img);
    } else if (Number.isInteger(track.index)) {
      art.dataset.artIndex = String(track.index);
    }
    const main = document.createElement("div");
    main.className = "row-main";
    const name = document.createElement("strong");
    name.textContent = track.name;
    const artist = document.createElement("span");
    artist.textContent = track.artist || "";
    main.append(name, artist);
    const genre = document.createElement("div");
    genre.className = "row-genre";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = tagFor(track, index);
    genre.append(tag);
    const actions = document.createElement("div");
    actions.className = "row-actions";
    const play = document.createElement("button");
    play.type = "button";
    play.className = "row-play";
    play.setAttribute("aria-label", `Play ${track.name}`);
    play.innerHTML = iconUse("play");
    play.addEventListener("click", () => onPlay(index));
    actions.append(play);
    if (onAdd && track.uri) {
      const add = document.createElement("button");
      add.type = "button";
      add.className = "row-add";
      add.setAttribute("aria-label", `Add ${track.name} to playlist`);
      add.innerHTML = `${iconUse("plus")}<span>Add</span>`;
      add.addEventListener("click", async () => {
        add.disabled = true;
        const ok = await onAdd(track, add);
        if (ok) {
          add.classList.add("added");
          add.innerHTML = `${iconUse("listen")}<span>Added</span>`;
        } else {
          add.disabled = false;
        }
      });
      actions.append(add);
    }
    row.append(num, art, main, genre, actions);
    container.append(row);
  });
}

function showDetail(config) {
  const panel = document.querySelector("#detail-panel");
  if (!panel) return;
  document.querySelector("#detail-title").textContent = config.title;
  document.querySelector("#detail-meta").textContent = config.meta || "";
  document.querySelector("#detail-desc").textContent = config.description || "";
  const hero = document.querySelector("#detail-hero");
  const heroArt = document.querySelector("#detail-hero-art");
  if (hero && heroArt) {
    hero.hidden = !config.image;
    if (config.image) heroArt.src = config.image;
    const heroLabel = document.querySelector("#detail-hero-label");
    if (heroLabel) heroLabel.textContent = config.heroLabel || "";
  }
  const status = document.querySelector("#detail-status");
  if (status) {
    status.textContent = config.status || "";
    status.classList.toggle("hidden", !config.status);
  }
  document.querySelector("#detail-primary-title").textContent = config.primaryTitle || "Tracks";
  document.querySelector("#detail-primary-count").textContent = `${(config.primary || []).length} tracks`;
  renderDetailRows(
    document.querySelector("#detail-primary"),
    config.primary || [],
    config.tagFor || (() => "model pick"),
    (index) => config.onPlayPrimary?.(index),
  );
  const secondary = config.secondary || [];
  const secondaryWrap = document.querySelector("#detail-secondary-wrap");
  if (secondaryWrap) secondaryWrap.hidden = secondary.length === 0;
  if (secondary.length) {
    document.querySelector("#detail-secondary-title").textContent = config.secondaryTitle || "Model recommendations";
    document.querySelector("#detail-secondary-count").textContent = `${secondary.length} tracks`;
    renderDetailRows(
      document.querySelector("#detail-secondary"),
      secondary,
      config.tagForSecondary || (() => "model pick"),
      (index) => config.onPlaySecondary?.(index),
      config.onAddSecondary || null,
    );
  }
  const exportForm = document.querySelector("#detail-export");
  if (exportForm) exportForm.hidden = !config.mood;
  if (config.mood) document.querySelector("#detail-mood").value = config.mood;
  const primaryButton = document.querySelector("#detail-play-primary");
  primaryButton.textContent = config.playPrimaryLabel || "Play";
  primaryButton.hidden = !config.onPlayPrimaryAll;
  detailPlayPrimary = config.onPlayPrimaryAll || null;
  const secondaryButton = document.querySelector("#detail-play-secondary");
  secondaryButton.hidden = !config.onPlaySecondaryAll;
  secondaryButton.textContent = config.playSecondaryLabel || "Play recommendations";
  detailPlaySecondary = config.onPlaySecondaryAll || null;
  panel.hidden = false;
  scrollMainTo(panel, 12);
}

function openMixDetail(mix) {
  showDetail({
    title: mix.name,
    meta: `${mix.tracks.length} tracks · ${mix.history_seeded ? "from your listening" : "model picks"}`,
    description: mix.description,
    image: mix.tracks?.find((track) => track.art)?.art || mix.image,
    heroLabel: `${mix.mood} · ${mix.seed_name} — ${mix.seed_artist}`,
    primaryTitle: "Tracks",
    primary: mix.tracks,
    tagFor: (track) => (track.anchor ? "anchor" : "model pick"),
    onPlayPrimary: (index) => playMix(mix, index),
    onPlayPrimaryAll: () => playMix(mix),
    playPrimaryLabel: "Play mix",
    mood: mix.mood,
  });
}

const playlistPicks = new Map();

function openPlaylistDetail(playlist, data = null) {
  const playlistId = playlist.id || data?.playlist_id;
  if (!playlistId) return;

  if (!data) {
    showDetail({
      title: playlist.name,
      meta: `${Number(playlist.tracks) || 0} tracks · loading model picks…`,
      description: playlist.description || "Loaded automatically from your Spotify playlists.",
      primaryTitle: "Playlist",
      primary: [],
      status: "Building model recommendations for this playlist…",
    });
    api(`/api/playlist/${encodeURIComponent(playlistId)}/recommendations`)
      .then((payload) => openPlaylistDetail(playlist, payload))
      .catch((error) => {
        const status = document.querySelector("#detail-status");
        if (status) { status.textContent = error.message; status.classList.remove("hidden"); }
      });
    return;
  }

  const name = data.name || playlist.name;
  const recommendations = data.recommendations || [];
  playlistPicks.set(playlistId, recommendations.length);
  const cardPicks = document.querySelector(`.playlist-card[data-playlist-id="${playlistId}"] .playlist-picks`);
  if (cardPicks) {
    cardPicks.textContent = recommendations.length
      ? `${recommendations.length} picks to add`
      : "No new picks right now";
  }
  showDetail({
    title: name,
    meta: `${data.tracks.length} tracks · ${recommendations.length} model picks to add`,
    description: data.description || "Loaded automatically from your Spotify playlists.",
    image: data.image,
    heroLabel: `${data.total_tracks || data.tracks.length} tracks · Spotify playlist`,
    primaryTitle: "Playlist",
    primary: data.tracks,
    tagFor: (track) => (track.index == null ? "not in catalogue" : "in catalogue"),
    onPlayPrimary: (index) => playList(data.tracks.slice(index), name, "playlist"),
    onPlayPrimaryAll: data.tracks.length ? () => playList(data.tracks, name, "playlist") : null,
    playPrimaryLabel: "Play playlist",
    secondaryTitle: "Recommended to add",
    secondary: recommendations,
    tagForSecondary: () => "model pick",
    onPlaySecondary: (index) => playList(recommendations.slice(index), `${name} picks`, "playlist"),
    onPlaySecondaryAll: recommendations.length ? () => playList(recommendations, `${name} picks`, "playlist") : null,
    playSecondaryLabel: "Play picks",
    onAddSecondary: async (track) => {
      try {
        await api("/api/playlist/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ playlist_id: playlistId, uri: track.uri }),
        });
        playlistPicks.set(playlistId, Math.max(0, (playlistPicks.get(playlistId) || 1) - 1));
        const card = document.querySelector(`.playlist-card[data-playlist-id="${playlistId}"] .playlist-picks`);
        if (card) card.textContent = `${playlistPicks.get(playlistId)} picks to add`;
        toast(`Added “${track.name}” to ${name}.`);
        return true;
      } catch (error) {
        toast(error.message, true);
        return false;
      }
    },
  });
}

let playlistsLoaded = false;

function renderSidePlaylists(playlists) {
  const box = document.querySelector("#side-playlists");
  const count = document.querySelector("#side-playlist-count");
  if (!box) return;
  if (count) count.textContent = playlists.length ? String(playlists.length) : "";
  box.replaceChildren();
  if (!playlists.length) {
    const empty = document.createElement("p");
    empty.className = "side-placeholder";
    empty.textContent = connected ? "No Spotify playlists found." : "Connect Spotify to see playlists.";
    box.append(empty);
    return;
  }
  playlists.forEach((playlist, index) => {
    box.append(sideItem({
      label: playlist.name,
      meta: `${Number(playlist.tracks) || 0} tracks`,
      image: playlist.image,
      seed: index,
      onClick: () => openPlaylistDetail(playlist),
    }));
  });
}

function renderPlaylistCards(playlists) {
  const grid = document.querySelector("#playlist-grid");
  if (!grid) return;
  grid.replaceChildren();
  playlists.forEach((playlist, index) => {
    const card = document.createElement("article");
    card.className = "playlist-card playlist-source-card";
    card.dataset.playlistId = playlist.id;

    const mosaic = document.createElement("div");
    mosaic.className = "mosaic playlist-cover";
    if (playlist.image) {
      mosaic.style.backgroundImage = `url("${playlist.image}")`;
      mosaic.classList.add("has-image");
    } else {
      [0, 1, 2, 3].forEach((n) => {
        const span = document.createElement("span");
        span.setAttribute("style", tileStyle(index + n * 7));
        mosaic.append(span);
      });
    }

    const body = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = playlist.name;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${Number(playlist.tracks) || 0} tracks`;
    const picks = document.createElement("span");
    picks.className = "playlist-picks";
    picks.textContent = "Open for model picks";
    body.append(name, meta, picks);

    const actions = document.createElement("div");
    actions.className = "card-actions";
    const open = document.createElement("button");
    open.type = "button";
    open.className = "btn small primary";
    open.innerHTML = `${iconUse("plus")}See picks`;
    open.addEventListener("click", () => openPlaylistDetail(playlist));
    actions.append(open);

    card.append(mosaic, body, actions);
    card.addEventListener("click", (event) => {
      if (event.target.closest("button, form, input, select, a")) return;
      openPlaylistDetail(playlist);
    });
    grid.append(card);
  });
  renderSidePlaylists(playlists);
}

async function loadPlaylists(force = false) {
  const grid = document.querySelector("#playlist-grid");
  const status = document.querySelector("#playlists-status");
  if (!connected || !grid) return;
  if (playlistsLoaded && !force) return;
  if (status) status.textContent = "Loading your Spotify playlists…";
  try {
    const { playlists, can_edit_public: canEditPublic } = await api("/api/spotify/playlists");
    playlistsLoaded = true;
    renderPlaylistCards(playlists);
    if (status) {
      const base = playlists.length
        ? `${playlists.length} playlists loaded · open one to see model picks you can add`
        : "No Spotify playlists found on this account.";
      status.textContent = canEditPublic ? base : `${base} · Reconnect Spotify once to add picks to public playlists.`;
    }
  } catch (error) {
    if (status) status.textContent = error.message;
  }
}

document.querySelector("#refresh-playlists")?.addEventListener("click", () => loadPlaylists(true));

document.querySelector("#detail-close")?.addEventListener("click", () => {
  const panel = document.querySelector("#detail-panel");
  if (panel) panel.hidden = true;
  scrollMainTo(document.querySelector(".mixes-section"), 0);
});
document.querySelector("#detail-play-primary")?.addEventListener("click", () => detailPlayPrimary?.());
document.querySelector("#detail-play-secondary")?.addEventListener("click", () => detailPlaySecondary?.());

document.querySelector("#refresh-mixes")?.addEventListener("click", () => loadMixes(true));

/* ------------------------------------------------------- Init */
renderNowHero();
if (!connected) {
  const title = document.querySelector("#player-title");
  const artist = document.querySelector("#player-artist");
  if (title) title.textContent = "Nothing playing";
  if (artist) artist.textContent = "Pick a song to start";
}
refreshLearning();
restoreQueue();
renderQueue();
paintAutoQueue();
loadMixes();
if (connected) {
  initialiseBrowserPlayer(); refreshPlayer();
  loadPlaylists();
  setInterval(refreshPlayer, 4000);
}
