/* Resona equalizer.
 *
 * A single lightweight 2D canvas (#fx-viz) draws a quiet equalizer strip along
 * the bottom of the app. Bars follow the track's cover colour (--cover-b on
 * #app) and jump when music is playing (#now-hero.is-playing). No libraries;
 * it runs at a capped frame rate and pauses while the tab is hidden or the
 * scroll hero (static/hero.js) still covers the app.
 */
(function () {
  "use strict";

  const root = document.documentElement;
  const app = document.getElementById("app");
  const canvas = document.getElementById("fx-viz");
  if (!app || !canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isMobile = () => window.innerWidth < 720;

  let W = 0;
  let H = 0;
  let barCount = 0;
  let heights = new Float32Array(0);
  let accent = "#e84393";

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const wanted = isMobile() ? 26 : 44;
    if (wanted !== barCount) { barCount = wanted; heights = new Float32Array(barCount); }
  }

  function syncAccent() {
    try {
      const value = getComputedStyle(app).getPropertyValue("--cover-b").trim();
      if (value && value.indexOf("var(") !== 0) accent = value;
    } catch (error) { /* keep the previous colour */ }
  }

  function damp(current, target, rate, dt) { return current + (target - current) * (1 - Math.exp(-rate * dt)); }

  function bar(x, y, w, h) {
    if (h <= 0.5) return;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, Math.min(w / 2, 6));
    else ctx.rect(x, y, w, h);
    ctx.fill();
  }

  let playing = false;
  let playingCheckAt = 0;
  let last = performance.now();
  let lastDraw = 0;

  function draw(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    const t = now / 1000;

    if (now - playingCheckAt > 300) {
      playingCheckAt = now;
      playing = Boolean(document.querySelector("#now-hero.is-playing"));
    }

    const x0 = W * 0.02;
    const x1 = W * 0.98;
    const base = H;
    const maxH = H * 0.17;
    const gain = playing ? 1 : 0.3;
    const beat = 0.5 + 0.5 * Math.sin(t * Math.PI * 2 * 2.1);
    const step = (x1 - x0) / barCount;
    const width = step * 0.58;

    ctx.clearRect(0, 0, W, H);
    const fill = ctx.createLinearGradient(0, base - maxH, 0, base);
    fill.addColorStop(0, "rgba(255,255,255,0.92)");
    fill.addColorStop(1, accent);
    ctx.fillStyle = fill;

    for (let i = 0; i < barCount; i++) {
      const wave = 0.5 + 0.5 * Math.sin(t * 2.6 + i * 0.7) * Math.sin(t * 1.3 + i * 0.31);
      const edge = Math.sin(Math.PI * (i + 0.5) / barCount); // taller in the middle
      let target = (0.1 + wave * 0.9 * (0.45 + 0.55 * edge)) * gain;
      if (playing) target += beat * 0.18 * Math.pow(Math.sin(i * 0.5 + 1.2), 2);
      heights[i] = damp(heights[i], Math.min(target, 1), 11, dt);
      const h = Math.max(4, heights[i] * maxH);
      bar(x0 + i * step + (step - width) / 2, base - h, width, h);
    }
  }

  function loop(now) {
    requestAnimationFrame(loop);
    if (document.hidden || root.classList.contains("hero-active")) { last = now; return; }
    if (now - lastDraw < 50) return; // ~20 fps is plenty for a background strip
    lastDraw = now;
    draw(now);
  }

  resize();
  syncAccent();
  window.addEventListener("resize", () => { resize(); if (reduceMotion) draw(performance.now()); });
  setInterval(syncAccent, 900);
  root.classList.add("viz-ready");
  if (reduceMotion) draw(performance.now());
  else requestAnimationFrame(loop);
})();
