/* TuneTwin visualizer + intro.
 *
 * A single lightweight 2D canvas (#fx-viz) draws an equalizer behind the UI:
 * big and centred while the intro is open, a quiet strip along the bottom
 * afterwards. Bars follow the track's cover colour (--cover-b on #app) and
 * jump when music is playing (#now-hero.is-playing). No libraries; it runs at
 * a capped frame rate and pauses when the tab is hidden.
 */
(function () {
  "use strict";

  const root = document.documentElement;
  const app = document.getElementById("app");
  const intro = document.getElementById("intro");
  const canvas = document.getElementById("fx-viz");
  if (!app) return;

  /* ---------------------------------------------------------------- intro */
  const INTRO_KEY = "tunetwin-intro-seen";
  let introOpen = root.classList.contains("intro-active");
  const lockables = Array.from(document.querySelectorAll("#app .sidebar, #app .window"));

  function setLocked(locked) {
    lockables.forEach((el) => {
      try { el.inert = locked; } catch (error) { /* inert unsupported: harmless */ }
    });
  }

  function closeIntro(animate) {
    if (!introOpen) return;
    introOpen = false;
    try { sessionStorage.setItem(INTRO_KEY, "1"); } catch (error) { /* private mode */ }
    setLocked(false);
    root.classList.remove("intro-active");
    if (!intro) return;
    if (!animate) { intro.hidden = true; return; }
    intro.classList.add("is-leaving");
    setTimeout(() => { intro.hidden = true; }, 800);
  }

  if (introOpen) {
    // Login errors/successes are flashed into the page: show them, not the intro.
    if (document.querySelector(".messages .message")) {
      closeIntro(false);
    } else {
      setLocked(true);
      document.getElementById("intro-start")?.addEventListener("click", () => closeIntro(true));
      window.addEventListener("keydown", (event) => {
        if (!introOpen) return;
        if (event.key === "Escape" || event.key === "Enter" || event.key === " ") {
          if (event.target && event.target.closest && event.target.closest("a")) return;
          event.preventDefault();
          closeIntro(true);
        }
      });
      setTimeout(() => document.getElementById("intro-start")?.focus({ preventScroll: true }), 400);
    }
  }

  /* ----------------------------------------------------------- visualizer */
  if (!canvas || !canvas.getContext) return;
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

  function lerp(a, b, k) { return a + (b - a) * k; }
  function damp(current, target, rate, dt) { return current + (target - current) * (1 - Math.exp(-rate * dt)); }

  function bar(x, y, w, h) {
    if (h <= 0.5) return;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, Math.min(w / 2, 6));
    else ctx.rect(x, y, w, h);
    ctx.fill();
  }

  let zone = introOpen ? 1 : 0;
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

    zone = damp(zone, introOpen ? 1 : 0, 2.6, dt);
    const wide = W / H >= 1.15;
    // Intro layout: right half on wide screens, upper area on phones.
    const intro = wide
      ? { x0: W * 0.5, x1: W * 0.95, base: H * 0.7, maxH: H * 0.5 }
      : { x0: W * 0.07, x1: W * 0.93, base: H * 0.38, maxH: H * 0.3 };
    const ambient = { x0: W * 0.02, x1: W * 0.98, base: H, maxH: H * 0.17 };
    const x0 = lerp(ambient.x0, intro.x0, zone);
    const x1 = lerp(ambient.x1, intro.x1, zone);
    const base = lerp(ambient.base, intro.base, zone);
    const maxH = lerp(ambient.maxH, intro.maxH, zone);

    const gain = playing ? 1 : (introOpen ? 0.62 : 0.3);
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
      const x = x0 + i * step + (step - width) / 2;
      bar(x, base - h, width, h);
    }

    // soft reflection under the intro bars
    if (zone > 0.05) {
      ctx.globalAlpha = 0.16 * zone;
      for (let i = 0; i < barCount; i++) {
        const h = Math.max(4, heights[i] * maxH) * 0.55;
        const x = x0 + i * step + (step - width) / 2;
        bar(x, base + 4, width, h);
      }
      ctx.globalAlpha = 1;
    }
  }

  let timer = 0;
  function loop(now) {
    timer = requestAnimationFrame(loop);
    if (document.hidden) { last = now; return; }
    // ~30 fps while the intro is open, ~20 fps as a quiet background.
    if (now - lastDraw < (introOpen ? 33 : 50)) return;
    lastDraw = now;
    draw(now);
  }

  resize();
  syncAccent();
  window.addEventListener("resize", () => { resize(); if (reduceMotion) draw(performance.now()); });
  setInterval(syncAccent, 900);
  canvas.dataset.ready = "1";
  root.classList.add("viz-ready");
  if (reduceMotion) draw(performance.now());
  else timer = requestAnimationFrame(loop);
})();
