/* Resona scroll hero.
 *
 * A full-screen layer above the app with its own scroll track. Scrolling plays
 * a short story in 3D (Three.js, loaded from a CDN only when the hero shows):
 *
 *   0.00  RESONA title with glossy shapes floating around it
 *   0.17  the shapes clear away and one sphere ("your song") takes the centre
 *   0.39  eight small spheres settle into an orbit around it (eight matches)
 *   0.61  liked spheres move closer, skipped ones drift away (it learns)
 *   0.81  call to action, then the camera dives in and the app fades up
 *
 * Kept light on purpose: one draw call per object (17 meshes + 1 point cloud),
 * no shadows or post-processing, capped pixel ratio, 30 fps on phones, paused
 * while the tab is hidden, and the WebGL context is destroyed once the
 * listener reaches the app.
 */

const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js";

const root = document.documentElement;
const hero = document.getElementById("hero");
const stage = document.getElementById("hero-stage");
const app = document.getElementById("app");
const canvas = document.getElementById("hero-gl");

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;

if (hero && stage && root.classList.contains("hero-active")) {
  start();
} else if (hero) {
  hero.hidden = true;
}

function start() {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mobile = window.innerWidth < 720 || window.matchMedia("(pointer: coarse)").matches;
  const pointer = { x: 0, y: 0 };
  const steps = Array.from(stage.querySelectorAll("[data-from]")).map((el) => ({
    el, from: Number(el.dataset.from), to: Number(el.dataset.to),
  }));

  let raw = 0;        // real scroll progress 0..1
  let sp = 0;         // smoothed progress that drives the animation
  let done = false;
  let stopped = false;
  let gl = null;
  let auto = null;    // running "Enter app" scroll animation

  try { app.inert = true; } catch (error) { /* older browsers */ }
  hero.focus({ preventScroll: true });

  // The scroll length is measured once (and on resize), never inside the
  // frame loop, so reading progress never forces a layout.
  let maxScroll = 1;
  function measure() { maxScroll = Math.max(1, hero.scrollHeight - hero.clientHeight); }
  function readScroll() { raw = clamp01(hero.scrollTop / maxScroll); }

  // Only touch the DOM when a value really changes. --hero-reveal lives on
  // <html>, so writing it every frame would restyle the whole app.
  const written = new Map();
  function write(el, key, value) {
    const id = key + (el === root ? "@root" : el === stage ? "@stage" : "");
    if (written.get(id) === value) return;
    written.set(id, value);
    el.style.setProperty(key, value);
  }
  let covered = null;
  function applyDom() {
    const rv = smooth(0.9, 0.995, sp);
    write(stage, "--p", sp.toFixed(3));
    write(stage, "--pa", smooth(0.02, 0.16, sp).toFixed(3));
    write(stage, "--rv", rv.toFixed(3));
    write(root, "--hero-reveal", rv.toFixed(3));
    // Until the reveal starts, the app underneath is not painted at all.
    const cover = rv <= 0;
    if (cover !== covered) { covered = cover; root.classList.toggle("hero-cover", cover); }
    for (const s of steps) {
      const fadeIn = smooth(s.from, s.from + 0.05, sp);
      const fadeOut = 1 - smooth(s.to - 0.05, s.to, sp);
      const o = Math.min(fadeIn, fadeOut);
      const dy = (1 - fadeIn) * 40 - (1 - fadeOut) * 40;
      const op = o.toFixed(2);
      if (s.op === op) continue;
      s.op = op;
      s.el.style.opacity = op;
      s.el.style.translate = `0 ${dy.toFixed(0)}px`;
      s.el.style.visibility = o > 0 ? "visible" : "hidden";
      s.el.style.pointerEvents = o > 0.6 ? "auto" : "none";
    }
  }

  function finish() {
    if (done) return;
    done = true;
    root.style.setProperty("--hero-reveal", "1");
    root.classList.remove("hero-cover");
    hero.classList.add("is-done");
    try { app.inert = false; } catch (error) { /* ignore */ }
    setTimeout(() => {
      stopped = true;
      root.classList.remove("hero-active");
      hero.hidden = true;
      if (gl) { gl.dispose(); gl = null; }
    }, 520);
  }

  function goToApp() {
    if (done) return;
    if (reduceMotion) { finish(); return; }
    const from = hero.scrollTop;
    const to = maxScroll;
    const duration = Math.max(900, 2400 * (1 - raw));
    const t0 = performance.now();
    auto = (now) => {
      const k = clamp01((now - t0) / duration);
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      hero.scrollTop = from + (to - from) * e;
      return k < 1;
    };
  }

  hero.addEventListener("scroll", readScroll, { passive: true });
  hero.querySelectorAll("[data-hero-enter]").forEach((btn) => btn.addEventListener("click", goToApp));
  window.addEventListener("keydown", (event) => {
    if (done) return;
    if (event.key === "Escape") { event.preventDefault(); finish(); }
    else if (event.key === "Enter" && !(event.target.closest && event.target.closest("a, button"))) {
      event.preventDefault();
      goToApp();
    }
  });
  window.addEventListener("pointermove", (event) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = (event.clientY / window.innerHeight) * 2 - 1;
  }, { passive: true });
  window.addEventListener("resize", () => { measure(); readScroll(); if (gl) gl.resize(); });

  // Adaptive quality: if frames come in slowly, drop the idle floating motion
  // and render at a lower resolution, only when the scroll position changes.
  let lowPower = mobile && (navigator.hardwareConcurrency || 8) <= 4;
  let slowFrames = 0;
  let sampled = 0;
  let lastRenderedSp = -1;
  let last = performance.now();
  let lastFrame = 0;
  function tick(now) {
    if (stopped) return;
    requestAnimationFrame(tick);
    if (document.hidden) { last = now; return; }
    const minGap = mobile || lowPower ? 32 : 15; // ~30 fps on phones / slow devices
    if (now - lastFrame < minGap) return;
    const gap = now - lastFrame;
    lastFrame = now;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (gl && !lowPower && sampled < 90) {
      sampled++;
      if (gap > 45) slowFrames++;
      if (sampled === 90 && slowFrames > 30) { lowPower = true; gl.setLowPower(); }
    }
    if (auto && !auto(now)) auto = null;
    readScroll();
    sp = reduceMotion ? raw : sp + (raw - sp) * (1 - Math.exp(-dt * 7));
    if (Math.abs(raw - sp) < 0.0004) sp = raw;
    applyDom();
    if (gl) {
      const still = reduceMotion || lowPower;
      if (!still || Math.abs(sp - lastRenderedSp) > 0.0003) {
        gl.render(still ? 0 : now / 1000, sp, still ? { x: 0, y: 0 } : pointer);
        lastRenderedSp = sp;
      }
    }
    if (!done && raw >= 0.995 && sp >= 0.985) finish();
  }

  measure();
  readScroll();
  applyDom();
  requestAnimationFrame(tick);

  if (canvas) {
    createScene(canvas, { mobile })
      .then((scene) => {
        if (done) { scene.dispose(); return; }
        gl = scene;
        gl.render(0, sp, pointer);
        canvas.classList.add("is-ready");
      })
      .catch((error) => console.warn("Resona hero: 3D scene unavailable,", error));
  }
}

async function createScene(canvas, { mobile }) {
  const THREE = await import(THREE_URL);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: !mobile, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1 : 1.25));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  const disposables = [];
  const keep = (x) => { disposables.push(x); return x; };

  const accentHex = (() => {
    try {
      const v = getComputedStyle(document.getElementById("app")).getPropertyValue("--cover-b").trim();
      return v && v.startsWith("#") ? v : "#e84393";
    } catch (error) { return "#e84393"; }
  })();
  const accent = new THREE.Color(accentHex);

  // Matcap materials: the lighting is painted once into a small sphere image
  // (a 2D canvas), so the GPU only does one texture lookup per pixel. No
  // lights, no environment map, no physically based shading.
  function matcap(base, rimLeft, rimRight, highlight) {
    const size = 256;
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const g = c.getContext("2d");
    const r = size / 2;
    let grad = g.createRadialGradient(r * 0.85, r * 0.8, r * 0.1, r, r, r);
    grad.addColorStop(0, base[0]); grad.addColorStop(0.7, base[1]); grad.addColorStop(1, base[2]);
    g.fillStyle = grad; g.fillRect(0, 0, size, size);
    const band = (x, color, alpha) => {
      grad = g.createRadialGradient(x, r, r * 0.55, x, r, r * 1.05);
      grad.addColorStop(0, "rgba(0,0,0,0)"); grad.addColorStop(0.75, color); grad.addColorStop(1, "rgba(0,0,0,0)");
      g.globalAlpha = alpha; g.fillStyle = grad; g.fillRect(0, 0, size, size); g.globalAlpha = 1;
    };
    band(r * 1.25, rimLeft, 0.95);
    band(r * 0.75, rimRight, 0.85);
    grad = g.createRadialGradient(r * 0.68, r * 0.55, 0, r * 0.68, r * 0.55, r * 0.42);
    grad.addColorStop(0, highlight); grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad; g.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return keep(tex);
  }
  const chromeCap = matcap(["#3a3344", "#121018", "#050407"], accentHex, "#6f7dff", "rgba(255,255,255,0.95)");
  const pearlCap = matcap(["#ffffff", "#e9dcef", "#8f7f9c"], accentHex, "#9fb0ff", "rgba(255,255,255,1)");
  const inkCap = matcap(["#2a2433", "#120f17", "#050407"], "#3d3346", "#2e3550", "rgba(255,255,255,0.55)");
  const tintCap = matcap([accentHex, accentHex, "#3a0f22"], "#ffd1e6", "#7b6dff", "rgba(255,255,255,0.9)");

  const chrome = keep(new THREE.MeshMatcapMaterial({ matcap: chromeCap }));
  const pearl = keep(new THREE.MeshMatcapMaterial({ matcap: pearlCap }));
  const ink = keep(new THREE.MeshMatcapMaterial({ matcap: inkCap }));
  const faceted = keep(new THREE.MeshMatcapMaterial({ matcap: chromeCap, flatShading: true }));
  const tint = keep(new THREE.MeshMatcapMaterial({ matcap: tintCap }));
  const glow = keep(new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.9 }));
  const glowFaint = keep(new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 }));

  const seg = mobile ? [32, 22] : [48, 32];
  const mesh = (geometry, material) => { keep(geometry); const m = new THREE.Mesh(geometry, material); scene.add(m); return m; };

  const core = mesh(new THREE.SphereGeometry(1, seg[0], seg[1]), chrome);
  const donut = mesh(new THREE.TorusGeometry(1, 0.42, mobile ? 20 : 32, mobile ? 56 : 96), pearl);
  const satGeo = keep(new THREE.SphereGeometry(0.2, 20, 14));
  const satMats = [chrome, tint, pearl, chrome, tint, pearl, chrome, tint];
  const sats = satMats.map((mat) => { const m = new THREE.Mesh(satGeo, mat); scene.add(m); return m; });
  const accents = [
    mesh(new THREE.CapsuleGeometry(0.16, 0.7, 6, 16), ink),
    mesh(new THREE.IcosahedronGeometry(0.36, 0), faceted),
    mesh(new THREE.TorusGeometry(0.28, 0.09, 16, 48), tint),
    mesh(new THREE.BoxGeometry(0.42, 0.42, 0.42), ink),
    mesh(new THREE.IcosahedronGeometry(0.26, 0), faceted),
    mesh(new THREE.CapsuleGeometry(0.12, 0.5, 6, 16), pearl),
  ];

  // The orbit: a thin glowing ring plus a faint outer ring, tilted toward us.
  const orbit = new THREE.Group();
  scene.add(orbit);
  const tilt = new THREE.Euler(0.42, 0, 0.2);
  orbit.rotation.copy(tilt);
  const ringGeo = keep(new THREE.TorusGeometry(1, 0.008, 6, 220));
  const ring = new THREE.Mesh(ringGeo, glow); ring.rotation.x = Math.PI / 2; orbit.add(ring);
  const ring2 = new THREE.Mesh(ringGeo, glowFaint); ring2.rotation.x = Math.PI / 2; orbit.add(ring2);

  // Dust
  const dustCount = mobile ? 160 : 360;
  const dustPos = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i++) {
    const r = 3 + Math.random() * 9;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    dustPos.set([r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph) * 0.7, r * Math.sin(ph) * Math.sin(th) - 2], i * 3);
  }
  const dustGeo = keep(new THREE.BufferGeometry());
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  const dustMat = keep(new THREE.PointsMaterial({ color: 0xffffff, size: mobile ? 0.04 : 0.032, transparent: true, opacity: 0.55, depthWrite: false }));
  const dust = new THREE.Points(dustGeo, dustMat);
  scene.add(dust);

  // Layout, recomputed on resize. Positions are written in "screen units"
  // (x: -1..1 across the view, y: -1..1 down the view) and mapped to world space.
  const CAM_Z = 12;
  const L = { vw: 1, vh: 1, u: 1, center: new THREE.Vector3(), R: 2, wide: true };
  const scatter = {
    core: [-0.58, -0.44, 0.6, 0.72],
    donut: [0.6, 0.44, -1.6, 1.1],
    accents: [[-0.84, 0.56, -1], [0.86, -0.52, 0.4], [-0.32, 0.8, -2.2], [0.28, -0.82, -1], [0.96, 0.06, -2.6], [-0.95, -0.08, -0.4]],
    sats: sats.map((_, i) => { const a = (i / 8) * Math.PI * 2 + 0.35; return [Math.cos(a) * 0.82, Math.sin(a) * 0.74, -2 + (i % 3)]; }),
  };
  const phase = sats.map((_, i) => i * 1.37);
  const liked = (i) => i % 3 !== 1;

  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    L.vh = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * CAM_Z;
    L.vw = L.vh * camera.aspect;
    L.wide = camera.aspect >= 1.05;
    L.u = Math.min(L.vw * 1.15, L.vh) / 3.79;
    if (L.wide) {
      L.center.set(L.vw * 0.36, 0.05, 0);
      L.R = Math.min(L.vw * 0.3, L.vh * 0.75);
    } else {
      L.center.set(0, L.vh * 0.36, 0);
      L.R = L.vw * 0.62;
    }
  }
  resize();

  const tmp = new THREE.Vector3();
  const toWorld = (p, out) => out.set(p[0] * L.vw, -p[1] * L.vh, p[2]);

  function render(t, sp, pointer) {
    const pF = smooth(0.15, 0.36, sp);   // shapes clear, core to centre
    const pR = smooth(0.38, 0.57, sp);   // satellites form the orbit
    const pL = smooth(0.6, 0.77, sp);    // like / skip
    const pD = smooth(0.86, 0.995, sp);  // dive into the core
    const u = L.u;
    const bob = (k, amp = 0.09) => Math.sin(t * 0.8 + k) * amp * u;

    // core
    toWorld(scatter.core, tmp);
    core.position.lerpVectors(tmp, L.center, pF);
    core.position.y += bob(0.3);
    core.scale.setScalar(lerp(scatter.core[3] * u, L.R * 0.36, pF) * (1 + pD * 0.4));
    core.rotation.y = t * 0.2 + sp * 3;

    // hero donut: big near the title, then flies off to the back
    toWorld(scatter.donut, tmp);
    donut.position.set(tmp.x + pF * L.vw * 0.9, tmp.y + pF * L.vh * 0.9 + bob(1.1), tmp.z - pF * 6);
    donut.scale.setScalar(scatter.donut[3] * u * (1 - pF * 0.6));
    donut.rotation.set(0.9 + t * 0.15 + sp * 2.2, t * 0.22, 0.4 + sp * 1.4);
    donut.visible = pF < 0.999;

    // small accents drift outward and shrink
    accents.forEach((m, i) => {
      toWorld(scatter.accents[i], tmp);
      m.position.set(tmp.x * (1 + pF * 0.9), tmp.y * (1 + pF * 0.9) + bob(i * 2.1, 0.12), tmp.z - pF * 3);
      m.scale.setScalar(u * (1 - pF * 0.55));
      m.rotation.set(t * (0.25 + i * 0.05) + sp * 3, t * (0.3 + i * 0.04) + i, sp * 2 + i);
      m.visible = pF < 0.999 || pD < 0.5;
    });

    // satellites: scattered -> orbit -> liked come closer, skipped drift away
    const spin = t * 0.22 * (0.4 + pR) + sp * 5;
    sats.forEach((m, i) => {
      toWorld(scatter.sats[i], tmp);
      tmp.lerp(L.center, pF * 0.25);
      const like = liked(i);
      const r = L.R * (1 + (like ? -0.16 : 0.42) * pL) * (1 + pD * 1.6);
      const a = (i / 8) * Math.PI * 2 + spin;
      const target = new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r).applyEuler(tilt).add(L.center);
      m.position.lerpVectors(tmp, target, pR);
      m.position.y += bob(phase[i], 0.06);
      const size = lerp(u * 1.1, L.R * 0.42, pR) * (1 + (like ? 0.35 : -0.5) * pL);
      m.scale.setScalar(Math.max(0.01, size));
    });

    // rings
    orbit.position.copy(L.center);
    const rs = Math.max(0.001, L.R * pR * (1 + pD * 1.6));
    ring.scale.setScalar(rs);
    ring2.scale.setScalar(rs * 1.42);
    orbit.rotation.set(tilt.x, t * 0.05, tilt.z);
    glow.opacity = 0.85 * pR * (1 - pD);
    glowFaint.opacity = 0.18 * pR * (1 - pD);
    orbit.visible = pR > 0.001;

    dust.rotation.y = t * 0.015 + sp * 1.2;
    dust.scale.setScalar(1 + pD * 1.8);

    // camera: gentle parallax, then a dive into the core
    const look = new THREE.Vector3().copy(L.center).multiplyScalar(pF * 0.35 + pD * 0.65);
    camera.position.set(
      lerp(pointer.x * 0.6, L.center.x, pD),
      lerp(-pointer.y * 0.4, L.center.y, pD),
      lerp(CAM_Z, L.R * 0.5 + 0.8, pD * pD),
    );
    camera.lookAt(look);

    renderer.render(scene, camera);
  }

  function dispose() {
    disposables.forEach((x) => x.dispose && x.dispose());
    renderer.dispose();
    try { renderer.forceContextLoss(); } catch (error) { /* ignore */ }
  }

  function setLowPower() {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 0.85));
    resize();
  }

  return { render, resize, dispose, setLowPower };
}
