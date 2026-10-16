"""Self-contained HTML 3D viewer (three.js) for sharing models, e.g. as a phone-friendly artifact page."""
from __future__ import annotations

import base64
import html
import tempfile
from pathlib import Path

_TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
/* Layout: one-screen model inspector, header strip / full-bleed viewport / control strip */
:root {
  --bg: #e9edf1; --panel: #f7f9fb; --fg: #1b2330; --muted: #5d6877; --line: #cdd5de;
  --accent: #b9682a; --accent-fg: #ffffff; --stage-top: #f4f6f8; --stage-bottom: #d5dce4;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", Roboto, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11151b; --panel: #181e26; --fg: #e4e8ee; --muted: #94a0ae; --line: #2a333f;
  --accent: #e3995a; --accent-fg: #1a1208; --stage-top: #1d242d; --stage-bottom: #0d1116; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11151b; --panel: #181e26; --fg: #e4e8ee; --muted: #94a0ae; --line: #2a333f;
  --accent: #e3995a; --accent-fg: #1a1208; --stage-top: #1d242d; --stage-bottom: #0d1116; color-scheme: dark }
html, body { height: 100%; }
body { background: var(--bg); color: var(--fg); font-family: var(--body); display: flex; flex-direction: column; }
header, .controls { padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header { padding-block: 12px; display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; border-bottom: 1px solid var(--line); background: var(--panel); }
h1 { font-family: var(--display); font-weight: 700; font-size: 1.15rem; margin: 0; letter-spacing: 0.01em; text-wrap: balance; }
.stats { display: flex; flex-wrap: wrap; gap: 4px 14px; font-family: var(--mono); font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums; min-width: 0; }
.stats b { color: var(--fg); font-weight: 500; }
#stage { position: relative; flex: 1; min-height: 240px; background: linear-gradient(var(--stage-top), var(--stage-bottom)); }
#stage canvas { display: block; width: 100%; height: 100%; touch-action: none; }
#status { position: absolute; inset: 0; display: grid; place-items: center; font-family: var(--mono); font-size: 0.85rem; color: var(--muted); pointer-events: none; }
.controls { padding-block: 10px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; border-top: 1px solid var(--line); background: var(--panel); }
.controls label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
button, select { font: 500 0.85rem var(--body); color: var(--fg); background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 7px 12px; min-height: 36px; cursor: pointer; }
button[aria-pressed="true"] { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
button:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.spacer { flex: 1; }
.files { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); overflow-wrap: anywhere; min-width: 0; }
</style>
<header>
  <h1>__NAME__</h1>
  <div class="stats" id="stats"><span>loading…</span></div>
</header>
<div id="stage"><div id="status">Loading model…</div></div>
<div class="controls">
  <label for="clip">Animation</label>
  <select id="clip" disabled><option>none</option></select>
  <button id="play" type="button" aria-pressed="true" disabled>Pause</button>
  <button id="spin" type="button" aria-pressed="false">Turntable</button>
  <button id="wire" type="button" aria-pressed="false">Wireframe</button>
  <button id="reset" type="button">Reset view</button>
  <span class="spacer"></span>
  <span class="files">__FILES__</span>
</div>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const MODEL_B64 = "__MODEL__";
const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
stage.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
let mixer = null, clips = [], action = null, playing = true, home = null;
const clock = new THREE.Clock();

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

function b64ToBuffer(b64) {
  const bin = atob(b64); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

new GLTFLoader().parse(b64ToBuffer(MODEL_B64), "", (gltf) => {
  const root = gltf.scene;
  scene.add(root);
  let tris = 0, lights = 0;
  root.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true; o.receiveShadow = true;
      const g = o.geometry; tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
    }
    if (o.isLight) lights++;
  });
  if (lights === 0) {
    scene.add(new THREE.HemisphereLight(0xdde6f0, 0x4a4036, 1.2));
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(4, 8, 6); key.castShadow = true;
    scene.add(key);
  }
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 0.01);
  const dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  const dir = new THREE.Vector3(0.75, 0.45, 1).normalize();
  home = { pos: center.clone().addScaledVector(dir, dist), target: center.clone() };
  camera.near = dist / 200; camera.far = dist * 50; camera.updateProjectionMatrix();
  resetView();
  const ground = new THREE.Mesh(new THREE.CircleGeometry(radius * 3, 64), new THREE.ShadowMaterial({ opacity: 0.18 }));
  ground.rotation.x = -Math.PI / 2; ground.position.y = box.min.y; ground.receiveShadow = true; scene.add(ground);
  const fmt = (v) => v.toFixed(v < 10 ? 2 : 1);
  document.getElementById("stats").innerHTML =
    `<span><b>${Math.round(tris).toLocaleString()}</b> triangles</span>` +
    `<span><b>${fmt(size.x)} × ${fmt(size.y)} × ${fmt(size.z)}</b> m</span>` +
    `<span><b>${gltf.animations.length}</b> animations</span>`;
  clips = gltf.animations;
  const sel = document.getElementById("clip");
  if (clips.length) {
    mixer = new THREE.AnimationMixer(root);
    sel.innerHTML = clips.map((c, i) => `<option value="${i}">${c.name || "Clip " + (i + 1)}</option>`).join("");
    sel.disabled = false; document.getElementById("play").disabled = false;
    playClip(0);
  }
  statusEl.hidden = true;
}, (err) => { statusEl.textContent = "Could not load the model: " + (err.message || err); });

function playClip(i) {
  if (!mixer) return;
  if (action) action.stop();
  action = mixer.clipAction(clips[i]); action.reset().play();
}
function resetView() {
  if (!home) return;
  camera.position.copy(home.pos); controls.target.copy(home.target); controls.update();
}
document.getElementById("clip").addEventListener("change", (e) => playClip(+e.target.value));
document.getElementById("play").addEventListener("click", (e) => {
  playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; e.target.setAttribute("aria-pressed", String(playing));
});
document.getElementById("spin").addEventListener("click", (e) => {
  controls.autoRotate = !controls.autoRotate; e.target.setAttribute("aria-pressed", String(controls.autoRotate));
});
document.getElementById("wire").addEventListener("click", (e) => {
  const on = e.target.getAttribute("aria-pressed") !== "true";
  e.target.setAttribute("aria-pressed", String(on));
  scene.traverse((o) => { if (o.isMesh && o.material) [].concat(o.material).forEach((m) => { m.wireframe = on; }); });
});
document.getElementById("reset").addEventListener("click", resetView);

renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  if (mixer && playing) mixer.update(dt);
  controls.update();
  renderer.render(scene, camera);
});
</script>
"""


def write_viewer(source, path, title: str | None = None, files: list[str] | None = None) -> Path:
    """Write an interactive HTML viewer page for a Scene or an existing .glb file.

    The model is embedded (base64), so the page is a single file you can
    publish/share. `files` lists exported file names to mention on the page.
    """
    path = Path(path)
    if isinstance(source, (str, Path)):
        glb = Path(source).read_bytes()
        name = Path(source).stem
    else:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "model.glb"
            source.export(tmp)
            glb = tmp.read_bytes()
        name = source.name
    title = title or name
    files_txt = " · ".join(files) if files else ""
    page = (_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__NAME__", html.escape(title))
            .replace("__FILES__", html.escape(files_txt)).replace("__MODEL__", base64.b64encode(glb).decode()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path
