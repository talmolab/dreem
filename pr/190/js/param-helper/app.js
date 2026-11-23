/**
 * "Choosing Parameters" page (docs/parameters.md): reads the user's files in the browser,
 * nothing is uploaded. The numbers come from stats.js; this file only wires the page.
 */
import { command, fromMasks, fromSleapIo, measure, quantile, recommend } from "./stats.js";

const SLEAP_IO = "https://unpkg.com/@talmolab/sleap-io.js@0.1.9/dist/index.js";
const GEOTIFF = "https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm";
const MAX_MASK_FRAMES = 300; // decoding every mask of a long stack would take minutes
// The Live Visualizer's track colors, so an animal looks the same on both pages.
const COLORS = ["#ff9800", "#7dd3fc", "#a7f3d0", "#fda4af", "#c4b5fd", "#fcd34d", "#6ee7b7", "#f9a8d4"];

const $ = (id) => document.getElementById(id);
const el = {
  slp: $("ph-slp"),
  video: $("ph-video"),
  masks: $("ph-masks"),
  status: $("ph-status"),
  anchor: $("ph-anchor"),
  results: $("ph-results"),
  canvas: $("ph-canvas"),
  frameLabel: $("ph-frame-label"),
  hist: $("ph-hist"),
  command: $("ph-command"),
  output: $("ph-output"),
};
const ctx = el.canvas.getContext("2d");

const state = {
  data: null, // the plain structure from stats.js
  source: "", // what the data came from, for the status line
  video: null, // <video> element when the user gave one
  fps: 30,
  frames: [], // labelled frames with instances (or video frame numbers when there are none)
  cursor: 0,
  box: null, // [x0, y0, x1, y1] drawn by the user, video pixels
  drag: null,
  rec: null,
  m: null,
};

const setStatus = (text, isError = false) => {
  el.status.textContent = text;
  el.status.classList.toggle("ph-error", isError);
};

const domain = () => document.querySelector('input[name="ph-domain"]:checked')?.value;
const anchorChoice = () => (el.anchor.value === "centroid" ? "centroid" : Number(el.anchor.value));
const boxSize = () => (state.box ? Math.max(Math.abs(state.box[2] - state.box[0]), Math.abs(state.box[3] - state.box[1])) : null);

// ---------------------------------------------------------------- loading

async function loadSlp(buffer, name) {
  setStatus(`Reading ${name}...`);
  const { loadSlp } = await import(SLEAP_IO);
  const labels = await loadSlp(buffer, { openVideos: false, h5: { stream: undefined, filenameHint: name } });
  const data = fromSleapIo(labels, 0);
  const extra = labels.videos.length > 1 ? ` (${labels.videos.length} videos in the file; using the first)` : "";
  state.source = `${name}: ${data.frames.length} labelled frames, ${data.nodes.length} node${data.nodes.length === 1 ? "" : "s"}${data.tracked ? ", tracked" : ""}${extra}`;
  if (data.kind === "pose") setDomain("animals");
  return data;
}

async function loadMasks(files) {
  setStatus(`Reading ${files.length} mask file${files.length > 1 ? "s" : ""}...`);
  const { fromArrayBuffer } = await import(GEOTIFF);
  const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const images = [];
  for (const file of sorted) {
    const tiff = await fromArrayBuffer(await file.arrayBuffer());
    const n = await tiff.getImageCount();
    for (let i = 0; i < n && images.length < MAX_MASK_FRAMES; i++) {
      const img = await tiff.getImage(i);
      const [band] = await img.readRasters();
      images.push({ width: img.getWidth(), height: img.getHeight(), data: band });
    }
    if (images.length >= MAX_MASK_FRAMES) break;
  }
  const data = fromMasks(images);
  const capped = images.length >= MAX_MASK_FRAMES ? ` (the first ${MAX_MASK_FRAMES})` : "";
  state.source = `${images.length} mask frames${capped}`;
  setDomain("cells");
  return data;
}

async function loadVideo(url) {
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.src = url;
  await new Promise((resolve, reject) => {
    v.addEventListener("loadeddata", resolve, { once: true });
    v.addEventListener("error", () => reject(new Error("The browser cannot play this video format.")), { once: true });
  });
  state.fps = await estimateFps(v);
  state.video = v;
}

// Frame rate from two consecutive presented frames (as the Live Visualizer does); 30 if unknown.
function estimateFps(v) {
  return new Promise((resolve) => {
    if (!("requestVideoFrameCallback" in v)) return resolve(30);
    const done = (fps) => {
      v.pause();
      v.currentTime = 0;
      resolve(fps);
    };
    setTimeout(() => done(30), 3000);
    v.requestVideoFrameCallback((_, a) => {
      v.requestVideoFrameCallback((__, b) => {
        const dt = b.mediaTime - a.mediaTime;
        done(dt > 0 ? Math.round(1 / dt) : 30);
      });
    });
    v.play().catch(() => done(30));
  });
}

// Seek only when the frame changes: a seek to the current time may never fire "seeked".
function seek(frameIdx) {
  const v = state.video;
  const t = Math.min(v?.duration ?? 0, (frameIdx + 0.5) / state.fps);
  if (!v || Math.abs(v.currentTime - t) < 1e-3) return Promise.resolve();
  return new Promise((resolve) => {
    v.addEventListener("seeked", resolve, { once: true });
    setTimeout(resolve, 2000);
    v.currentTime = t;
  });
}

// ---------------------------------------------------------------- analysis

function setDomain(value) {
  const input = document.querySelector(`input[name="ph-domain"][value="${value}"]`);
  if (input) input.checked = true;
}

function fillAnchors(nodes) {
  el.anchor.replaceChildren(new Option("centroid (mean of visible keypoints)", "centroid"));
  nodes.forEach((name, i) => el.anchor.add(new Option(name, String(i))));
  el.anchor.disabled = nodes.length < 2;
}

function analyse() {
  const data = state.data;
  if (!data) {
    // A video alone: only the drawn box gives a suggestion.
    const size = boxSize();
    el.results.replaceChildren(row("Crop size", "--crop-size", size ? Math.ceil(size / 8) * 8 : null,
      size ? `From the box you drew: ${Math.round(size)} px, rounded up to a multiple of 8.` : "Drag a box around one animal or cell on the frame below.", null));
    el.command.textContent = "";
    el.output.hidden = !size;
    if (size) el.command.textContent = `--crop-size ${Math.ceil(size / 8) * 8}`;
    drawHist([]);
    return;
  }
  state.m = measure(data, { anchor: anchorChoice() });
  state.rec = recommend(state.m, data, { anchor: anchorChoice(), domain: domain(), boxSize: boxSize() });
  const r = state.rec;
  const o = r.orientation;
  el.results.replaceChildren(
    row("Model", "--checkpoint", r.model.value, r.model.why),
    row("Crop size", "--crop-size", r.crop.value, r.crop.why),
    row("Max tracks", "--max-tracks", r.maxTracks.value, r.maxTracks.why),
    row("Max distance", "--max-dist", r.maxDist.value, r.maxDist.why, r.maxDist.warn),
    row("Max gap", "--max-gap", r.maxGap.value, r.maxGap.why),
    row("Anchor", "--anchor", r.anchor.value, r.anchor.why),
    row("Orientation", "--front-node / --back-node / --max-angle", o.maxAngle != null ? `${o.front} / ${o.back} / ${o.maxAngle}` : null, o.why),
  );
  el.command.textContent = command(r);
  el.output.hidden = false;
  drawHist(state.m.moves, r.maxDist.value, state.m.spacing.length ? quantile(state.m.spacing, 0.05) : null);
}

function row(label, flag, value, why, warn) {
  const tr = document.createElement("tr");
  const th = document.createElement("th");
  th.scope = "row";
  th.innerHTML = `${label}<br><code></code>`;
  th.querySelector("code").textContent = flag;
  const val = document.createElement("td");
  val.className = "ph-value";
  val.textContent = value ?? "leave unset";
  if (value == null) val.classList.add("ph-unset");
  const reason = document.createElement("td");
  reason.textContent = why;
  if (warn) {
    const w = document.createElement("div");
    w.className = "ph-warn";
    w.textContent = warn;
    reason.append(w);
  }
  tr.append(th, val, reason);
  return tr;
}

// ---------------------------------------------------------------- frame view

const currentFrameIdx = () => {
  const f = state.frames[state.cursor];
  return state.data ? f?.frameIdx : f;
};

async function showFrame() {
  if (state.video) await seek(currentFrameIdx() ?? 0);
  draw();
}

/** Redraw the current video frame with keypoints, crop squares and the drawn box. */
function draw() {
  const v = state.video;
  const f = state.frames[state.cursor];
  const frameIdx = currentFrameIdx();
  const w = v?.videoWidth || extent(0);
  const h = v?.videoHeight || extent(1);
  if (el.canvas.width !== w || el.canvas.height !== h) {
    el.canvas.width = w;
    el.canvas.height = h;
  }
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, w, h);
  if (v) ctx.drawImage(v, 0, 0, w, h);
  const lw = Math.max(1.5, w / 400);
  if (state.data && f) {
    const crop = state.rec?.crop.value;
    f.instances.forEach((inst, i) => {
      const color = COLORS[(inst.track ?? i) % COLORS.length];
      ctx.fillStyle = color;
      for (const p of inst.points) {
        if (!p) continue;
        ctx.beginPath();
        ctx.arc(p[0], p[1], lw * 1.6, 0, 2 * Math.PI);
        ctx.fill();
      }
      const c = anchorOf(inst);
      if (crop && c) {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.strokeRect(c[0] - crop / 2, c[1] - crop / 2, crop, crop);
      }
    });
  }
  if (state.box) {
    const [x0, y0, x1, y1] = state.box;
    ctx.setLineDash([lw * 4, lw * 3]);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = lw;
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.setLineDash([]);
  }
  const total = state.frames.length;
  el.frameLabel.textContent = total ? `Frame ${frameIdx} (${state.cursor + 1} of ${total}${state.data ? " labelled" : ""})` : "";
}

function anchorOf(inst) {
  const a = anchorChoice();
  if (a !== "centroid" && inst.points[a]) return inst.points[a];
  const pts = inst.points.filter(Boolean);
  if (!pts.length) return null;
  return [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
}

// Canvas size from the labels when there is no video: the furthest keypoint plus a margin.
function extent(axis) {
  let m = 0;
  for (const f of state.data?.frames ?? []) for (const inst of f.instances) for (const p of inst.points) if (p) m = Math.max(m, p[axis]);
  return Math.ceil(m * 1.05) || 512;
}

function canvasPoint(e) {
  const r = el.canvas.getBoundingClientRect();
  return [((e.clientX - r.left) * el.canvas.width) / r.width, ((e.clientY - r.top) * el.canvas.height) / r.height];
}

el.canvas.addEventListener("pointerdown", (e) => {
  if (!state.video && !state.data) return;
  el.canvas.setPointerCapture(e.pointerId);
  const [x, y] = canvasPoint(e);
  state.drag = [x, y];
  state.box = [x, y, x, y];
});
el.canvas.addEventListener("pointermove", (e) => {
  if (!state.drag) return;
  const [x, y] = canvasPoint(e);
  state.box = [state.drag[0], state.drag[1], x, y];
  draw();
});
el.canvas.addEventListener("pointerup", () => {
  if (!state.drag) return;
  state.drag = null;
  if (boxSize() < 3) state.box = null; // a click, not a box
  analyse();
  draw();
});

// ---------------------------------------------------------------- histogram

const SVG = "http://www.w3.org/2000/svg";
function svg(tag, attrs, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

/** Moves per frame, with the suggested --max-dist and the close spacing marked. */
function drawHist(moves, maxDist, close) {
  el.hist.replaceChildren();
  if (moves.length < 2) return;
  const W = 640;
  const H = 220;
  const pad = { l: 44, r: 12, t: 28, b: 40 };
  const marks = [quantile(moves, 0.995), maxDist ?? 0].filter(Number.isFinite);
  if (close != null && close < 3 * Math.max(...marks)) marks.push(close);
  const xmax = Math.max(...marks) * 1.1 || 1;
  const nb = 30;
  const bins = new Array(nb).fill(0);
  let over = 0;
  for (const d of moves) {
    if (d > xmax) over += 1;
    else bins[Math.min(nb - 1, Math.floor((d / xmax) * nb))] += 1;
  }
  const ymax = Math.max(...bins) || 1;
  const x = (d) => pad.l + (d / xmax) * (W - pad.l - pad.r);
  const y = (n) => H - pad.b - (n / ymax) * (H - pad.t - pad.b);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Histogram of how far objects move between consecutive frames" });
  root.append(svg("text", { x: pad.l, y: 16, class: "ph-hist-title" }, "How far objects move between consecutive frames"));
  const bw = (W - pad.l - pad.r) / nb;
  bins.forEach((n, i) => {
    if (!n) return;
    const r = svg("rect", { x: x((i * xmax) / nb) + 1, y: y(n), width: Math.max(1, bw - 2), height: H - pad.b - y(n), rx: 1.5, class: "ph-hist-bar" });
    const lo = (i * xmax) / nb;
    r.append(svg("title", {}, `${lo.toFixed(1)}-${(lo + xmax / nb).toFixed(1)} px: ${n} moves`));
    root.append(r);
  });
  root.append(svg("line", { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b, class: "ph-hist-axis" }));
  for (let k = 0; k <= 4; k++) {
    const d = (k * xmax) / 4;
    root.append(svg("text", { x: x(d), y: H - pad.b + 16, "text-anchor": "middle", class: "ph-hist-tick" }, d >= 10 ? Math.round(d) : d.toFixed(1)));
  }
  root.append(svg("text", { x: (pad.l + W - pad.r) / 2, y: H - 6, "text-anchor": "middle", class: "ph-hist-tick" }, `pixels per frame (${moves.length} moves${over ? `, ${over} beyond the axis` : ""})`));
  const mark = (d, label, cls, dy) => {
    root.append(svg("line", { x1: x(d), x2: x(d), y1: pad.t - 4, y2: H - pad.b, class: cls }));
    root.append(svg("text", { x: x(d) + 4, y: pad.t + dy, class: "ph-hist-label" }, label));
  };
  if (maxDist) mark(maxDist, `--max-dist ${maxDist}`, "ph-hist-mark", 6);
  if (close != null && close <= xmax) mark(close, `closest spacing ${close >= 10 ? Math.round(close) : close.toFixed(1)} px`, "ph-hist-mark ph-hist-mark--alt", 22);
  el.hist.append(root);
}

// ---------------------------------------------------------------- wiring

async function run({ slp, slpName, masks, videoUrl, domainHint }) {
  try {
    state.data = null;
    state.video = null;
    state.box = null;
    state.cursor = 0;
    for (const input of document.querySelectorAll('input[name="ph-domain"]')) input.checked = false;
    if (domainHint) setDomain(domainHint);
    if (slp) state.data = await loadSlp(slp, slpName);
    else if (masks?.length) state.data = await loadMasks(masks);
    if (videoUrl) {
      setStatus("Opening the video...");
      await loadVideo(videoUrl);
    }
    fillAnchors(state.data?.nodes ?? []);
    if (state.data) {
      state.frames = state.data.frames.filter((f) => f.instances.length);
    } else if (state.video) {
      const n = Math.floor(state.video.duration * state.fps);
      state.frames = Array.from({ length: 20 }, (_, i) => Math.floor((i * n) / 20));
      state.source = "Video only: drag a box around one animal or cell to size the crop.";
    } else {
      setStatus("Choose a labels file, mask TIFFs or a video.");
      return;
    }
    // Start on a frame with the most common number of objects.
    if (state.data) {
      const mode = measure(state.data).counts.mode;
      state.cursor = Math.max(0, state.frames.findIndex((f) => f.instances.length === mode));
    }
    analyse();
    await showFrame();
    $("ph-view").hidden = false;
    setStatus(state.source);
  } catch (err) {
    console.error(err);
    setStatus(`Could not read the files: ${err.message}`, true);
  }
}

$("ph-run").addEventListener("click", async () => {
  const slpFile = el.slp.files[0];
  const videoFile = el.video.files[0];
  await run({
    slp: slpFile ? await slpFile.arrayBuffer() : null,
    slpName: slpFile?.name,
    masks: el.masks.files,
    videoUrl: videoFile ? URL.createObjectURL(videoFile) : null,
  });
});

for (const [id, name] of [["ph-demo-animals", "animals"], ["ph-demo-cells", "microscopy"]]) {
  $(id).addEventListener("click", async () => {
    setStatus("Downloading the demo...");
    const base = new URL("../assets/demos/", window.location.href);
    const [slp, video] = await Promise.all([fetch(new URL(`${name}.slp`, base)), fetch(new URL(`${name}.mp4`, base))]);
    if (!slp.ok || !video.ok) return setStatus("Could not download the demo files.", true);
    await run({
      slp: await slp.arrayBuffer(),
      slpName: `${name}.slp`,
      videoUrl: URL.createObjectURL(await video.blob()),
      domainHint: name === "animals" ? "animals" : "cells",
    });
  });
}

el.anchor.addEventListener("change", () => {
  analyse();
  draw();
});
for (const input of document.querySelectorAll('input[name="ph-domain"]')) input.addEventListener("change", analyse);
$("ph-prev").addEventListener("click", () => {
  if (!state.frames.length) return;
  state.cursor = (state.cursor - 1 + state.frames.length) % state.frames.length;
  showFrame();
});
$("ph-next").addEventListener("click", () => {
  if (!state.frames.length) return;
  state.cursor = (state.cursor + 1) % state.frames.length;
  showFrame();
});
$("ph-clear-box").addEventListener("click", () => {
  state.box = null;
  analyse();
  draw();
});
$("ph-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el.command.textContent);
    $("ph-copy").textContent = "Copied";
    setTimeout(() => ($("ph-copy").textContent = "Copy"), 1500);
  } catch {
    setStatus("Copy failed: select the command and copy it by hand.", true);
  }
});
