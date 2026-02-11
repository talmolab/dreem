# Live Visualizer

Visualize your DREEM tracking results directly in the browser. Upload an SLP file and a video file to see pose annotations and track IDs overlaid on your data.

!!! info "How it works"
    This visualizer uses [sleap-io.js](https://github.com/talmolab/sleap-io.js) to parse SLP files entirely in your browser. No data is uploaded to any server.

---

<div id="visualizer-app">
  <div class="viz-demos">
    <button class="viz-demo-btn" id="demo-animals" data-slp="../assets/demos/animals.slp" data-video="../assets/demos/animals.mp4">
      <div class="viz-demo-title">Animals Demo</div>
      <div class="viz-demo-desc">Pose tracking on animal behavior</div>
    </button>
    <button class="viz-demo-btn" id="demo-microscopy" data-slp="../assets/demos/microscopy.slp" data-video="../assets/demos/microscopy.mp4">
      <div class="viz-demo-title">Microscopy Demo</div>
      <div class="viz-demo-desc">Cell tracking on microscopy data</div>
    </button>
  </div>
  <div class="viz-layout">
    <div class="viz-main">
      <div class="viz-player">
        <video id="video" playsinline muted></video>
        <canvas id="overlay"></canvas>
      </div>
      <div class="viz-controls">
        <button id="play-btn">Play</button>
        <input id="seek" type="range" min="0" max="0" value="0" step="1" />
        <span id="frame-label" class="viz-pill">Frame 0</span>
      </div>
    </div>

    <div class="viz-panel">
      <label for="slp-file">SLP File</label>
      <input id="slp-file" type="file" accept=".slp,.h5,.hdf5" />


      <label for="video-file" style="margin-top: 14px;">Video File</label>
      <input id="video-file" type="file" accept="video/*,.mp4,.avi,.mov" />

      <button id="load-btn">Load</button>
      <div class="viz-status" id="status">Select an SLP file to begin.</div>
      <div class="viz-meta" id="meta"></div>
    </div>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "h5wasm": "https://unpkg.com/h5wasm@0.8.8/dist/esm/hdf5_hl.js",
    "yaml": "https://esm.sh/yaml@2.6.1",
    "skia-canvas": "data:text/javascript,export class Canvas{}",
    "child_process": "data:text/javascript,export function spawn(){}"
  }
}
</script>

<style>
#visualizer-app {
  --viz-panel: rgba(127, 127, 127, 0.1);
  --viz-line: rgba(255, 152, 0, 0.3);
  --viz-accent: #ff9800;
  --viz-muted: #888;
}

[data-md-color-scheme="slate"] #visualizer-app {
  --viz-panel: rgba(255, 255, 255, 0.06);
}

#visualizer-app * { box-sizing: border-box; }

.viz-demos {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.viz-demo-btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 16px 20px;
  border-radius: 12px;
  border: 1px solid var(--viz-line);
  background: var(--viz-panel);
  color: var(--md-typeset-color);
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  text-align: left;
}

.viz-demo-btn:hover {
  border-color: var(--viz-accent);
  background: rgba(255, 152, 0, 0.08);
}

.viz-demo-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.viz-demo-title {
  font-size: 15px;
  font-weight: 600;
}

.viz-demo-desc {
  font-size: 12px;
  color: var(--viz-muted);
  margin-top: 4px;
}

.viz-layout {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  align-items: start;
  margin-top: 1rem;
}

.viz-panel {
  background: var(--viz-panel);
  border: 1px solid var(--viz-line);
  border-radius: 12px;
  padding: 16px;
}

.viz-panel label {
  display: block;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--viz-muted);
  margin-bottom: 6px;
}

.viz-panel input[type="text"] {
  width: 100%;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid transparent;
  background: rgba(127, 127, 127, 0.15);
  color: var(--md-typeset-color);
  font-size: 14px;
}

.viz-panel input[type="file"] {
  width: 100%;
  padding: 8px 0;
  font-size: 14px;
}

.viz-panel button {
  margin-top: 12px;
  padding: 10px 20px;
  border-radius: 999px;
  border: 1px solid var(--viz-accent);
  background: transparent;
  color: var(--md-typeset-color);
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}

.viz-panel button:hover {
  background: rgba(255, 152, 0, 0.1);
}

.viz-panel button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.viz-status {
  margin-top: 12px;
  font-size: 13px;
  color: var(--viz-muted);
}

.viz-meta {
  font-size: 12px;
  color: var(--viz-muted);
  margin-top: 8px;
  line-height: 1.5;
}

.viz-player {
  position: relative;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--viz-line);
  background: #000;
  min-height: 500px;
}

.viz-player video {
  width: 100%;
  display: block;
}

.viz-player canvas {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.viz-controls {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}

.viz-controls input[type="range"] {
  flex: 1;
}

.viz-controls button {
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid var(--viz-accent);
  background: transparent;
  color: var(--md-typeset-color);
  cursor: pointer;
  font-size: 13px;
}

.viz-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--viz-line);
  font-size: 12px;
  white-space: nowrap;
}

@media (max-width: 900px) {
  .viz-demos {
    flex-direction: column;
  }
}
</style>

<script type="module">
// Import sleap-io.js from CDN
const sleapioUrl = "https://unpkg.com/@talmolab/sleap-io.js@0.1.9/dist/index.js";
const { loadSlp } = await import(sleapioUrl);

// DOM elements
const fileInput = document.querySelector("#slp-file");
const videoFileInput = document.querySelector("#video-file");
const loadBtn = document.querySelector("#load-btn");
const statusEl = document.querySelector("#status");
const metaEl = document.querySelector("#meta");
const videoEl = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const playerEl = document.querySelector(".viz-player");
const seek = document.querySelector("#seek");
const playBtn = document.querySelector("#play-btn");
const frameLabel = document.querySelector("#frame-label");
const coordsEl = document.querySelector("#coords");
const demoAnimalsBtn = document.querySelector("#demo-animals");
const demoMicroscopyBtn = document.querySelector("#demo-microscopy");

const ctx = canvas.getContext("2d");

// Track colors - distinct colors for different track IDs
const colors = ["#ff9800", "#7dd3fc", "#a7f3d0", "#fda4af", "#c4b5fd", "#fcd34d", "#6ee7b7", "#f9a8d4"];
const trackColors = new Map();

const getTrackKey = (track) => {
  if (!track) return null;
  if (typeof track === "object") return track.id ?? track.name ?? track;
  return track;
};

const getInstanceColor = (instance, fallbackIndex) => {
  const trackKey = getTrackKey(instance.track);
  if (trackKey != null) {
    if (!trackColors.has(trackKey)) {
      trackColors.set(trackKey, colors[trackColors.size % colors.length]);
    }
    return trackColors.get(trackKey);
  }
  const stableIndex = instance.id ?? instance.instanceId ?? fallbackIndex;
  return colors[stableIndex % colors.length];
};

const formatPoint = (point) => {
  if (!point) return "None";
  if (!point.visible) return "None";
  const [x, y] = point.xy;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return "None";
  return `${x.toFixed(2)}, ${y.toFixed(2)}`;
};

const formatFrameCoords = (frame, skeleton) => {
  if (!coordsEl) return;
  if (!frame || !skeleton) {
    coordsEl.textContent = "—";
    return;
  }
  const lines = [];
  frame.instances.forEach((instance, idx) => {
    const trackKey = getTrackKey(instance.track);
    const trackName = instance.track?.name ?? `untracked`;
    const trackId = trackKey != null ? ` [Track ${trackKey}]` : "";
    const color = getInstanceColor(instance, idx);
    lines.push(`Instance ${idx} (${trackName})${trackId}`);
    instance.points.forEach((point, nodeIdx) => {
      const nodeName = skeleton.nodes[nodeIdx]?.name ?? `node ${nodeIdx}`;
      lines.push(`  ${nodeName}: ${formatPoint(point)}`);
    });
    lines.push("");
  });
  coordsEl.textContent = lines.length ? lines.join("\n").trim() : "—";
};

// State
let labels = null;
let framesByIndex = new Map();
let skeleton = null;
let videoFps = 30;
let maxFrame = 0;
let frameCount = 0;
let currentFrame = 0;
let isPlaying = false;
let playHandle = null;
let playbackStartTime = 0;
let playbackStartFrame = 0;
let renderToken = 0;
let embeddedMode = false;
let labeledFramesList = [];
let currentLabeledFrameIndex = 0;
// trackCentroids: Map<trackKey, Array<{frameIdx, x, y}>> sorted by frameIdx
let trackCentroids = new Map();
const TRAIL_LENGTH = 100;

const setStatus = (message) => {
  statusEl.textContent = message;
};

const setMeta = (data) => {
  if (!data) {
    metaEl.textContent = "";
    return;
  }
  let text = `Frames: ${data.frames} | Instances: ${data.instances} | Nodes: ${data.nodes}`;
  if (data.tracks) {
    text += ` | Tracks: ${data.tracks}`;
  }
  if (data.videos > 1) {
    text += ` | Videos: ${data.videos}`;
  }
  if (data.mode) {
    text += `\nMode: ${data.mode}`;
  }
  metaEl.textContent = text;
};

const configureCanvas = (width, height, setPlayerHeight = false) => {
  const w = width || videoEl?.videoWidth || 1280;
  const h = height || videoEl?.videoHeight || 720;
  canvas.width = w;
  canvas.height = h;
  if (setPlayerHeight) {
    const player = playerEl || document.querySelector(".viz-player");
    if (player) {
      const aspectRatio = h / w;
      player.style.paddingBottom = `${aspectRatio * 100}%`;
    }
  }
};

const computeCentroid = (instance) => {
  let sumX = 0, sumY = 0, count = 0;
  for (const point of instance.points) {
    if (!point || !point.visible) continue;
    const [x, y] = point.xy;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    sumX += x;
    sumY += y;
    count++;
  }
  if (count === 0) return null;
  return { x: sumX / count, y: sumY / count };
};

const buildFrameIndex = () => {
  framesByIndex = new Map();
  trackCentroids = new Map();
  let instanceCount = 0;
  for (const frame of labels.labeledFrames) {
    if (!Number.isFinite(frame.frameIdx)) continue;
    framesByIndex.set(frame.frameIdx, frame);
    instanceCount += frame.instances.length;

    frame.instances.forEach((instance, idx) => {
      const trackKey = getTrackKey(instance.track) ?? `untracked_${idx}`;
      const centroid = computeCentroid(instance);
      if (!centroid) return;
      if (!trackCentroids.has(trackKey)) trackCentroids.set(trackKey, []);
      trackCentroids.get(trackKey).push({ frameIdx: frame.frameIdx, ...centroid });
    });
  }

  // Sort each track's centroids by frame index
  for (const [, trail] of trackCentroids) {
    trail.sort((a, b) => a.frameIdx - b.frameIdx);
  }

  const frameIndices = Array.from(framesByIndex.keys()).filter((value) => Number.isFinite(value));
  maxFrame = frameIndices.length ? Math.max(...frameIndices) : 0;
  frameCount = frameIndices.length;

  trackColors.clear();
  labels.tracks.forEach((track, index) => {
    const key = getTrackKey(track) ?? track;
    trackColors.set(key, colors[index % colors.length]);
  });

  return instanceCount;
};

const drawMotionTrail = (trackKey, frameIdx, color) => {
  const trail = trackCentroids.get(trackKey);
  if (!trail || trail.length < 2) return;

  // Find the index of the current frame (or closest before it) via binary search
  let hi = trail.length - 1, lo = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (trail[mid].frameIdx <= frameIdx) lo = mid + 1;
    else hi = mid - 1;
  }
  const endIdx = hi; // last entry at or before frameIdx
  if (endIdx < 0) return;

  const startIdx = Math.max(0, endIdx - TRAIL_LENGTH + 1);
  const segment = trail.slice(startIdx, endIdx + 1);
  if (segment.length < 2) return;

  // Draw trail with fading opacity (oldest = 0.15, newest = 0.9)
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let i = 1; i < segment.length; i++) {
    const t = i / (segment.length - 1);
    ctx.globalAlpha = 0.15 + t * 0.75;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 + t * 2.5;
    ctx.beginPath();
    ctx.moveTo(segment[i - 1].x, segment[i - 1].y);
    ctx.lineTo(segment[i].x, segment[i].y);
    ctx.stroke();
  }

  // Draw dots at trail positions
  for (let i = 0; i < segment.length; i++) {
    const t = i / (segment.length - 1);
    ctx.globalAlpha = 0.2 + t * 0.8;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(segment[i].x, segment[i].y, 1.5 + t * 2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.globalAlpha = 1;
  ctx.lineCap = "butt";
  ctx.lineJoin = "miter";
};

const drawTrailsForFrame = (frameIdx) => {
  if (!ctx || !trackCentroids.size) return;
  for (const [trackKey] of trackCentroids) {
    const color = trackColors.get(trackKey) ?? colors[0];
    drawMotionTrail(trackKey, frameIdx, color);
  }
};

const drawSkeleton = (frame, skel) => {
  if (!ctx || !skel || !frame) return;

  frame.instances.forEach((instance, idx) => {
    const color = getInstanceColor(instance, idx);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;

    // Draw edges
    for (const edge of skel.edges) {
      const sourceIdx = skel.index(edge.source.name);
      const destIdx = skel.index(edge.destination.name);
      const source = instance.points[sourceIdx];
      const dest = instance.points[destIdx];
      if (!source || !dest) continue;
      if (!source.visible || !dest.visible) continue;
      if (Number.isNaN(source.xy[0]) || Number.isNaN(dest.xy[0])) continue;
      ctx.beginPath();
      ctx.moveTo(source.xy[0], source.xy[1]);
      ctx.lineTo(dest.xy[0], dest.xy[1]);
      ctx.stroke();
    }

    // Draw nodes
    instance.points.forEach((point) => {
      if (!point.visible || Number.isNaN(point.xy[0])) return;
      ctx.beginPath();
      ctx.arc(point.xy[0], point.xy[1], 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw track label at centroid offset
    const trackKey = getTrackKey(instance.track);
    if (trackKey != null) {
      const centroid = computeCentroid(instance);
      if (centroid) {
        const label = `${trackKey}`;
        const fontSize = Math.max(12, Math.round(Math.min(canvas.width, canvas.height) * 0.05));
        const offset = Math.round(fontSize * 0.5);
        ctx.font = `bold ${fontSize}px sans-serif`;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        // Background pill for readability
        const metrics = ctx.measureText(label);
        const px = centroid.x + offset;
        const py = centroid.y - offset;
        const pad = Math.round(fontSize * 0.25);
        ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
        ctx.beginPath();
        ctx.roundRect(px - pad, py - metrics.actualBoundingBoxAscent - pad, metrics.width + pad * 2, metrics.actualBoundingBoxAscent + pad * 2, pad);
        ctx.fill();
        ctx.fillStyle = color;
        ctx.fillText(label, px, py);
        ctx.textAlign = "start";
        ctx.textBaseline = "alphabetic";
      }
    }
  });
};

const renderExternalFrame = (frameIdx) => {
  if (!ctx || !skeleton) return;
  const frame = framesByIndex.get(frameIdx);

  // Seek video: frameIdx maps directly to time via fps
  const time = frameIdx / videoFps;
  if (Math.abs(videoEl.currentTime - time) > 0.001) {
    videoEl.currentTime = time;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawTrailsForFrame(frameIdx);

  if (frame) {
    drawSkeleton(frame, skeleton);
    formatFrameCoords(frame, skeleton);
  }
};

const renderEmbeddedFrame = async (labeledFrameIndex) => {
  if (!ctx || !skeleton || labeledFrameIndex < 0 || labeledFrameIndex >= labeledFramesList.length) return;

  const frame = labeledFramesList[labeledFrameIndex];
  const video = frame.video;
  const token = ++renderToken;

  let imageData = null;
  if (video.backend) {
    try {
      imageData = await video.backend.getFrame(frame.frameIdx);
    } catch (err) {
      console.warn("Error getting embedded frame:", err);
    }
  }

  if (token !== renderToken) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (imageData) {
    if (imageData instanceof ImageBitmap) {
      configureCanvas(imageData.width, imageData.height, true);
      ctx.drawImage(imageData, 0, 0);
    } else if (imageData instanceof ImageData) {
      configureCanvas(imageData.width, imageData.height, true);
      ctx.putImageData(imageData, 0, 0);
    } else if (imageData instanceof Uint8Array) {
      const blob = new Blob([imageData], { type: "image/png" });
      const bitmap = await createImageBitmap(blob);
      configureCanvas(bitmap.width, bitmap.height, true);
      ctx.drawImage(bitmap, 0, 0);
    }
  } else {
    configureCanvas(canvas.width || 640, canvas.height || 480, true);
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#888";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    const videoIdx = labels.videos.indexOf(video);
    ctx.fillText(`Video ${videoIdx}, Frame ${frame.frameIdx}`, canvas.width / 2, canvas.height / 2);
    ctx.fillText(`(No embedded image data)`, canvas.width / 2, canvas.height / 2 + 24);
  }

  drawTrailsForFrame(frame.frameIdx);
  drawSkeleton(frame, skeleton);
  formatFrameCoords(frame, skeleton);
};


const frameIndexForTime = (time) => {
  if (!Number.isFinite(time)) return 0;
  return Math.min(maxFrame, Math.max(0, Math.round(time * videoFps)));
};

const updateEmbeddedFrame = (labeledFrameIndex) => {
  if (labeledFrameIndex < 0 || labeledFrameIndex >= labeledFramesList.length) return;
  currentLabeledFrameIndex = labeledFrameIndex;
  seek.value = String(labeledFrameIndex);
  const frame = labeledFramesList[labeledFrameIndex];
  const videoIdx = labels.videos.indexOf(frame.video);
  frameLabel.textContent = `Frame ${labeledFrameIndex + 1}/${labeledFramesList.length} (v${videoIdx}:${frame.frameIdx})`;
  renderEmbeddedFrame(labeledFrameIndex);
};

const playLoop = () => {
  if (!isPlaying) return;
  const time = videoEl.currentTime;
  const frameIdx = frameIndexForTime(time);
  if (frameIdx !== currentFrame) {
    currentFrame = frameIdx;
    seek.value = String(frameIdx);
    frameLabel.textContent = `Frame ${frameIdx}`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawTrailsForFrame(frameIdx);
    const frame = framesByIndex.get(frameIdx);
    if (frame) {
      drawSkeleton(frame, skeleton);
      formatFrameCoords(frame, skeleton);
    }
  }
  if (videoEl.ended || currentFrame >= maxFrame) {
    stopPlayback();
    return;
  }
  playHandle = requestAnimationFrame(playLoop);
};

const startPlayback = () => {
  if (isPlaying) return;
  isPlaying = true;
  videoEl.play().catch(() => {});
  playHandle = requestAnimationFrame(playLoop);
};

const stopPlayback = () => {
  isPlaying = false;
  videoEl.pause();
  if (playHandle) cancelAnimationFrame(playHandle);
  playHandle = null;
};

const hasEmbeddedImages = () => {
  if (!labels?.videos?.length) return false;
  return labels.videos.some((v) => v.backend?.dataset || v.backendMetadata?.dataset);
};

const doLoad = async (slpSource, slpFilename, videoObjectUrl) => {
  canvas.style.display = "";
  loadBtn.disabled = true;
  setStatus("Loading SLP...");

  try {
    const useEmbedded = !videoObjectUrl;

    labels = await loadSlp(slpSource, {
      openVideos: useEmbedded,
      h5: { stream: undefined, filenameHint: slpFilename },
    });
    skeleton = labels.skeletons[0];
    const instanceCount = buildFrameIndex();
    labeledFramesList = labels.labeledFrames;

    embeddedMode = useEmbedded && (hasEmbeddedImages() || labels.videos.length > 1);

    if (embeddedMode) {
      setMeta({
        frames: labeledFramesList.length,
        instances: instanceCount,
        nodes: skeleton?.nodes.length ?? 0,
        tracks: labels.tracks.length,
        videos: labels.videos.length,
        mode: "embedded images",
      });

      seek.max = String(labeledFramesList.length - 1);
      videoEl.style.display = "none";
      playBtn.style.display = "none";
      configureCanvas(1024, 1024, true);
      currentLabeledFrameIndex = 0;
      updateEmbeddedFrame(0);
      setStatus("Ready. Use slider or arrow keys to navigate.");
    } else {
      if (!videoObjectUrl) {
        setStatus("No video provided and no embedded images found.");
        loadBtn.disabled = false;
        return;
      }

      setMeta({
        frames: frameCount,
        instances: instanceCount,
        nodes: skeleton?.nodes.length ?? 0,
        tracks: labels.tracks.length,
        videos: 1,
        mode: "external video",
      });

      setStatus("Loading video...");
      videoEl.style.display = "block";
      playBtn.style.display = "inline-block";
      videoEl.src = videoObjectUrl;
      await new Promise((resolve) => {
        if (videoEl.readyState >= 1) return resolve();
        videoEl.addEventListener("loadedmetadata", resolve, { once: true });
      });

      // Detect actual fps from the video using requestVideoFrameCallback
      videoFps = await new Promise((resolve) => {
        if (!("requestVideoFrameCallback" in videoEl)) {
          // Fallback: estimate from total SLP frames / video duration
          const est = maxFrame / videoEl.duration;
          resolve(est > 0 ? est : 30);
          return;
        }
        videoEl.currentTime = 0;
        videoEl.requestVideoFrameCallback((now, meta1) => {
          videoEl.requestVideoFrameCallback((now2, meta2) => {
            const dt = meta2.mediaTime - meta1.mediaTime;
            videoEl.pause();
            videoEl.currentTime = 0;
            resolve(dt > 0 ? Math.round(1 / dt) : 30);
          });
          videoEl.play().catch(() => {});
        });
        videoEl.play().catch(() => {});
      });

      // Clamp maxFrame to video duration
      const videoMaxFrame = Math.floor(videoEl.duration * videoFps);
      if (videoMaxFrame > 0 && videoMaxFrame < maxFrame) {
        maxFrame = videoMaxFrame;
      }

      configureCanvas(videoEl.videoWidth, videoEl.videoHeight);
      videoEl.currentTime = 0;
      seek.max = String(maxFrame);
      currentFrame = 0;
      renderToken = 0;
      renderExternalFrame(0);
      setStatus("Ready. Use Play or arrow keys to navigate.");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    setStatus(`Load failed: ${message}`);
    console.error(error);
  } finally {
    loadBtn.disabled = false;
  }
};

const handleLoad = async () => {
  const file = fileInput?.files?.[0];
  const videoFile = videoFileInput?.files?.[0];

  if (!file) {
    setStatus("Select an SLP file to begin.");
    return;
  }

  const slpSource = await file.arrayBuffer();
  const slpFilename = file.name;
  const videoObjectUrl = videoFile ? URL.createObjectURL(videoFile) : null;
  await doLoad(slpSource, slpFilename, videoObjectUrl);
};

const loadDemo = async (name) => {
  const btn = document.querySelector(`#demo-${name}`);
  const slpUrl = btn?.dataset.slp;
  const videoUrl = btn?.dataset.video;
  if (!slpUrl || !videoUrl) return;

  setStatus(`Loading ${name} demo...`);
  document.querySelectorAll(".viz-demo-btn").forEach(b => b.disabled = true);

  try {
    canvas.style.display = "";

    const [slpResp, videoResp] = await Promise.all([fetch(slpUrl), fetch(videoUrl)]);
    if (!slpResp.ok) throw new Error(`Failed to fetch SLP: ${slpResp.status}`);
    if (!videoResp.ok) throw new Error(`Failed to fetch video: ${videoResp.status}`);

    const slpSource = await slpResp.arrayBuffer();
    const videoBlob = await videoResp.blob();
    const videoObjectUrl = URL.createObjectURL(videoBlob);
    const slpFilename = slpUrl.split("/").pop();

    await doLoad(slpSource, slpFilename, videoObjectUrl);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    setStatus(`Demo load failed: ${message}`);
    console.error(error);
  } finally {
    document.querySelectorAll(".viz-demo-btn").forEach(b => b.disabled = false);
    loadBtn.disabled = false;
  }
};

// Event listeners
seek.addEventListener("input", () => {
  if (embeddedMode) {
    updateEmbeddedFrame(Number(seek.value));
  } else {
    const frameIdx = Number(seek.value);
    currentFrame = frameIdx;
    frameLabel.textContent = `Frame ${frameIdx}`;
    renderExternalFrame(frameIdx);
  }
});

playBtn?.addEventListener("click", () => {
  if (embeddedMode) return;
  if (isPlaying) {
    stopPlayback();
    playBtn.textContent = "Play";
  } else {
    startPlayback();
    playBtn.textContent = "Pause";
  }
});

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (!labels) return;
  // Don't capture if typing in an input
  if (e.target.tagName === "INPUT") return;

  if (e.key === "ArrowLeft" || e.key === "a") {
    if (embeddedMode) {
      if (currentLabeledFrameIndex > 0) {
        updateEmbeddedFrame(currentLabeledFrameIndex - 1);
      }
    } else {
      if (currentFrame > 0) {
        currentFrame--;
        seek.value = String(currentFrame);
        frameLabel.textContent = `Frame ${currentFrame}`;
        renderExternalFrame(currentFrame);
      }
    }
  } else if (e.key === "ArrowRight" || e.key === "d") {
    if (embeddedMode) {
      if (currentLabeledFrameIndex < labeledFramesList.length - 1) {
        updateEmbeddedFrame(currentLabeledFrameIndex + 1);
      }
    } else {
      if (currentFrame < maxFrame) {
        currentFrame++;
        seek.value = String(currentFrame);
        frameLabel.textContent = `Frame ${currentFrame}`;
        renderExternalFrame(currentFrame);
      }
    }
  } else if (e.key === " ") {
    e.preventDefault();
    if (!embeddedMode) {
      playBtn?.click();
    }
  }
});

loadBtn.addEventListener("click", handleLoad);
demoAnimalsBtn?.addEventListener("click", () => loadDemo("animals"));
demoMicroscopyBtn?.addEventListener("click", () => loadDemo("microscopy"));
</script>

---


## Supported Formats

- **SLP files**: `.slp`, `.h5`, `.hdf5` (SLEAP format)
- **Videos**: `.mp4`, `.avi`, `.mov` (browser-supported formats)

!!! tip "Track Colors"
    Each track ID is assigned a distinct color. The track ID is displayed next to the first visible keypoint of each instance.
