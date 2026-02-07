# Live Visualizer

Visualize your DREEM tracking results directly in the browser. Upload an SLP file and a video file to see pose annotations and track IDs overlaid on your data.

!!! info "How it works"
    This visualizer uses [sleap-io.js](https://github.com/talmolab/sleap-io.js) to parse SLP files entirely in your browser. No data is uploaded to any server.

---

<div id="visualizer-app">
  <div class="viz-layout">
    <div class="viz-panel">
      <label for="slp-file">SLP File</label>
      <input id="slp-file" type="file" accept=".slp,.h5,.hdf5" />


      <label for="video-file" style="margin-top: 14px;">Video File</label>
      <input id="video-file" type="file" accept="video/*,.mp4,.avi,.mov" />

      <button id="load-btn">Load</button>
      <div class="viz-status" id="status">Select an SLP file to begin.</div>
      <div class="viz-meta" id="meta"></div>
    </div>

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
      <div class="viz-panel viz-coords">
        <div class="viz-coords-label">Coordinates & Track Info</div>
        <pre id="coords" class="viz-coords-body">—</pre>
      </div>
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

.viz-layout {
  display: grid;
  grid-template-columns: 300px 1fr;
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
  min-height: 300px;
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

.viz-coords {
  margin-top: 12px;
}

.viz-coords-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--viz-muted);
  margin-bottom: 6px;
}

.viz-coords-body {
  margin: 0;
  max-height: 200px;
  overflow: auto;
  font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.4;
  background: var(--viz-panel);
  border: 1px solid var(--viz-line);
  border-radius: 10px;
  padding: 10px;
  white-space: pre;
}

@media (max-width: 900px) {
  .viz-layout {
    grid-template-columns: 1fr;
  }
}
</style>

<script type="module">
// Import sleap-io.js from CDN
const sleapioUrl = "https://unpkg.com/@talmolab/sleap-io.js@0.1.9/dist/index.js";
const { loadSlp, loadVideo } = await import(sleapioUrl);

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
let videoModel = null;
let frameTimes = null;
let framesByIndex = new Map();
let skeleton = null;
let fps = 30;
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

const buildFrameIndex = () => {
  framesByIndex = new Map();
  let instanceCount = 0;
  for (const frame of labels.labeledFrames) {
    if (!Number.isFinite(frame.frameIdx)) continue;
    framesByIndex.set(frame.frameIdx, frame);
    instanceCount += frame.instances.length;
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

    // Draw track label
    const trackKey = getTrackKey(instance.track);
    if (trackKey != null && instance.points.length > 0) {
      const firstVisible = instance.points.find(p => p.visible && !Number.isNaN(p.xy[0]));
      if (firstVisible) {
        ctx.font = "bold 12px sans-serif";
        ctx.fillStyle = color;
        ctx.fillText(`T${trackKey}`, firstVisible.xy[0] + 8, firstVisible.xy[1] - 8);
      }
    }
  });
};

const renderExternalFrame = async (frameIdx) => {
  if (!ctx || !skeleton) return;
  const frame = framesByIndex.get(frameIdx);
  if (!frame) return;
  const token = ++renderToken;
  const videoFrame = await videoModel?.getFrame(frameIdx);
  if (token !== renderToken) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (videoFrame instanceof ImageBitmap) {
    ctx.drawImage(videoFrame, 0, 0, canvas.width, canvas.height);
  } else if (videoFrame instanceof ImageData) {
    ctx.putImageData(videoFrame, 0, 0);
  }

  drawSkeleton(frame, skeleton);
  formatFrameCoords(frame, skeleton);
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

  drawSkeleton(frame, skeleton);
  formatFrameCoords(frame, skeleton);
};

const updateFpsFromVideo = () => {
  if (frameTimes?.length) return;
  if (!Number.isFinite(videoEl.duration) || videoEl.duration <= 0) return;
  if (!frameCount) return;
  fps = frameCount / videoEl.duration;
};

const getFrameIndexForTime = (time) => {
  if (!Number.isFinite(time)) return 0;
  if (frameTimes?.length) {
    let low = 0;
    let high = frameTimes.length - 1;
    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      const value = frameTimes[mid];
      if (value === time) return mid;
      if (value < time) low = mid + 1;
      else high = mid - 1;
    }
    const idx = Math.min(frameTimes.length - 1, Math.max(0, low));
    const prev = Math.max(0, idx - 1);
    return Math.abs(frameTimes[idx] - time) < Math.abs(frameTimes[prev] - time) ? idx : prev;
  }
  return Math.min(maxFrame, Math.max(0, Math.round(time * fps)));
};

const getTimeForFrameIndex = (frameIdx) => {
  if (frameTimes?.length && frameTimes[frameIdx] != null) return frameTimes[frameIdx];
  return frameIdx / fps;
};

const updateFrameFromVideo = (time = 0) => {
  const frameIdx = getFrameIndexForTime(time);
  if (!Number.isFinite(frameIdx)) return;
  currentFrame = frameIdx;
  seek.value = String(frameIdx);
  frameLabel.textContent = `Frame ${frameIdx}`;
  renderExternalFrame(frameIdx);
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

const playLoop = (timestamp) => {
  if (!isPlaying) return;
  if (!playbackStartTime) playbackStartTime = timestamp;
  const elapsed = (timestamp - playbackStartTime) / 1000;
  const startTime = getTimeForFrameIndex(playbackStartFrame);
  const nextFrame = getFrameIndexForTime(startTime + elapsed);
  if (nextFrame !== currentFrame) {
    updateFrameFromVideo(startTime + elapsed);
  }
  if (currentFrame >= maxFrame) {
    stopPlayback();
    return;
  }
  playHandle = requestAnimationFrame(playLoop);
};

const startPlayback = () => {
  if (isPlaying) return;
  isPlaying = true;
  playbackStartTime = 0;
  playbackStartFrame = currentFrame;
  playHandle = requestAnimationFrame(playLoop);
};

const stopPlayback = () => {
  isPlaying = false;
  if (playHandle) cancelAnimationFrame(playHandle);
  playHandle = null;
};

const hasEmbeddedImages = () => {
  if (!labels?.videos?.length) return false;
  return labels.videos.some((v) => v.backend?.dataset || v.backendMetadata?.dataset);
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

  loadBtn.disabled = true;
  setStatus("Loading SLP...");

  try {
    const useEmbedded = !videoFile;

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
      if (!videoFile) {
        setStatus("No video provided and no embedded images found.");
        loadBtn.disabled = false;
        return;
      }

      const videoSource = URL.createObjectURL(videoFile);

      setMeta({
        frames: frameCount,
        instances: instanceCount,
        nodes: skeleton?.nodes.length ?? 0,
        tracks: labels.tracks.length,
        videos: 1,
        mode: "external video",
      });

      setStatus("Loading video metadata...");
      videoModel = await loadVideo(videoSource);
      frameTimes = await videoModel.getFrameTimes();
      fps = videoModel.fps ?? labels.video?.fps ?? 30;

      setStatus("Loading video...");
      videoEl.style.display = "block";
      playBtn.style.display = "inline-block";
      videoEl.src = videoSource;
      await videoEl.play().catch(() => {});
      videoEl.pause();
      await new Promise((resolve) => {
        if (videoEl.readyState >= 1) return resolve();
        videoEl.addEventListener("loadedmetadata", resolve, { once: true });
      });

      const shape = videoModel?.shape;
      configureCanvas(shape?.[2], shape?.[1]);
      updateFpsFromVideo();
      seek.max = String(maxFrame);
      currentFrame = 0;
      renderToken = 0;
      updateFrameFromVideo(0);
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
</script>

---


## Supported Formats

- **SLP files**: `.slp`, `.h5`, `.hdf5` (SLEAP format)
- **Videos**: `.mp4`, `.avi`, `.mov` (browser-supported formats)

!!! tip "Track Colors"
    Each track ID is assigned a distinct color. The track ID is displayed next to the first visible keypoint of each instance.
