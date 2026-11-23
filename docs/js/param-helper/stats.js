/**
 * The numbers behind the "Choosing Parameters" page: pure functions, no DOM, so they can be
 * tested outside the browser (`node --test docs_parameter_helper/`).
 *
 * Data flows as one plain structure, whatever the input was:
 *   {
 *     kind: "pose" | "centroids" | "masks",
 *     nodes: [name, ...],                     // skeleton nodes ([] for masks)
 *     frames: [{frameIdx, instances: [{points: [[x, y] | null, ...], track, span?}]}],
 *     tracked: bool,                          // instances carry track identities
 *   }
 * `span` (masks only) is the side of the smallest square centred on the object that holds it.
 */

/** Tunable rules, named so the page can quote them. */
export const RULES = {
  countShare: 0.9, // the most common count must cover this share of frames to set --max-tracks
  distFactor: 1.5, // --max-dist = this x the 99th percentile movement per frame
  angleFactor: 1.5, // --max-angle = this x the 99th percentile turn per frame
  crowdedRatio: 0.5, // warn when that movement reaches this share of the close spacing (5th pct)
  minPairs: 20, // consecutive-frame measurements needed before suggesting a distance or angle
  cropStep: 8, // crop sizes are rounded up to a multiple of this
};

const FRONT = /head|nose|snout|front|neck|anterior|mouth|beak/i;
const BACK = /tail|abdomen|back|posterior|rump|hind/i;

/** Linear-interpolated quantile, as numpy's default. */
export function quantile(values, q) {
  if (!values.length) return NaN;
  const s = Float64Array.from(values).sort();
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

export const roundUp = (x, step) => Math.ceil(x / step) * step;

/** Mean of an instance's visible points, or null. */
export function centroidOf(inst) {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const p of inst.points) {
    if (!p) continue;
    sx += p[0];
    sy += p[1];
    n += 1;
  }
  return n ? [sx / n, sy / n] : null;
}

/**
 * Where DREEM centres the crop: "centroid" (or "midpoint") = mean of the visible points;
 * a node index = that point, falling back to the centroid when it is missing, as DREEM's
 * default `handle_missing: "centroid"` does.
 */
export function anchorOf(inst, anchor = "centroid") {
  if (anchor === "centroid" || anchor === "midpoint") return centroidOf(inst);
  return inst.points[anchor] ?? centroidOf(inst);
}

/** Side of the smallest anchor-centred square holding every visible point (pose data). */
export function poseSpan(inst, anchor = "centroid") {
  const c = anchorOf(inst, anchor);
  let half = 0;
  let n = 0;
  for (const p of inst.points) {
    if (!p) continue;
    half = Math.max(half, Math.abs(p[0] - c[0]), Math.abs(p[1] - c[1]));
    n += 1;
  }
  return n > 1 ? 2 * half : null;
}

/**
 * sleap-io.js Labels -> the plain structure. One video at a time (index into labels.videos).
 * A frame holding user-labelled instances keeps only those, as SLEAP does when both a user
 * and a predicted instance exist for the same animal.
 */
export function fromSleapIo(labels, videoIndex = 0) {
  const video = labels.videos?.[videoIndex];
  const nodes = (labels.skeletons?.[0]?.nodes ?? []).map((n) => n.name ?? String(n));
  const trackKey = (t) => {
    if (t == null) return null;
    const i = labels.tracks?.indexOf(t) ?? -1;
    return i >= 0 ? i : (t.name ?? t);
  };
  // sleap-io.js 0.1.9 bundles the class as `_PredictedInstance`; only predictions have `score`.
  const isPredicted = (inst) =>
    /PredictedInstance$/.test(inst.constructor?.name ?? "") || typeof inst.score === "number";
  const frames = [];
  let tracked = false;
  for (const lf of labels.labeledFrames ?? []) {
    if (labels.videos?.length > 1 && lf.video !== video) continue;
    if (!Number.isFinite(lf.frameIdx)) continue;
    let insts = lf.instances ?? [];
    if (insts.some((i) => !isPredicted(i))) insts = insts.filter((i) => !isPredicted(i));
    const instances = [];
    for (const inst of insts) {
      const points = (inst.points ?? []).map((p) =>
        p && p.visible !== false && p.xy && Number.isFinite(p.xy[0]) && Number.isFinite(p.xy[1])
          ? [p.xy[0], p.xy[1]]
          : null,
      );
      if (!points.some(Boolean)) continue;
      const track = trackKey(inst.track);
      if (track !== null) tracked = true;
      instances.push({ points, track });
    }
    frames.push({ frameIdx: lf.frameIdx, instances });
  }
  frames.sort((a, b) => a.frameIdx - b.frameIdx);
  return { kind: nodes.length > 1 ? "pose" : "centroids", nodes, frames, tracked };
}

/**
 * Segmentation masks (one labelled image per frame: 0 = background, each object its own id)
 * -> the plain structure. Each object is a centroid plus its span (the side of the smallest
 * centroid-centred square holding its bounding box), as DREEM crops CTC objects around their
 * centroids. Label ids are not treated as identities: a segmentation's ids are often
 * renumbered every frame.
 */
export function fromMasks(images) {
  const frames = images.map((img, frameIdx) => {
    const { width, data } = img;
    const acc = new Map(); // id -> [sumX, sumY, n, x0, y0, x1, y1]
    for (let i = 0; i < data.length; i++) {
      const id = data[i];
      if (!id) continue;
      const x = i % width;
      const y = (i - x) / width;
      let a = acc.get(id);
      if (!a) acc.set(id, (a = [0, 0, 0, x, y, x, y]));
      a[0] += x;
      a[1] += y;
      a[2] += 1;
      if (x < a[3]) a[3] = x;
      if (y < a[4]) a[4] = y;
      if (x > a[5]) a[5] = x;
      if (y > a[6]) a[6] = y;
    }
    const instances = [];
    for (const [sx, sy, n, x0, y0, x1, y1] of acc.values()) {
      const cx = sx / n;
      const cy = sy / n;
      // pixel i covers [i - 0.5, i + 0.5] around its centre, the coordinates the centroid uses
      const half = Math.max(cx - x0 + 0.5, x1 + 0.5 - cx, cy - y0 + 0.5, y1 + 0.5 - cy);
      instances.push({ points: [[cx, cy]], track: null, span: 2 * half });
    }
    return { frameIdx, instances };
  });
  return { kind: "masks", nodes: [], frames, tracked: false };
}

/** Pairs each instance of frame a with one of frame b, closest pairs first (untracked data). */
function greedyPairs(a, b) {
  const pairs = [];
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++)
      pairs.push([Math.hypot(a[i][0] - b[j][0], a[i][1] - b[j][1]), i, j]);
  pairs.sort((p, q) => p[0] - q[0]);
  const usedA = new Set();
  const usedB = new Set();
  const out = [];
  for (const [d, i, j] of pairs) {
    if (usedA.has(i) || usedB.has(j)) continue;
    usedA.add(i);
    usedB.add(j);
    out.push(d);
  }
  return out;
}

/** Index of the most visible node whose name matches re, or -1. */
function pickNode(nodes, visibility, re, exclude = -1) {
  let best = -1;
  nodes.forEach((name, i) => {
    if (i === exclude || !re.test(name)) return;
    if (best < 0 || visibility[i] > visibility[best]) best = i;
  });
  return best;
}

/** Everything the suggestions are made from. */
export function measure(data, { anchor = "centroid" } = {}) {
  const frames = data.frames.filter((f) => f.instances.length);
  const counts = data.frames.map((f) => f.instances.length);
  const hist = new Map();
  let cMin = Infinity;
  let cMax = -Infinity;
  for (const c of counts) {
    hist.set(c, (hist.get(c) ?? 0) + 1);
    cMin = Math.min(cMin, c);
    cMax = Math.max(cMax, c);
  }
  let mode = 0;
  let modeN = -1;
  for (const [c, n] of hist) if (n > modeN || (n === modeN && c > mode)) [mode, modeN] = [c, n];

  const nNodes = data.nodes.length;
  const seen = new Array(nNodes).fill(0);
  let nInst = 0;
  const spans = [];
  const spacing = [];
  for (const f of frames) {
    const centres = [];
    for (const inst of f.instances) {
      nInst += 1;
      inst.points.forEach((p, i) => {
        if (p && i < nNodes) seen[i] += 1;
      });
      const span = data.kind === "masks" ? inst.span : poseSpan(inst, anchor);
      if (span != null) spans.push(span);
      const c = anchorOf(inst, anchor);
      if (c) centres.push(c);
    }
    for (let i = 0; i < centres.length; i++) {
      let nn = Infinity;
      for (let j = 0; j < centres.length; j++)
        if (j !== i) nn = Math.min(nn, Math.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1]));
      if (Number.isFinite(nn)) spacing.push(nn);
    }
  }
  const visibility = seen.map((s) => (nInst ? s / nInst : 0));

  // Movement, gaps and turns between consecutive frames.
  const front = pickNode(data.nodes, visibility, FRONT);
  const back = pickNode(data.nodes, visibility, BACK, front);
  const heading = (inst) => {
    const f = inst.points[front];
    const b = inst.points[back];
    return f && b ? Math.atan2(f[1] - b[1], f[0] - b[0]) : null;
  };
  const moves = [];
  const gaps = [];
  const turns = [];
  if (data.tracked) {
    const last = new Map(); // track -> [frameIdx, anchor, heading]
    for (const f of frames) {
      for (const inst of f.instances) {
        if (inst.track == null) continue;
        const c = anchorOf(inst, anchor);
        const h = front >= 0 && back >= 0 ? heading(inst) : null;
        const prev = last.get(inst.track);
        if (prev) {
          const dt = f.frameIdx - prev[0];
          if (dt === 1) {
            moves.push(Math.hypot(c[0] - prev[1][0], c[1] - prev[1][1]));
            if (h != null && prev[2] != null) {
              let d = Math.abs(h - prev[2]) % (2 * Math.PI);
              if (d > Math.PI) d = 2 * Math.PI - d;
              turns.push((d * 180) / Math.PI);
            }
          } else if (dt > 1) {
            gaps.push(dt - 1);
          }
        }
        last.set(inst.track, [f.frameIdx, c, h]);
      }
    }
  } else {
    for (let k = 1; k < frames.length; k++) {
      if (frames[k].frameIdx - frames[k - 1].frameIdx !== 1) continue;
      const a = frames[k - 1].instances.map((i) => anchorOf(i, anchor));
      const b = frames[k].instances.map((i) => anchorOf(i, anchor));
      moves.push(...greedyPairs(a, b));
    }
  }

  return {
    frames: data.frames.length,
    counts: { mode, share: counts.length ? modeN / counts.length : 0, min: cMin, max: cMax },
    spans,
    spacing,
    moves,
    moveSource: data.tracked ? "tracks" : "matched",
    gaps,
    turns,
    visibility,
    front,
    back,
  };
}

const fmt = (x) => (Number.isFinite(x) ? (x >= 10 ? Math.round(x) : Math.round(x * 10) / 10) : "?");

/**
 * Suggested settings with a one-line reason each. `opts.domain` ("animals" | "cells") picks the
 * pretrained model; `opts.boxSize` (pixels) is a box the user drew around one object, used for
 * the crop when the labels cannot give a size.
 */
export function recommend(m, data, { anchor = "centroid", domain, boxSize } = {}) {
  const out = {};
  const animals = domain ? domain === "animals" : data.kind === "pose";
  out.model = {
    value: animals ? "animals" : "microscopy",
    why:
      data.kind === "pose"
        ? `${data.nodes.length}-node skeleton: animal pose data.`
        : data.kind === "masks"
          ? "Segmentation masks: cell data."
          : `Centroid-only labels: pick animals or cells above (${animals ? "animals" : "cells"} chosen).`,
  };

  // Crop: cover the whole object.
  if (m.spans.length) {
    const p50 = quantile(m.spans, 0.5);
    const p95 = quantile(m.spans, 0.95);
    out.crop = {
      value: roundUp(p50, RULES.cropStep),
      why: `Covers the whole ${animals ? "animal" : "object"}: median size ${fmt(p50)} px (95% of them within ${fmt(p95)} px), rounded up to a multiple of ${RULES.cropStep}.`,
    };
  } else if (boxSize) {
    out.crop = {
      value: roundUp(boxSize, RULES.cropStep),
      why: `From the box you drew: ${fmt(boxSize)} px, rounded up to a multiple of ${RULES.cropStep}.`,
    };
  } else {
    const hint = m.spacing.length ? ` Objects are about ${fmt(quantile(m.spacing, 0.5))} px apart, an upper bound for a single object.` : "";
    out.crop = {
      value: null,
      why: `Centroid labels carry no size. Add your video and drag a box around one object.${hint}`,
    };
  }
  if (boxSize && m.spans.length) out.crop.why += ` (Your box: ${fmt(boxSize)} px.)`;

  // Max tracks: a fixed number of objects.
  const { mode, share, min, max } = m.counts;
  if (mode > 0 && share >= RULES.countShare) {
    out.maxTracks = {
      value: mode,
      why: `${mode} per frame in ${Math.round(share * 100)}% of frames: a fixed number of ${animals ? "animals" : "objects"}.`,
    };
  } else {
    out.maxTracks = {
      value: null,
      why: `The number changes from ${min} to ${max} per frame (objects entering, leaving or dividing). Leave it unset so new tracks can start.`,
    };
  }

  // Max distance from movement per frame.
  if (m.moves.length >= RULES.minPairs) {
    const p99 = quantile(m.moves, 0.99);
    const close = m.spacing.length ? quantile(m.spacing, 0.05) : NaN;
    const src = m.moveSource === "tracks" ? "tracked" : "nearest-match (no tracks in the file)";
    out.maxDist = {
      value: Math.max(5, roundUp(RULES.distFactor * p99, 5)),
      why: `99% of moves between consecutive frames are under ${fmt(p99)} px (${src}); ${RULES.distFactor}x that, rounded up to 5.`,
    };
    if (Number.isFinite(close) && p99 >= RULES.crowdedRatio * close)
      out.maxDist.warn = `Objects move up to ${fmt(p99)} px per frame but can be ${fmt(close)} px apart, so distance alone cannot tell them apart. Keep --max-dist-multiplier at its default.`;
  } else {
    out.maxDist = { value: null, why: `Needs at least ${RULES.minPairs} moves between consecutive labelled frames; found ${m.moves.length}.` };
  }

  // Max gap.
  if (out.maxTracks.value != null) {
    out.maxGap = { value: null, why: "A fixed number of animals: tracks never need to end, and --max-tracks makes the tracker reuse them. Leave it unset." };
  } else if (data.tracked && m.gaps.length) {
    const p99 = quantile(m.gaps, 0.99);
    out.maxGap = { value: Math.max(1, Math.ceil(p99)), why: `99% of gaps in your tracks are ${fmt(p99)} frames or shorter.` };
  } else if (data.tracked) {
    out.maxGap = { value: null, why: "No track in these labels has a gap. Leave it unset, or set a small value to end tracks of objects that leave." };
  } else {
    out.maxGap = { value: null, why: "Gaps can only be measured from tracked labels (e.g. proofread ground truth)." };
  }

  // Anchor.
  const steady = data.nodes.filter((_, i) => m.visibility[i] >= 0.99);
  out.anchor = {
    value: anchor === "centroid" ? "centroid" : data.nodes[anchor],
    why:
      anchor === "centroid"
        ? `The mean of the visible keypoints: defined whenever any keypoint is.${steady.length && data.nodes.length > 1 ? ` Nodes visible in 99% of instances: ${steady.join(", ")}.` : ""}`
        : `Node "${data.nodes[anchor]}", visible in ${Math.round(m.visibility[anchor] * 100)}% of instances (the centroid is used when it is missing).`,
  };

  // Orientation.
  if (m.front >= 0 && m.back >= 0) {
    const pair = `${data.nodes[m.front]} -> ${data.nodes[m.back]}`;
    if (m.turns.length >= RULES.minPairs) {
      const p99 = quantile(m.turns, 0.99);
      out.orientation = {
        front: data.nodes[m.front],
        back: data.nodes[m.back],
        maxAngle: Math.min(180, roundUp(RULES.angleFactor * p99, 5)),
        why: `Body axis ${pair}; 99% of turns between consecutive frames are under ${fmt(p99)} degrees, ${RULES.angleFactor}x that.`,
      };
    } else {
      out.orientation = {
        front: null,
        back: null,
        maxAngle: null,
        why: `Body axis ${pair} found, but turning can only be measured from tracked labels; orientation filtering stays off without --max-angle.`,
      };
    }
  } else {
    out.orientation = { front: null, back: null, maxAngle: null, why: data.kind === "pose" ? "No head- or tail-like node names." : "Needs pose keypoints." };
  }
  return out;
}

/** The `dreem track` command for a set of suggestions. */
export function command(rec, { input = "./data", output = "./results" } = {}) {
  const parts = [`dreem track ${input}`, `--checkpoint ${rec.model.value}`, `--output ${output}`];
  parts.push(`--crop-size ${rec.crop.value ?? "<SIZE>"}`);
  if (rec.anchor.value && rec.anchor.value !== "centroid") parts.push(`--anchor ${rec.anchor.value}`);
  if (rec.maxTracks.value != null) parts.push(`--max-tracks ${rec.maxTracks.value}`);
  if (rec.maxDist.value != null) parts.push(`--max-dist ${rec.maxDist.value}`);
  if (rec.maxGap.value != null) parts.push(`--max-gap ${rec.maxGap.value}`);
  const o = rec.orientation;
  if (o.maxAngle != null) parts.push(`--front-node ${o.front}`, `--back-node ${o.back}`, `--max-angle ${o.maxAngle}`);
  return parts.join(" \\\n    ");
}
