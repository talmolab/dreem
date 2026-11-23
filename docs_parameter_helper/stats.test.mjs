// Tests for docs/js/param-helper/stats.js. Run with:
//   FIXTURES=<dir from reference.py> node --test docs_parameter_helper/
// Without FIXTURES only the self-contained unit tests run.
import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

// stats.js is an ES module for the browser; Node reads a .js file outside a "type": "module"
// package as CommonJS, so import a .mjs copy rather than ship a package.json with the docs.
const src = join(dirname(fileURLToPath(import.meta.url)), "..", "docs", "js", "param-helper", "stats.js");
const copy = join(mkdtempSync(join(tmpdir(), "param-helper-")), "stats.mjs");
copyFileSync(src, copy);
const { command, fromMasks, fromSleapIo, measure, quantile, recommend } = await import(pathToFileURL(copy));

test("quantile matches numpy's linear interpolation", () => {
  assert.equal(quantile([1, 2, 3, 4], 0.5), 2.5);
  assert.equal(quantile([10, 0, 5], 0.99), 9.9);
  assert.ok(Number.isNaN(quantile([], 0.5)));
});

test("fromMasks: centroid and centred span per object", () => {
  // 6x4 image: object 1 is a 2x2 block at x 0-1, y 0-1; object 2 a 3x1 bar at x 3-5, y 3.
  const w = 6;
  const data = new Uint16Array(24);
  for (const [x, y] of [[0, 0], [1, 0], [0, 1], [1, 1]]) data[y * w + x] = 1;
  for (const x of [3, 4, 5]) data[3 * w + x] = 2;
  const d = fromMasks([{ width: w, height: 4, data }]);
  const [a, b] = d.frames[0].instances;
  assert.deepEqual(a.points[0], [0.5, 0.5]);
  assert.equal(a.span, 2);
  assert.deepEqual(b.points[0], [4, 3]);
  assert.equal(b.span, 3);
  assert.equal(d.kind, "masks");
});

test("fromSleapIo: user instances win over predictions, invisible points dropped", () => {
  class PredictedInstance {
    constructor(points) {
      this.points = points;
      this.score = 0.9;
    }
  }
  const trackA = { name: "a" };
  const video = {};
  const p = (x, y, visible = true) => ({ xy: [x, y], visible });
  const labels = {
    videos: [video],
    tracks: [trackA],
    skeletons: [{ nodes: [{ name: "head" }, { name: "tail" }] }],
    labeledFrames: [
      { frameIdx: 1, video, instances: [{ points: [p(0, 0), p(4, 0)], track: trackA }, new PredictedInstance([p(9, 9), p(9, 9)])] },
      { frameIdx: 0, video, instances: [new PredictedInstance([p(1, 1), p(2, 2, false)])] },
    ],
  };
  const d = fromSleapIo(labels);
  assert.equal(d.kind, "pose");
  assert.deepEqual(d.frames.map((f) => f.frameIdx), [0, 1]);
  assert.deepEqual(d.frames[0].instances[0].points, [[1, 1], null]);
  assert.equal(d.frames[1].instances.length, 1);
  assert.equal(d.frames[1].instances[0].track, 0);
  assert.ok(d.tracked);
});

test("centroid-only labels ask for a box, then use it", () => {
  const data = {
    kind: "centroids",
    nodes: ["c"],
    tracked: false,
    frames: [0, 1, 2].map((i) => ({ frameIdx: i, instances: [{ points: [[10 + i, 10]], track: null }, { points: [[50, 50]], track: null }] })),
  };
  const m = measure(data);
  assert.equal(recommend(m, data).crop.value, null);
  assert.equal(recommend(m, data, { boxSize: 21 }).crop.value, 24);
  assert.equal(recommend(m, data).maxTracks.value, 2);
  assert.match(command(recommend(m, data)), /--crop-size <SIZE>/);
});

const dir = process.env.FIXTURES;
const names = dir ? readdirSync(dir).filter((f) => f.endsWith(".data.json")).map((f) => f.replace(".data.json", "")) : [];
for (const name of names) {
  test(`matches the Python reference: ${name}`, () => {
    const data = JSON.parse(readFileSync(join(dir, `${name}.data.json`), "utf8"));
    const exp = JSON.parse(readFileSync(join(dir, `${name}.expected.json`), "utf8"));
    const m = measure(data);
    const r = recommend(m, data);
    const got = {
      crop: r.crop.value,
      maxTracks: r.maxTracks.value,
      maxDist: r.maxDist.value,
      crowded: Boolean(r.maxDist.warn),
      maxGap: r.maxGap.value,
      front: r.orientation.front,
      back: r.orientation.back,
      maxAngle: r.orientation.maxAngle,
      moves: m.moves.length,
      spans: m.spans.length,
    };
    assert.deepEqual(got, exp);
    console.log(name, JSON.stringify(got));
  });
}
