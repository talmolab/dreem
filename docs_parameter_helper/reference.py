"""Python reference for the parameter helper's rules (docs/js/param-helper/stats.js).

Reads a .slp with sleap-io, writes the plain structure the page builds in the browser
(`<name>.data.json`) and the suggestions these rules give (`<name>.expected.json`), so
`stats.test.mjs` can check the JavaScript against an independent implementation.

    uv run python docs_parameter_helper/reference.py OUT_DIR NAME FILE.slp [--untracked] [--max-frames N]
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import sleap_io as sio

COUNT_SHARE, DIST_FACTOR, ANGLE_FACTOR, CROWDED, MIN_PAIRS, CROP_STEP = (
    0.9,
    1.5,
    1.5,
    0.5,
    20,
    8,
)
FRONT = re.compile(r"head|nose|snout|front|neck|anterior|mouth|beak", re.I)
BACK = re.compile(r"tail|abdomen|back|posterior|rump|hind", re.I)


def plain(labels, untracked, max_frames):
    """The structure `fromSleapIo` builds in the browser, from the first video's frames."""
    nodes = [n.name for n in labels.skeletons[0].nodes]
    video = labels.videos[0]
    frames, tracked = [], False
    for lf in sorted(labels.labeled_frames, key=lambda f: f.frame_idx):
        if lf.video is not video:
            continue
        insts = list(lf.instances)
        if any(not isinstance(i, sio.PredictedInstance) for i in insts):
            insts = [i for i in insts if not isinstance(i, sio.PredictedInstance)]
        out = []
        for inst in insts:
            pts = inst.numpy()
            points = [
                None if np.isnan(p).any() else [float(p[0]), float(p[1])] for p in pts
            ]
            if not any(points):
                continue
            track = (
                None
                if untracked or inst.track is None
                else labels.tracks.index(inst.track)
            )
            tracked |= track is not None
            out.append({"points": points, "track": track})
        frames.append({"frameIdx": int(lf.frame_idx), "instances": out})
        if len(frames) >= max_frames:
            break
    return {
        "kind": "pose" if len(nodes) > 1 else "centroids",
        "nodes": nodes,
        "frames": frames,
        "tracked": tracked,
    }


def centroid(inst):
    """Mean of an instance's visible points (DREEM's `centroid` anchor)."""
    v = [p for p in inst["points"] if p]
    return np.mean(v, axis=0)


def suggest(data):
    """The settings the helper's rules give for one plain structure, with the measurement counts."""
    frames = [f for f in data["frames"] if f["instances"]]
    counts = [len(f["instances"]) for f in data["frames"]]
    vals, cnt = np.unique(counts, return_counts=True)
    best = max(zip(cnt, vals))  # ties -> the larger count
    mode, share = int(best[1]), best[0] / len(counts)

    n_nodes, seen, n_inst, spans, spacing = (
        len(data["nodes"]),
        np.zeros(len(data["nodes"])),
        0,
        [],
        [],
    )
    for f in frames:
        cs = []
        for inst in f["instances"]:
            n_inst += 1
            for i, p in enumerate(inst["points"]):
                if p and i < n_nodes:
                    seen[i] += 1
            c = centroid(inst)
            v = np.array([p for p in inst["points"] if p])
            if len(v) > 1:
                spans.append(2 * np.abs(v - c).max())
            cs.append(c)
        if len(cs) > 1:
            C = np.array(cs)
            d = np.linalg.norm(C[:, None] - C[None], axis=-1)
            d[np.eye(len(C), dtype=bool)] = np.inf
            spacing += d.min(axis=1).tolist()
    vis = seen / max(n_inst, 1)

    def pick(rx, exclude=-1):
        best = -1
        for i, n in enumerate(data["nodes"]):
            if i != exclude and rx.search(n) and (best < 0 or vis[i] > vis[best]):
                best = i
        return best

    front = pick(FRONT)
    back = pick(BACK, front)
    moves, gaps, turns = [], [], []
    if data["tracked"]:
        last = {}
        for f in frames:
            for inst in f["instances"]:
                if inst["track"] is None:
                    continue
                c = centroid(inst)
                fp = inst["points"][front] if front >= 0 else None
                bp = inst["points"][back] if back >= 0 else None
                h = math.atan2(fp[1] - bp[1], fp[0] - bp[0]) if fp and bp else None
                if inst["track"] in last:
                    fi, lc, lh = last[inst["track"]]
                    dt = f["frameIdx"] - fi
                    if dt == 1:
                        moves.append(float(np.linalg.norm(c - lc)))
                        if h is not None and lh is not None:
                            d = abs(h - lh) % (2 * math.pi)
                            turns.append(
                                math.degrees(2 * math.pi - d if d > math.pi else d)
                            )
                    elif dt > 1:
                        gaps.append(dt - 1)
                last[inst["track"]] = (f["frameIdx"], c, h)
    else:
        for a, b in zip(frames, frames[1:]):
            if b["frameIdx"] - a["frameIdx"] != 1:
                continue
            A = [centroid(i) for i in a["instances"]]
            B = [centroid(i) for i in b["instances"]]
            pairs = sorted(
                (float(np.linalg.norm(x - y)), i, j)
                for i, x in enumerate(A)
                for j, y in enumerate(B)
            )
            ua, ub = set(), set()
            for d, i, j in pairs:
                if i not in ua and j not in ub:
                    ua.add(i)
                    ub.add(j)
                    moves.append(d)

    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    crop = math.ceil(q(spans, 50) / CROP_STEP) * CROP_STEP if spans else None
    max_tracks = mode if mode > 0 and share >= COUNT_SHARE else None
    max_dist = (
        max(5, math.ceil(DIST_FACTOR * q(moves, 99) / 5) * 5)
        if len(moves) >= MIN_PAIRS
        else None
    )
    crowded = bool(max_dist and spacing and q(moves, 99) >= CROWDED * q(spacing, 5))
    if max_tracks is not None:
        max_gap = None
    elif data["tracked"] and gaps:
        max_gap = max(1, math.ceil(q(gaps, 99)))
    else:
        max_gap = None
    max_angle = None
    if front >= 0 and back >= 0 and len(turns) >= MIN_PAIRS:
        max_angle = min(180, math.ceil(ANGLE_FACTOR * q(turns, 99) / 5) * 5)
    return {
        "crop": crop,
        "maxTracks": max_tracks,
        "maxDist": max_dist,
        "crowded": crowded,
        "maxGap": max_gap,
        "front": data["nodes"][front] if max_angle else None,
        "back": data["nodes"][back] if max_angle else None,
        "maxAngle": max_angle,
        "moves": len(moves),
        "spans": len(spans),
    }


def main():
    """Export one .slp as a test fixture with its expected suggestions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("name")
    ap.add_argument("slp")
    ap.add_argument(
        "--untracked",
        action="store_true",
        help="drop track identities (tests nearest matching)",
    )
    ap.add_argument("--max-frames", type=int, default=3000)
    a = ap.parse_args()
    data = plain(sio.load_slp(a.slp, open_videos=False), a.untracked, a.max_frames)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}.data.json").write_text(json.dumps(data))
    exp = suggest(data)
    (a.out / f"{a.name}.expected.json").write_text(json.dumps(exp, indent=1))
    print(a.name, exp)


if __name__ == "__main__":
    main()
