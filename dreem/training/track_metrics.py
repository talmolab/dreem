"""Self-contained tracking metrics for training-time diagnostics.

Computes single-alpha HOTA-family metrics (HOTA/AssA/AssPr/AssRe/DetA), IDF1, and
ID switches, plus a predicted-vs-GT confusion matrix, directly from per-instance
``(frame_idx, gt_id, pred_id)`` records. Because DREEM tracks *pre-detected*
instances, every instance already carries both its ground-truth and predicted
track id, so we never need motmetrics' LAP matching (which is where the eval
path's ``KeyError`` comes from) and we add no TrackEval dependency.

Definitions follow the standard formulations:
  * HOTA family — Luiten et al. 2021, single localization threshold. The
    association math (AssA/AssPr/AssRe via TPA/FNA/FPA counts) is the same as the
    ``hota_one_clip`` helper in the larvae star scripts.
  * IDF1 — Ristani et al. 2016 (optimal 1-1 identity matching).
  * ID switches — CLEAR-MOT: per GT track, the predicted id changing between
    consecutive frames.

`pred_id == -1` means an instance the tracker left untracked (e.g. confidence
thresholded) -> counted as a missed detection (lowers DetA/IDF1), not matched.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


def compute_track_metrics(records: list[tuple[int, int, int]]) -> dict | None:
    """Tracking metrics from per-instance (frame_idx, gt_id, pred_id) records.

    Args:
        records: one tuple per tracked instance: (frame index in the video,
            ground-truth track id, predicted track id). gt_id < 0 (unlabeled) is
            dropped; pred_id == -1 counts as a missed detection.

    Returns:
        Dict with scalars ``hota, assa, asspr, assre, deta, idf1, num_switches,
        n_instances`` and confusion data ``confusion`` (pred-by-gt count matrix,
        numpy array), ``gt_ids``, ``pred_ids`` (sorted label lists), and
        ``gt_to_pred`` (optimal gt->pred id alignment). None if there is no
        usable data.
    """
    import numpy as np

    labeled = [(int(f), int(g), int(p)) for f, g, p in records if int(g) >= 0]
    if not labeled:
        return None
    matched = [(f, g, p) for f, g, p in labeled if p != -1]
    n_total = len(labeled)  # all GT detections
    if not matched:
        return None

    g_seq = [g for _, g, _ in matched]
    h_seq = [p for _, _, p in matched]
    tp = len(matched)

    pair = Counter(zip(g_seq, h_seq))  # co-occurrence of (gt, pred)
    gc = Counter(g_seq)  # matched detections per gt track
    hc = Counter(h_seq)  # detections per pred track
    gt_ids = sorted(gc)
    pred_ids = sorted(hc)

    # --- HOTA family (single alpha) ---
    # DetA: fraction of GT detections that were tracked (fp assumed 0 since preds
    # only exist on GT instances in DREEM).
    det_a = tp / n_total
    ass = aspr = asre = 0.0
    for g, h in zip(g_seq, h_seq):
        tpa = pair[(g, h)]
        fna = gc[g] - tpa
        fpa = hc[h] - tpa
        ass += tpa / (tpa + fna + fpa)
        aspr += tpa / (tpa + fpa)
        asre += tpa / (tpa + fna)
    ass_a = ass / tp
    hota = math.sqrt(det_a * ass_a)

    # --- confusion matrix (rows=pred, cols=gt) + optimal gt<->pred alignment ---
    from scipy.optimize import linear_sum_assignment

    gi = {g: i for i, g in enumerate(gt_ids)}
    pidx = {p: i for i, p in enumerate(pred_ids)}
    conf = np.zeros((len(pred_ids), len(gt_ids)), dtype=float)
    for (g, h), c in pair.items():
        conf[pidx[h], gi[g]] = c
    row, col = linear_sum_assignment(-conf)  # maximize matched detections
    idtp = float(conf[row, col].sum())
    total_pred = tp
    idf1 = 2 * idtp / (n_total + total_pred) if (n_total + total_pred) else 0.0
    gt_to_pred = {gt_ids[c]: pred_ids[r] for r, c in zip(row, col)}

    # --- ID switches (CLEAR-MOT): per GT track, pred id changes over frames ---
    by_gt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for f, g, p in matched:
        by_gt[g].append((f, p))
    num_switches = 0
    for seq in by_gt.values():
        seq.sort()
        prev = None
        for _, p in seq:
            if prev is not None and p != prev:
                num_switches += 1
            prev = p

    return dict(
        hota=hota,
        assa=ass_a,
        asspr=aspr / tp,
        assre=asre / tp,
        deta=det_a,
        idf1=float(idf1),
        num_switches=int(num_switches),
        n_instances=n_total,
        confusion=conf,
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        gt_to_pred=gt_to_pred,
    )
