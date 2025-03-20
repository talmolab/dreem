"""Helper functions for calculating mot metrics."""

import numpy as np
import motmetrics as mm
import torch
import pandas as pd
import logging
from typing import Iterable

logger = logging.getLogger("dreem.inference")


def get_matches(frames: list["dreem.io.Frame"]) -> tuple[dict, list, int]:
    """Get comparison between predicted and gt trajectory labels.
    Deprecated.

    Args:
        frames: a list of Frames containing the video_id, frame_id,
            gt labels and predicted labels

    Returns:
        matches: a dict containing predicted and gt trajectory labels
        indices: the frame indices being compared
        video_id: the video being
    """
    matches = {}
    indices = []

    video_id = frames[0].video_id.item()

    if any([frame.has_instances() for frame in frames]):
        for idx, frame in enumerate(frames):
            indices.append(frame.frame_id.item())
            for gt_track_id, pred_track_id in zip(
                frame.get_gt_track_ids(), frame.get_pred_track_ids()
            ):
                match = f"{gt_track_id} -> {pred_track_id}"

                if match not in matches:
                    matches[match] = np.full(len(frames), 0)

                matches[match][idx] = 1
    else:
        logger.debug("No instances detected!")
    return matches, indices, video_id


def get_switches(matches: dict, indices: list) -> dict:
    """Get misassigned predicted trajectory labels.
    Deprecated.

    Args:
        matches: a dict containing the gt and predicted labels
        indices: a list of frame indices being used

    Returns:
        A dict of dicts containing the frame at which the switch occured
        and the change in labels
    """
    track, switches = {}, {}
    if len(matches) > 0 and len(indices) > 0:
        matches_key = np.array(list(matches.keys()))
        matches = np.array(list(matches.values()))
        num_frames = matches.shape[1]

        assert num_frames == len(indices)

        for i, idx in zip(range(num_frames), indices):
            switches[idx] = {}

            col = matches[:, i]
            match_indices = np.where(col == 1)[0]
            match_i = [
                (m.split(" ")[0], m.split(" ")[-1]) for m in matches_key[match_indices]
            ]

            for m in match_i:
                gt, pred = m

                if gt in track and track[gt] != pred:
                    switches[idx][gt] = {
                        "frames": (idx - 1, idx),
                        "pred tracks (from, to)": (track[gt], pred),
                    }

                track[gt] = pred

    return switches


def get_switch_count(switches: dict) -> int:
    """Get the number of mislabeled predicted trajectories.
    Deprecated.

    Args:
        switches: a dict of dicts containing the mislabeled trajectories
            and the frames at which they occur

    Returns:
        the number of switched labels in the video chunk
    """
    only_switches = {k: v for k, v in switches.items() if v != {}}
    sw_cnt = sum([len(i) for i in list(only_switches.values())])
    return sw_cnt


def evaluate(preds, metrics):
    """Evaluate metrics for a list of frames.

    Args:
        preds: list of Frame objects with gt and pred track ids
        metrics: list of metrics to compute

    Returns:
        A dict of metrics with key being the metric, and value being the metric value computed.
    """
    metric_fcn_map = {
        "num_switches": compute_motmetrics,
        "global_tracking_accuracy": compute_global_tracking_accuracy,
    }
    list_frame_info = []
    test_results = {}

    # create gt/pred df
    for frame in preds:
        for instance in frame.instances:
            centroid = instance.centroid["centroid"]
            list_frame_info.append(
                {
                    "frame_id": frame.frame_id.item(),
                    "gt_track_id": instance.gt_track_id.item(),
                    "pred_track_id": instance.pred_track_id.item(),
                    "centroid_x": centroid[0],
                    "centroid_y": centroid[1],
                }
            )

    df = pd.DataFrame(list_frame_info)

    for metric in metrics:
        result = metric_fcn_map[metric](df)
        test_results[metric] = result

    return test_results


def compute_motmetrics(df):
    """Get pymotmetrics summary and mot_events.

    Args:
        df: dataframe with ground truth and predicted centroids matched from match_centroids

    Returns:
        Tuple containing:
        summary_dreem: Motmetrics summary
        acc_dreem.mot_events: Frame by frame MOT events log
    """
    summary_dreem = {}
    acc_dreem = mm.MOTAccumulator(auto_id=True)
    frame_switch_map = {}
    for frame, framedf in df.groupby("frame_id"):
        gt_ids = framedf["gt_track_id"].values
        pred_tracks = framedf["pred_track_id"].values
        # if no matching preds, fill with nan to let motmetrics handle it
        if (pred_tracks == -1).all():
            pred_tracks = np.full(len(gt_ids), np.nan)

        expected_tm_ids = []
        cost_gt_dreem = np.full((len(gt_ids), len(gt_ids)), np.nan)
        np.fill_diagonal(cost_gt_dreem, 1)
        acc_dreem.update(
            oids=gt_ids,
            hids=pred_tracks,
            dists=cost_gt_dreem,
        )

    # get pymotmetrics summary
    mh = mm.metrics.create()
    summary_dreem = mh.compute(acc_dreem, name="acc").transpose()
    motevents = acc_dreem.mot_events.reset_index()
    for idx, row in motevents.iterrows():
        if row["Type"] == "SWITCH":
            frame_switch_map[int(row["FrameId"])] = True
        else:
            frame_switch_map[int(row["FrameId"])] = False

    return summary_dreem, motevents, frame_switch_map


def compute_global_tracking_accuracy(df):
    """Compute global tracking accuracy for each ground truth track. Average the results to get overall accuracy.

    Args:
        df: dataframe with ground truth and predicted centroids and track ids

    Returns:
        gta_by_gt_track_filt: global tracking accuracy for each ground truth track
    """
    track_confusion_dict = {i: [] for i in df.gt_track_id.unique()}
    gt_track_len = df.gt_track_id.value_counts().to_dict()
    gta_by_gt_track = {}

    for idx, row in df.iterrows():
        if ~np.isnan(row["gt_track_id"]) and ~np.isnan(row["pred_track_id"]):
            track_confusion_dict[int(row["gt_track_id"])].append(
                int(row["pred_track_id"])
            )

    for gt_track_id, pred_track_ids in track_confusion_dict.items():
        # Use numpy's mode function to find the most common predicted track ID
        if pred_track_ids:
            # Get the most frequent prediction using numpy's mode
            most_common_pred, count = np.unique(pred_track_ids, return_counts=True)
            gta_by_gt_track[gt_track_id] = np.max(count) / float(
                gt_track_len[gt_track_id]
            )
        else:
            gta_by_gt_track[gt_track_id] = 0

    return gta_by_gt_track
