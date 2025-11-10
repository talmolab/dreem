"""Helper functions for calculating mot metrics."""

import logging
from typing import TYPE_CHECKING

import motmetrics as mm
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dreem.io import Frame

logger = logging.getLogger("dreem.inference")


def get_matches(frames: list["Frame"]) -> tuple[dict, list, int]:
    """Get comparison between predicted and gt trajectory labels. Deprecated.

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
    """Get misassigned predicted trajectory labels. Deprecated.

    Args:
        matches: a dict containing the gt and predicted labels
        indices: a list of frame indices being used

    Returns:
        A dict of dicts containing the frame at which the switch occurred
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
    """Get the number of mislabeled predicted trajectories. Deprecated.

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
        "motmetrics": compute_motmetrics,
        "global_tracking_accuracy": compute_global_tracking_accuracy,
    }
    list_frame_info = []
    test_results = {}

    # create gt/pred df
    for frame in preds:
        for instance in frame.instances:
            anchor = instance.anchor[0]
            if anchor in instance.centroid:
                centroid = instance.centroid[anchor]
            else:  # if for some reason the anchor is not in the centroid dict, use the first key-value pair
                for key, value in instance.centroid.items():
                    centroid = value
                    break
            centroid_x, centroid_y = centroid[0], centroid[1]
            list_frame_info.append(
                {
                    "frame_id": frame.frame_id.item(),
                    "gt_track_id": instance.gt_track_id.item(),
                    "pred_track_id": instance.pred_track_id.item(),
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
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
    track_name_mapping = {}
    curr_track = 0
    for trk in df["gt_track_id"].unique():
        track_name_mapping[trk] = curr_track
        curr_track += 1
    # filter out -1 track_ids; these are untracked instances due to confidence thresholding
    df = df[df["pred_track_id"] != -1]
    preds_motevents_map = {}
    motevents_frame_id_map = {}
    motevents_frame_id = 0
    for frame, framedf in df.groupby("frame_id"):
        # if a frame has no preds, motevents just enumerates in order, leading to mismatch in frame ids
        preds_motevents_map[frame] = motevents_frame_id
        motevents_frame_id_map[motevents_frame_id] = frame
        motevents_frame_id += 1
        gt_ids = framedf["gt_track_id"].values
        gt_ids = [track_name_mapping[trk] for trk in gt_ids]
        pred_tracks = framedf["pred_track_id"].values
        # if no matching preds, fill with nan to let motmetrics handle it
        if (pred_tracks == -1).all():
            pred_tracks = np.full(len(gt_ids), np.nan)

        # expected_tm_ids = []  # Currently unused but may be needed for future TRA metric calculations
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
    switch_frames = []
    for frame_id in sorted(df["frame_id"].unique()):
        frame_switch_map[frame_id] = (
            False  # just populate with false for all frames at first
        )
        motevent = motevents[motevents["FrameId"] == preds_motevents_map[frame_id]]
        if motevent.empty:  # if no assigned instances in this frame, skip
            continue
        if (motevent["Type"] == "SWITCH").any():
            switch_frames.append(frame_id)

    for i, switch_frame in enumerate(switch_frames):
        frame_switch_map[switch_frame] = True
    for idx, row in motevents.iterrows():
        motevents.loc[idx, "FrameId"] = motevents_frame_id_map[row["FrameId"]]

    return summary_dreem, frame_switch_map, motevents


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
