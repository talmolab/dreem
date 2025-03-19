"""Helper functions for calculating mot metrics."""

import numpy as np
import motmetrics as mm
import torch
import pandas as pd
import logging
from typing import Iterable

logger = logging.getLogger("dreem.inference")

# from dreem.inference.post_processing import _pairwise_iou
# from dreem.inference.boxes import Boxes


def get_matches(frames: list["dreem.io.Frame"]) -> tuple[dict, list, int]:
    """Get comparison between predicted and gt trajectory labels.

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

    Args:
        switches: a dict of dicts containing the mislabeled trajectories
            and the frames at which they occur

    Returns:
        the number of switched labels in the video chunk
    """
    only_switches = {k: v for k, v in switches.items() if v != {}}
    sw_cnt = sum([len(i) for i in list(only_switches.values())])
    return sw_cnt


def to_track_eval(frames: list["dreem.io.Frame"]) -> dict:
    """Reformats frames the output from `sliding_inference` to be used by `TrackEval`.
    Deprecated.

    Args:
        frames: A list of Frames. `See dreem.io.data_structures for more info`.

    Returns:
        data: A dictionary. Example provided below.

    # --------------------------- An example of data --------------------------- #

    *: number of ids for gt at every frame of the video
    ^: number of ids for tracker at every frame of the video
    L: length of video

    data = {
        "num_gt_ids": total number of unique gt ids,
        "num_tracker_dets": total number of detections by your detection algorithm,
        "num_gt_dets": total number of gt detections,
        "gt_ids": (L, *),  # Ragged np.array
        "tracker_ids": (L, ^),  # Ragged np.array
        "similarity_scores": (L, *, ^),  # Ragged np.array
        "num_timesteps": L,
    }
    """
    unique_gt_ids = []
    num_tracker_dets = 0
    num_gt_dets = 0
    gt_ids = []
    track_ids = []
    similarity_scores = []

    data = {}
    cos_sim = torch.nn.CosineSimilarity()

    for fidx, frame in enumerate(frames):
        gt_track_ids = frame.get_gt_track_ids().cpu().numpy().tolist()
        pred_track_ids = frame.get_pred_track_ids().cpu().numpy().tolist()
        # boxes = Boxes(frame.get_bboxes().cpu())

        gt_ids.append(np.array(gt_track_ids))
        track_ids.append(np.array(pred_track_ids))

        num_tracker_dets += len(pred_track_ids)
        num_gt_dets += len(gt_track_ids)

        if not set(gt_track_ids).issubset(set(unique_gt_ids)):
            unique_gt_ids.extend(list(set(gt_track_ids).difference(set(unique_gt_ids))))

        # eval_matrix = _pairwise_iou(boxes, boxes)
        eval_matrix = np.full((len(gt_track_ids), len(pred_track_ids)), np.nan)

        for i, feature_i in enumerate(frame.get_features()):
            for j, feature_j in enumerate(frame.get_features()):
                eval_matrix[i][j] = cos_sim(
                    feature_i.unsqueeze(0), feature_j.unsqueeze(0)
                )

        # eval_matrix
        #                      pred_track_ids
        #                            0        1
        #  gt_track_ids    1        ...      ...
        #                  0        ...      ...
        #
        # Since the order of both gt_track_ids and pred_track_ids matter (maps from pred to gt),
        # we know the diagonal is the important part. E.g. gt_track_ids=1 maps to pred_track_ids=0
        # and gt_track_ids=0 maps to pred_track_ids=1 because they are ordered in that way.

        # Based on assumption that eval_matrix is always a square matrix.
        # This is true because we are using ground-truth detections.
        #
        # - The number of predicted tracks for a frame will always be the same number
        # of ground truth tracks for a frame.
        # - The number of predicted and ground truth detections will always be the same
        # for any frame.
        # - Because we map detections to features one-to-one, there will always be the same
        # number of features for both predicted and ground truth for any frame.

        # Mask upper and lower triangles of the square matrix (set to 0).
        eval_matrix = np.triu(np.tril(eval_matrix))

        # Replace the 0s with np.nans.
        i, j = np.where(eval_matrix == 0)
        eval_matrix[i, j] = np.nan

        similarity_scores.append(eval_matrix)

    data["num_gt_ids"] = len(unique_gt_ids)
    data["num_tracker_dets"] = num_tracker_dets
    data["num_gt_dets"] = num_gt_dets
    data["gt_ids"] = gt_ids
    data["tracker_ids"] = track_ids
    data["similarity_scores"] = similarity_scores
    data["num_timesteps"] = len(frames)

    return data


def get_track_evals(data: dict, metrics: dict) -> dict:
    """Run track_eval and get mot metrics.
    Deprecated.
    Args:
        data: A dictionary. Example provided below.
        metrics: mot metrics to be computed
    Returns:
        A dictionary with key being the metric, and value being the metric value computed.
    # --------------------------- An example of data --------------------------- #

    *: number of ids for gt at every frame of the video
    ^: number of ids for tracker at every frame of the video
    L: length of video

    data = {
        "num_gt_ids": total number of unique gt ids,
        "num_tracker_dets": total number of detections by your detection algorithm,
        "num_gt_dets": total number of gt detections,
        "gt_ids": (L, *),  # Ragged np.array
        "tracker_ids": (L, ^),  # Ragged np.array
        "similarity_scores": (L, *, ^),  # Ragged np.array
        "num_timsteps": L,
    }
    """
    results = {}
    for metric_name, metric in metrics.items():
        result = metric.eval_sequence(data)
        results.update(result)
    return results

def evaluate(test_results, metrics):
    """Evaluate metrics for a list of frames.

    Args:
        test_results: dict containing predictions and metrics to be filled out
        metrics: list of metrics to compute

    Returns:
        A dict of metrics with key being the metric, and value being the metric value computed.
    """
    metric_fcn_map = {
        "num_switches": compute_motmetrics,
        "global_tracking_accuracy": compute_global_tracking_accuracy,
    }
    preds = test_results["preds"]
    list_frame_info = []

    # create gt/pred df
    for frame in preds:
        for instance in frame.instances:
            centroid = np.nanmean(instance.numpy(), axis=0).round()
            list_frame_info.append({
                "frame_id": frame.frame_id,
                "gt_track_id": instance.track_id,
                "pred_track_id": instance.pred_track_id,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1]
            })

    df = pd.DataFrame(list_frame_info)

    for metric in metrics:
        result = metric_fcn_map[metric](df)
        test_results["metrics"][metric] = result
    
    return test_results


def compute_motmetrics(df):
    """Get pymotmetrics summary and mot_events.

    Args:
        df: dataframe with ground truth and predicted centroids matched from match_centroids

    Returns:
        summary_dreem: Motmetrics summary
        acc_dreem.mot_events: Frame by frame MOT events log
    """
    summary_dreem = {}
    for frame, framedf in df.groupby('frame_id'):
        gt_ids = framedf['track_id'].values
        pred_tracks = framedf['pred_track_id'].values
        # if no matching preds, fill with nan to let motmetrics handle it
        if (pred_tracks==-1).all():
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

    return summary_dreem, acc_dreem.mot_events


def compute_global_tracking_accuracy(df):
    """Compute global tracking accuracy for each ground truth track. Average the results to get overall accuracy.

    Args:
        df: dataframe with ground truth and predicted centroids and track ids

    Returns:
        gta_by_gt_track_filt: global tracking accuracy for each ground truth track
    """
    # sometimes some gt ids are skipped so track_id.max() > track_id.unique()
    # track_ids are 1-indexed
    track_confusion_matrix = np.zeros((df.track_id.max() + 1, df.pred_track_id.max() + 1))
    gt_track_len = {i: 0 for i in range(track_confusion_matrix.shape[0])} # same shape as gt track ids
    gt_track_len.update(df.track_id.value_counts().to_dict())
    for idx, row in df.iterrows():
        if ~np.isnan(row['gt_track_id']) and ~np.isnan(row['pred_track_id']): 
            track_confusion_matrix[row['gt_track_id'], row['pred_track_id']] += 1
    
    gta_by_gt_track = (100 * track_confusion_matrix.max(axis=1) / gt_track_len)
    # Filter out rows that are all null; this is when gt tracks aren't consecutive but the track confusion matrix still has those rows
    gta_by_gt_track_filt = gta_by_gt_track[~np.isnan(gta_by_gt_track).all(axis=1)]
                        
    return gta_by_gt_track_filt