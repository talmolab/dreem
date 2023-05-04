"""
Helper functions for calculating mot metrics
"""
import numpy as np


def get_matches(instances: list[dict]) -> tuple[dict, list, int]:
    """Get comparison between predicted and gt trajectory labels
    Args:
        instances: a list of dicts where each dict corresponds to a frame and contains the video_id, frame_id, gt labels and predicted labels
    Returns:
        matches: a dict containing predicted and gt trajectory labels
        indices: the frame indices being compared
        video_id: the video being
    """
    matches = {}
    indices = []

    video_id = instances[0]["video_id"].item()

    for idx, instance in enumerate(instances):
        indices.append(instance["frame_id"].item())
        for i, gt_track_id in enumerate(instance["gt_track_ids"]):
            gt_track_id = instance["gt_track_ids"][i]
            pred_track_id = instance["pred_track_ids"][i]
            match = f"{gt_track_id} -> {pred_track_id}"

            if match not in matches:
                matches[match] = np.full(len(instances), 0)

            matches[match][idx] = 1
    return matches, indices, video_id


def get_switches(matches: dict, indices: list) -> dict:
    """Get misassigned predicted trajectory labels
    Args:
        matches: a dict containing the gt and predicted labels
        indices: a list of frame indices being used
    Returns: A dict of dicts containing the frame at which the switch occured and the change in labels
    """
    track, switches = {}, {}
    # unique_gt_ids = np.unique([k.split(" ")[0] for k in list(matches.keys())])
    matches_key = np.array(list(matches.keys()))
    matches = np.array(list(matches.values()))
    num_frames = matches.shape[1]

    assert num_frames == len(indices)

    for i, idx in zip(range(num_frames), indices):
        switches[idx] = {}

        col = matches[:, i]
        indices = np.where(col == 1)[0]
        match_i = [(m.split(" ")[0], m.split(" ")[-1]) for m in matches_key[indices]]

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
    """
    Get the number of mislabeled predicted trajectories
    Returns: the number of switched labels in the video chunk
    Args:
        switches: a dict of dicts containing the mislabeled trajectories and the frames at which they occur
    """
    only_switches = {k: v for k, v in switches.items() if v != {}}
    sw_cnt = sum([len(i) for i in list(only_switches.values())])
    return sw_cnt
