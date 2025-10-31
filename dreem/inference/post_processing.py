"""Helper functions for post-processing association matrix pre-tracking."""

import torch
from dreem.datasets.data_utils import get_pose_principal_axis, gather_pose_array


def weight_iou(
    asso_output: torch.Tensor, method: str | None = None, last_ious: torch.Tensor = None
) -> torch.Tensor:
    """Weight the association matrix by the IOU between object bboxes across frames.

    Args:
        asso_output: An N_t x N association matrix
        method: string indicating whether to use a max weighting or multiplicative weighting
                Max weighting: take `max(traj_score, iou)`
                multiplicative weighting: `iou*weight + traj_score`
        last_ious: torch Tensor containing the ious between current and previous frames

    Returns:
        An N_t x N association matrix weighted by the IOU
    """
    if method is not None and method != "":
        assert last_ious is not None, "Need `last_ious` to weight traj_score by `IOU`"
        if method.lower() == "mult":
            weights = torch.abs(last_ious - asso_output)
            weighted_iou = weights * last_ious
            weighted_iou = torch.nan_to_num(weighted_iou, 0)
            asso_output = asso_output + weighted_iou
        elif method.lower() == "max":
            asso_output = torch.max(asso_output, last_ious)
        else:
            raise ValueError(
                f"`method` must be one of ['mult' or 'max'] got '{method.lower()}'"
            )
    return asso_output

def _compute_principal_axis_with_pca(instance):
    """PCA fallback method."""
    instance_pose_arr = gather_pose_array([instance])
    return get_pose_principal_axis(instance_pose_arr)[0]

def _compute_principal_axis_with_prompts(instance, orientation_prompt):
    """Primary method using orientation prompts."""
    head = None
    tip = None
    for node_name, node_coords in instance.items():
        if node_name == orientation_prompt[0]:
            head = node_coords
        elif node_name == orientation_prompt[1]:
            tip = node_coords
    if head is not None and tip is not None and ~torch.isnan(torch.tensor(head)).any() and ~torch.isnan(torch.tensor(tip)).any():
        vec = torch.tensor(head - tip)
        # Normalize to unit vector to compare to PCA derived principal axis
        norm = torch.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec
    return None

def get_principal_axis_with_fallback(instance, orientation_prompt: list[str] | None, logger, frame_id):
    """Compute principal axis with automatic fallback.

    Tries the preferred method first, falls back to PCA if it fails.
    """
    if orientation_prompt is not None:
        assert len(orientation_prompt) == 2, "Orientation prompt must be a list of only two skeleton node names, with the 'head' node first"
        result = _compute_principal_axis_with_prompts(instance, orientation_prompt)
        if result is not None:
            return result, False
    return _compute_principal_axis_with_pca(instance), True

def weight_by_angle_diff(
    asso_output: torch.Tensor,
    max_angle_diff: float = 0,
    last_pred_ids: list[int] | None = None,
    query_principal_axes: torch.Tensor | None = None,
    last_principal_axes: torch.Tensor | None = None,
    fallback: bool = False,
) -> torch.Tensor:
    """Weight trajectory score by angle difference between objects across frames.

    Args:
        asso_output: An N_t x N association matrix
        last_pred_ids: the track ids of the most recent occurrence of the instances before the query frame, in the order that asso_output is indexed
        query_principal_axes: the principal axes of the current frame instances. Shape: (q, 2)
        last_principal_axes: the principal axes of the instances in the last frame. Shape: (nq, 2)
    """
    assert query_principal_axes is not None and last_principal_axes is not None, (
        "Need `query_principal_axes`, and `last_principal_axes` to weight by angle difference"
    )
    dot = (query_principal_axes[:,None,:] * last_principal_axes[None,:,:]).sum(dim=-1) # (q, nq)
    # cross product is scalar in 2D
    cross_z = query_principal_axes[:,None, 0] * last_principal_axes[None,:, 1] - query_principal_axes[:,None, 1] * last_principal_axes[None,:, 0] # (q, nq)
    # product of norms cancels out in arctan so no need to calculate
    angle_diff = torch.abs(torch.atan2(cross_z, dot))
    # wrap angle diff to [0, pi/2] for PCA method since there is no head/tail disambiguation
    if fallback:
        angle_diff = torch.where(angle_diff > torch.pi / 2, torch.pi - angle_diff, angle_diff)
    # reindex the columns of the angle_diff matrix baesd on the index of last pred ids
    angle_diff = angle_diff[:,last_pred_ids]
    weight = asso_output.mean(dim=1) # row wise aggregation of association scores; used to weight the angle diff 
    penalty = -weight * torch.where(angle_diff > max_angle_diff, angle_diff - max_angle_diff, 0)
    asso_out = asso_output + penalty
    return asso_out


def filter_max_center_dist(
    asso_output: torch.Tensor,
    max_center_dist: float = 0,
    id_inds: torch.Tensor | None = None,
    query_boxes_px: torch.Tensor | None = None,
    nonquery_boxes_px: torch.Tensor | None = None,
) -> torch.Tensor:
    """Filter trajectory score by distances between objects across frames.

    Args:
        asso_output: An N_t x N association matrix
        max_center_dist: The euclidean distance threshold between bboxes
        id_inds: track ids
        query_boxes_px: the raw bbox coords of the current frame instances
        nonquery_boxes_px: the raw bbox coords of the instances in the nonquery frames (context window)

    Returns:
        An N_t x N association matrix
    """
    if max_center_dist is not None and max_center_dist > 0:
        assert query_boxes_px is not None and nonquery_boxes_px is not None, (
            "Need `query_boxes_px`, and `nonquery_boxes_px` to filter by `max_center_dist`"
        )

        k_ct = (query_boxes_px[:, :, :2] + query_boxes_px[:, :, 2:]) / 2
        # k_s = ((curr_frame_boxes[:, :, 2:] - curr_frame_boxes[:, :, :2]) ** 2).sum(dim=2)  # n_k
        # nonk boxes are only from previous frame rather than entire window
        nonk_ct = (nonquery_boxes_px[:, :, :2] + nonquery_boxes_px[:, :, 2:]) / 2

        # pairwise euclidean distance in units of pixels
        dist = ((k_ct[:, None, :, :] - nonk_ct[None, :, :, :]) ** 2).sum(dim=-1) ** (
            1 / 2
        )  # n_k x n_nonk
        # norm_dist = dist / (k_s[:, None, :] + 1e-8)

        valid = dist.squeeze() < max_center_dist  # n_k x n_nonk
        # handle case where id_inds and valid is a single value
        # handle this better
        if valid.ndim == 0:
            valid = valid.unsqueeze(0)
        if valid.ndim == 1:
            if id_inds.shape[0] == 1:
                valid_mult = valid.float().unsqueeze(-1)
            else:
                valid_mult = valid.float().unsqueeze(0)
        else:
            valid_mult = valid.float()

        valid_assn = (
            torch.mm(valid_mult, id_inds.to(valid.device)).clamp_(max=1.0).long().bool()
        )  # n_k x M
        asso_output_filtered = asso_output.clone()
        asso_output_filtered[~valid_assn] = 0  # n_k x M
        return asso_output_filtered
    else:
        return asso_output
