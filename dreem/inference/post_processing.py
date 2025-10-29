"""Helper functions for post-processing association matrix pre-tracking."""

import torch
from dreem.datasets.data_utils import principal_axis_numpy


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

def filter_max_angle_diff(
    asso_output: torch.Tensor,
    max_angle_diff: float = 0,
    id_inds: torch.Tensor | None = None,
    query_boxes_px: torch.Tensor | None = None,
    nonquery_boxes_px: torch.Tensor | None = None,
) -> torch.Tensor:
    """Filter trajectory score by angle difference between objects across frames.

    Args:
        asso_output: An N_t x N association matrix
        max_angle_diff: The max angle difference between pose principal axes when considering association between two instances
        id_inds: track ids
        query_boxes_px: the raw bbox coords of the current frame instances
        nonquery_boxes_px: the raw bbox coords of the instances in the nonquery frames (context window)
    """




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
