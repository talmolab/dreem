"""Helper functions for post-processing association matrix pre-tracking."""

import torch

from dreem.inference.boxes import Boxes


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


def filter_max_center_dist(
    asso_output: torch.Tensor,
    max_center_dist: float = 0,
    query_boxes_px: torch.Tensor | None = None,
    nonquery_boxes_px: torch.Tensor | None = None,
    last_inds: list[int] | None = None,
    h: int | None = None,
    w: int | None = None,
) -> torch.Tensor:
    """Filter trajectory score by distances between objects across frames.

    Args:
        asso_output: An N_t x N association matrix
        max_center_dist: The euclidean distance threshold between bboxes
        query_boxes_px: the raw bbox coords of the current frame instances
        nonquery_boxes_px: the raw bbox coords of the instances in the nonquery frames (context window)
        last_inds: the track ids of the most recent occurrence of the instances before the query frame, in the order that asso_output is indexed
        h: the height of the image in pixels
        w: the width of the image in pixels
    Returns:
        An N_t x N association matrix
    """
    assert query_boxes_px is not None and nonquery_boxes_px is not None and h is not None and w is not None, (
        "Need `query_boxes_px`, `nonquery_boxes_px`, and `h`, `w` to filter by `max_center_dist`"
    )
    diag_length = (h**2 + w**2)**(1/2) # diagonal length of the image in pixels
    max_center_dist_normalized = max_center_dist/diag_length
    k_ct = (query_boxes_px[:, :, :2] + query_boxes_px[:, :, 2:]) / 2
    # nonquery boxes are the most recent occurrence of each instance; could be many frames ago
    nonk_ct = (nonquery_boxes_px[:, :, :2] + nonquery_boxes_px[:, :, 2:]) / 2
    # pairwise euclidean distance in units of pixels
    dist = ((k_ct[:, None, :, :] - nonk_ct[None, :, :, :]) ** 2).sum(dim=-1) ** (
        1 / 2
    )  # n_k x n_nonk
    dist = dist.squeeze()/diag_length # n_k x n_nonk
    dist = dist[:,last_inds]
    asso_scale = asso_output.mean(dim=1).cpu()
    penalty = torch.where(dist > max_center_dist_normalized, dist - max_center_dist_normalized, 0).cpu() # n_k x n_nonk
    scale = asso_scale.cpu() / (penalty.mean(dim=1).cpu() + 1e-8)
    scaled_penalty = -scale.unsqueeze(-1).cpu() * penalty   # n_k x n_nonk
    asso_out = asso_output.cpu() + scaled_penalty
    return asso_out