"""Helper functions for post-processing association matrix pre-tracking."""

import torch

from dreem.inference.boxes import Boxes


def _pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute the intersection area between __all__ N x M pairs of boxes.

    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1: First set of boxes (Boxes object containing N boxes).
        boxes2: Second set of boxes (Boxes object containing M boxes).

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, :, 2:], boxes2[:, :, 2:]) - torch.max(
        boxes1[:, None, :, :2], boxes2[:, :, :2]
    )  # [N,M,n_anchors,     2]
    width_height.clamp_(min=0)  # [N,M, n_anchors, 2]

    intersection = width_height.prod(dim=3)  # [N,M, n_anchors]

    return intersection


def _pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute intersection over union between all N x M pairs of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1: First set of boxes (Boxes object containing N boxes).
        boxes2: Second set of boxes (Boxes object containing M boxes).

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]

    inter = _pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter >= 0,
        inter / (area1[:, None, :] + area2 - inter),
        torch.nan,
    )
    return iou.nanmean(dim=-1)


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
    last_inds: torch.Tensor | None = None,
    query_boxes_px: torch.Tensor | None = None,
    nonquery_boxes_px: torch.Tensor | None = None,
    instances_per_frame: torch.Tensor | None = None,
    true_frame_ids: list[int] | None = None,
) -> torch.Tensor:
    """Filter trajectory score by distances between objects across frames.

    Args:
        asso_output: An N_t x N association matrix
        max_center_dist: The euclidean distance threshold between bboxes
        last_inds: the indices of the last instances in the nonquery frames
        query_boxes_px: the raw bbox coords of the current frame instances
        nonquery_boxes_px: the raw bbox coords of the instances in the nonquery frames (context window)
        instances_per_frame: the number of instances per frame, excluding the query frame
        true_frame_ids: the true frame ids of the nonquery and query frames (last index is the current frame)
    Returns:
        An N_t x N association matrix
    """
    if max_center_dist is not None and max_center_dist > 0:
        assert query_boxes_px is not None and nonquery_boxes_px is not None, (
            "Need `query_boxes_px`, and `nonquery_boxes_px` to filter by `max_center_dist`"
        )
        k_ct = (query_boxes_px[:, :, :2] + query_boxes_px[:, :, 2:]) / 2
        # nonk boxes are only from previous frame rather than entire window
        nonk_ct = (nonquery_boxes_px[:, :, :2] + nonquery_boxes_px[:, :, 2:]) / 2

        # pairwise euclidean distance in units of pixels
        dist = ((k_ct[:, None, :, :] - nonk_ct[None, :, :, :]) ** 2).sum(dim=-1) ** (
            1 / 2
        )  # n_k x n_nonk
        max_center_dist_adjusted = torch.ones(nonk_ct.shape[0])
        # find out the number of frames elapsed since each instance was seen
        # map last_inds to frames
        n_nonquery = sum(instances_per_frame)
        cumulative = torch.cumsum(instances_per_frame, dim=0)
        # bin ids are the indices (into the true_frame_ids list) of the last known position of the instances
        bin_ids = torch.searchsorted(cumulative, last_inds + 1, right=False)
        curr_frame_id = true_frame_ids[-1].item()
        # scale max_center_dist by num of frames i.e. grow the possible region that the instance can be in
        max_center_dist_adjusted = max_center_dist * (torch.max(torch.tensor(1), curr_frame_id - true_frame_ids[:-1][bin_ids]))
        valid = dist.squeeze() < max_center_dist_adjusted  # n_k x n_nonk

        # handle case where id_inds and valid is a single value
        # handle this better
        if valid.ndim == 0:
            valid = valid.unsqueeze(0)
        if valid.ndim == 1:
            if last_inds.shape[0] == 1:
                valid_mult = valid.unsqueeze(-1)
            else:
                valid_mult = valid.unsqueeze(0)
        else:
            valid_mult = valid
        # valid_assn = (
        #     torch.mm(valid_mult, id_inds.to(valid.device)).clamp_(max=1.0).long().bool()
        # )  # n_k x M
        asso_output_filtered = asso_output.clone()
        asso_output_filtered[~valid_mult] = 0  # n_k x M
        return asso_output_filtered, valid_mult
    else:
        return asso_output, None
