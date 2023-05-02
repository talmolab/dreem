import torch


def weight_decay_time(
    asso_output: torch.Tensor,
    decay_time: float = 0,
    reid_features: torch.Tensor = None,
    T: int = None,
    k: int = None,
) -> torch.Tensor:
    """
    Weight association matrix by number of frames the ith object is from the jth object in the association matrix
    Returns: The N_t x N association matrix weighted by decay time
    Args:
        asso_output: the association matrix to be reweighted
        decay_time: the scale to weight the asso_output by
        reid_features: The n x d matrix of feature vectors for each object
        T: The length of the window
        k: an integer for the query frame within the window of instances
    """
    if decay_time is not None and decay_time > 0:
        assert (
            reid_features is not None and T is not None and k is not None
        ), "Need reid_features to weight traj_score by `decay_time`!"
        N_t = asso_output.shape[0]
        dts = torch.cat(
            [
                x.new_full((N_t,), T - t - 2)
                for t, x in enumerate(reid_features)
                if t != k
            ],
            dim=0,
        ).cpu()  # Np
        # asso_output = asso_output.to(self.device) * (self.decay_time ** dts[None, :])
        asso_output = asso_output * (decay_time ** dts[:, None])
    return asso_output


def weight_iou(
    asso_output: torch.Tensor, method: str = None, last_ious: torch.Tensor = None
):
    """
    Weight the association matrix by the IOU between object bboxes across frames
    Returns: An N_t x N association matrix weighted by the IOU
    Args:
        asso_output: An N_t x N association matrix
        method: string indicating whether to use a max weighting or multiplicative weighting
                Max weighting: take `max(traj_score, iou)`
                multiplicative weighting: `iou*weight + traj_score`
        last_ious: torch Tensor containing the ious between current and previous frames
    """
    if method is not None and method != "":
        assert last_ious is not None, "Need `last_ious` to weight traj_score by `IOU`"
        if method.lower() == "mult":
            weights = torch.abs(last_ious - asso_output)
            asso_output = asso_output + (weights * last_ious)
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
    k_boxes: torch.Tensor = None,
    nonk_boxes: torch.Tensor = None,
    id_inds: torch.Tensor = None,
):
    """
    Filter trajectory score by distances between objects across frames
    Returns: An N_t x N association matrix
    Args:
        asso_output: An N_t x N association matrix
        max_center_dist: The euclidean distance threshold between bboxes
        k_boxes: The bounding boxes in the current frame
        nonk_boxes: the boxes not in the current frame
        id_inds: track ids
    """
    if max_center_dist is not None and max_center_dist > 0:
        assert (
            k_boxes is not None and nonk_boxes is not None and id_inds is not None
        ), "Need `k_boxes`, `nonk_boxes`, and `id_ind` to filter by `max_center_dist`"
        k_ct = (k_boxes[:, :2] + k_boxes[:, 2:]) / 2
        k_s = ((k_boxes[:, 2:] - k_boxes[:, :2]) ** 2).sum(dim=1)  # n_k
        nonk_ct = (nonk_boxes[:, :2] + nonk_boxes[:, 2:]) / 2
        dist = ((k_ct[:, None] - nonk_ct[None, :]) ** 2).sum(dim=2)  # n_k x Np
        norm_dist = dist / (k_s[:, None] + 1e-8)  # n_k x Np
        # id_inds # Np x M
        valid = norm_dist < max_center_dist  # n_k x Np
        valid_assn = (
            torch.mm(valid.float(), id_inds).clamp_(max=1.0).long().bool()
        )  # n_k x M
        asso_output[~valid_assn] = 0  # n_k x M
    return asso_output
