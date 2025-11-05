"""Helper functions for post-processing association matrix pre-tracking."""

import logging
from typing import Callable
import torch
from dreem.datasets.data_utils import get_pose_principal_axis, gather_pose_array, is_pose_centroid_only
from scipy import ndimage
import numpy as np
logger = logging.getLogger(__name__)


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

# Registry of principal axis computation methods
# Each entry: (can_compute_func, compute_func)
_PRINCIPAL_AXIS_METHODS = []


def _can_compute_with_prompts(instance, **kwargs) -> bool:
    """Check if orientation prompt method can be used.

    Conditions:
    - Skeleton must NOT be only centroid (must have more than 1 keypoint)
    - orientation_prompt must be provided and valid
    - Both prompt nodes must exist and be valid (non-NaN)
    """
    orientation_prompt = kwargs.get("orientation_prompt")
    # Must have more than just centroid
    if is_pose_centroid_only(instance):
        logger.debug(f"Pose is only centroid. Cannot compute principal axis with orientation prompts.")
        return False
    # Must have valid orientation prompt
    if orientation_prompt is None or len(orientation_prompt) != 2:
        logger.debug(f"Orientation prompt is not provided or is not valid. Cannot compute principal axis with orientation prompts.")
        return False
    # Both nodes must exist
    head = instance.get(orientation_prompt[0])
    tip = instance.get(orientation_prompt[1])
    # Both nodes must be valid (non-NaN)
    head_tensor = torch.tensor(head)
    tip_tensor = torch.tensor(tip)
    if head is None or tip is None or torch.isnan(head_tensor).any() or torch.isnan(tip_tensor).any():
        logger.debug(f"Orientation prompt nodes not visible. Cannot compute principal axis with orientation prompts.")
        return False
    return not (torch.isnan(head_tensor).any() or torch.isnan(tip_tensor).any())


def _compute_with_prompts(instance, **kwargs) -> torch.Tensor:
    """Compute principal axis using orientation prompts."""
    orientation_prompt = kwargs.get("orientation_prompt")
    head = instance[orientation_prompt[0]]
    tip = instance[orientation_prompt[1]]
    vec = torch.tensor(head - tip)
    norm = torch.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def _can_compute_with_pca(instance, **kwargs) -> bool:
    """Check if PCA method can be used (always available as fallback)."""
    result = not is_pose_centroid_only(instance)
    if not result:
        logger.debug(f"PCA method cannot be used. Cannot compute principal axis with PCA.")
    return result


def _compute_with_pca(instance, **kwargs) -> torch.Tensor:
    """Compute principal axis using PCA."""
    instance_pose_arr = gather_pose_array([instance])
    return get_pose_principal_axis(instance_pose_arr)[0]


def _can_compute_with_img_grad(instance, **kwargs) -> bool:
    """Check if image gradient method can be used.

    Conditions:
    - Skeleton must be ONLY centroid (exactly 1 key, which is "centroid")
    - Crop must be available
    """
    crop = kwargs.get("crop")
    result = is_pose_centroid_only(instance) and crop is not None and len(crop) > 0
    if not result:
        logger.debug(f"Cannot compute principal axis with image gradient. Pose must be only centroid, and crop must be available.")
    return result


def _compute_with_img_grad(instance, **kwargs) -> torch.Tensor:
    """Compute principal axis using image gradient."""
    crop = kwargs.get("crop")
    crop = crop.squeeze().permute(1,2,0).numpy()
    Ix = ndimage.sobel(crop, axis=1)
    Iy = ndimage.sobel(crop, axis=0)
    Ixx = (Ix*Ix).mean()
    Iyy = (Iy*Iy).mean()
    Ixy = (Ix*Iy).mean()
    covar_matrix = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eig(covar_matrix)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return torch.as_tensor(eigvecs[:, 1], dtype=torch.float32) # smaller eigval is tangent to instance

# Register methods in priority order
_PRINCIPAL_AXIS_METHODS.extend([
    ("orientation_prompt", _can_compute_with_prompts, _compute_with_prompts),
    ("img_grad", _can_compute_with_img_grad, _compute_with_img_grad),
    ("pca", _can_compute_with_pca, _compute_with_pca),
])


def get_principal_axis_with_fallback(
    instance, orientation_prompt: list[str] | None, crop: torch.Tensor | None, logger, frame_id
) -> tuple[torch.Tensor, bool]:
    """Compute principal axis with automatic fallback.

    Tries registered methods in priority order until one succeeds.

    Args:
        instance: Instance dict with pose keypoints
        orientation_prompt: Optional list of two node names for orientation-based method
        crop: Optional crop tensor
        logger: Logger instance
        frame_id: Frame ID for logging

    Returns:
        Tuple of (principal_axis_vector, success) where success indicates
        if the principal axis was successfully computed.
    """
    kwargs = {"orientation_prompt": orientation_prompt,
              "crop": crop}

    for method_name, can_compute, compute in _PRINCIPAL_AXIS_METHODS:
        if can_compute(instance, **kwargs):
            result = compute(instance, **kwargs)
            if result is not None:
                return result, True
    return torch.full((2,), torch.nan, dtype=torch.float32), False


def register_principal_axis_method(
    name: str,
    can_compute: Callable,
    compute: Callable,
    priority: int | None = None
) -> None:
    """Register a new principal axis computation method.

    Args:
        can_compute: Function that checks if method can be used for an instance.
                    Signature: can_compute(instance, **kwargs) -> bool
        compute: Function that computes principal axis.
                Signature: compute(instance, **kwargs) -> torch.Tensor
        name: Name of the method
        priority: Optional priority index. If None, appends to end. Lower index = higher priority.
    """
    entry = (name, can_compute, compute)
    if priority is None:
        _PRINCIPAL_AXIS_METHODS.append(entry)
    else:
        _PRINCIPAL_AXIS_METHODS.insert(priority, entry)

def weight_by_angle_diff(
    asso_output: torch.Tensor,
    max_angle_diff: float = 0,
    last_inds: list[int] | None = None,
    query_principal_axes: torch.Tensor | None = None,
    last_principal_axes: torch.Tensor | None = None,
    fallback: bool = False,
) -> torch.Tensor:
    """Weight trajectory score by angle difference between objects across frames.

    Args:
        asso_output: An N_t x N association matrix
        max_angle_diff: Maximum angle difference threshold in radians
        last_pred_ids: the track ids of the most recent occurrence of the instances before the query frame, in the order that asso_output is indexed
        query_principal_axes: the principal axes of the current frame instances. Shape: (q, 2)
        last_principal_axes: the principal axes of the instances in the last frame. Shape: (nq, 2)
        fallback: Whether a fallback method was used (affects angle wrapping to [0, pi/2])
    """
    assert query_principal_axes is not None and last_principal_axes is not None, (
        "Need `query_principal_axes`, and `last_principal_axes` to weight by angle difference"
    )
    dot = (query_principal_axes[:,None,:] * last_principal_axes[None,:,:]).sum(dim=-1) # (q, nq)
    # cross product is scalar in 2D
    cross_z = query_principal_axes[:,None, 0] * last_principal_axes[None,:, 1] - query_principal_axes[:,None, 1] * last_principal_axes[None,:, 0] # (q, nq)
    # product of norms cancels out in arctan so no need to calculate
    angle_diff = torch.abs(torch.atan2(cross_z, dot))
    # wrap angle diff to [0, pi/2] since there is no head/tail disambiguation in general
    angle_diff = torch.where(angle_diff > torch.pi / 2, torch.pi - angle_diff, angle_diff)
    # reindex the columns of the angle_diff matrix based on the index of last pred ids
    angle_diff = angle_diff[:,last_inds]
    weight = asso_output.mean(dim=1)  # row wise aggregation of association scores; used to weight the angle diff
    penalty = torch.where(angle_diff > max_angle_diff, (angle_diff - max_angle_diff)/(torch.pi / 2), 0)
    scale = weight / (penalty.mean(dim=1) + 1e-8)
    # normalize angle difference to [0, 1]
    scaled_penalty = -scale.unsqueeze(-1) * penalty
    asso_out = asso_output + scaled_penalty
    return asso_out


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
    asso_scale = asso_output.mean(dim=1)
    penalty = torch.where(dist > max_center_dist_normalized, dist - max_center_dist_normalized, 0) # n_k x n_nonk
    scale = asso_scale / (penalty.mean(dim=1) + 1e-8)
    scaled_penalty = -scale.unsqueeze(-1) * penalty   # n_k x n_nonk
    asso_out = asso_output + scaled_penalty
    return asso_out
