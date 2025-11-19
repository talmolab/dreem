import logging
from typing import Callable
import torch
from dreem.datasets.data_utils import (
    get_pose_principal_axis,
    gather_pose_array,
    is_pose_centroid_only,
)
from scipy import ndimage
import numpy as np

logger = logging.getLogger(__name__)

# Registry of principal axis computation methods
# Each entry: (can_compute_func, compute_func)
_PRINCIPAL_AXIS_METHODS = []


def _can_compute_with_prompts(instance, **kwargs) -> bool:
    """Check if orientation prompt method can be used.

    Conditions:
    - Skeleton must NOT be only centroid (must have more than 1 keypoint)
    - front_nodes and back_nodes must be provided and valid
    - Both prompt nodes must exist and be valid (non-NaN)
    """
    front_nodes = kwargs.get("front_nodes")
    back_nodes = kwargs.get("back_nodes")
    # Must have more than just centroid
    if is_pose_centroid_only(instance):
        logger.debug(
            f"Pose is only centroid. Cannot compute principal axis with orientation prompts."
        )
        return False
    # Must have valid orientation prompt
    if (
        front_nodes is None
        or back_nodes is None
        or len(front_nodes) < 1
        or len(back_nodes) < 1
    ):
        logger.debug(
            f"Orientation prompt is not provided or is not valid. Cannot compute principal axis with orientation prompts."
        )
        return False
    # Both nodes must exist
    front = None
    back = None
    for front_node in front_nodes:
        if front_node in instance:
            front = instance.get(front_node)
            break
    for back_node in back_nodes:
        if back_node in instance:
            back = instance.get(back_node)
            break
    # Both nodes must be valid (non-NaN)
    front_tensor = torch.tensor(front)
    back_tensor = torch.tensor(back)
    if (
        front is None
        or back is None
        or torch.isnan(front_tensor).any()
        or torch.isnan(back_tensor).any()
    ):
        logger.debug(
            f"Orientation prompt nodes not visible. Cannot compute principal axis with orientation prompts."
        )
        return False
    return not (torch.isnan(front_tensor).any() or torch.isnan(back_tensor).any())


def _compute_with_prompts(instance, **kwargs) -> torch.Tensor:
    """Compute principal axis using orientation prompts."""
    front_nodes = kwargs.get("front_nodes")
    back_nodes = kwargs.get("back_nodes")
    # we've already checked that front_nodes and back_nodes are valid
    for node in front_nodes:
        if node in instance:
            front = instance.get(node)
            break
    for node in back_nodes:
        if node in instance:
            back = instance.get(node)
            break
    vec = torch.tensor(front - back)
    norm = torch.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def _can_compute_with_pca(instance, **kwargs) -> bool:
    """Check if PCA method can be used (always available as fallback)."""
    result = not is_pose_centroid_only(instance)
    if not result:
        logger.debug(
            f"PCA method cannot be used. Cannot compute principal axis with PCA."
        )
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
        logger.debug(
            f"Cannot compute principal axis with image gradient. Pose must be only centroid, and crop must be available."
        )
    return result


def _compute_with_img_grad(instance, **kwargs) -> torch.Tensor:
    """Compute principal axis using image gradient."""
    crop = kwargs.get("crop")
    crop = crop.squeeze().permute(1, 2, 0).numpy()
    Ix = ndimage.sobel(crop, axis=1)
    Iy = ndimage.sobel(crop, axis=0)
    Ixx = (Ix * Ix).mean()
    Iyy = (Iy * Iy).mean()
    Ixy = (Ix * Iy).mean()
    covar_matrix = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eig(covar_matrix)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return torch.as_tensor(
        eigvecs[:, 1], dtype=torch.float32
    )  # smaller eigval is tangent to instance


# Register methods in priority order
_PRINCIPAL_AXIS_METHODS.extend(
    [
        ("orientation_prompt", _can_compute_with_prompts, _compute_with_prompts),
        ("img_grad", _can_compute_with_img_grad, _compute_with_img_grad),
        ("pca", _can_compute_with_pca, _compute_with_pca),
    ]
)


def get_principal_axis_with_fallback(
    instance,
    front_nodes: list[str] | None,
    back_nodes: list[str] | None,
    crop: torch.Tensor | None,
    logger,
    frame_id,
) -> tuple[torch.Tensor, bool]:
    """Compute principal axis with automatic fallback.

    Tries registered methods in priority order until one succeeds.

    Args:
        instance: Instance dict with pose keypoints
        front_nodes: List of front nodes
        back_nodes: List of back nodes
        crop: Optional crop tensor
        logger: Logger instance
        frame_id: Frame ID for logging

    Returns:
        Tuple of (principal_axis_vector, success) where success indicates
        if the principal axis was successfully computed.
    """
    kwargs = {"front_nodes": front_nodes, "back_nodes": back_nodes, "crop": crop}

    for method_name, can_compute, compute in _PRINCIPAL_AXIS_METHODS:
        if can_compute(instance, **kwargs):
            result = compute(instance, **kwargs)
            if result is not None:
                return result, True
    return torch.full((2,), torch.nan, dtype=torch.float32), False


def register_principal_axis_method(
    name: str, can_compute: Callable, compute: Callable, priority: int | None = None
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
