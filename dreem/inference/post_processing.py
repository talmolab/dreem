"""Helper functions for post-processing association matrix pre-tracking."""

import logging
from typing import Any, Dict

import torch

from dreem.io.flags import FrameFlagCode
from dreem.utils.processors import ProcessingStep

logger = logging.getLogger(__name__)


class IOUWeighting(ProcessingStep):
    """Weight trajectory score by IOU between object bboxes across frames.

    This step applies IOU-based weighting to the trajectory score using either
    multiplicative or max weighting methods.

    Expected state keys:
        - traj_score: torch.Tensor (n_query, n_traj)
        - last_ious: torch.Tensor - IOU values between current and previous frames

    Modified state keys:
        - traj_score: updated with IOU-based weighting
    """

    def __init__(self, method: str | None = None):
        """Initialize IOUWeighting step.

        Args:
            method: Weighting method. One of:
                - None or "": Skip IOU weighting (no-op)
                - "mult": Multiplicative weighting: `iou*weight + traj_score`
                - "max": Max weighting: `max(traj_score, iou)`
        """
        super().__init__(name="iou_weighting")
        self.method = method

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply IOU weighting to trajectory score.

        Args:
            state: State dictionary with required keys.

        Returns:
            Modified state with updated traj_score.
        """
        if self.method is None or self.method == "":
            return state

        traj = state["traj_score"]
        last_ious = state["last_ious"]

        assert last_ious is not None, "Need `last_ious` to weight traj_score by `IOU`"

        if self.method.lower() == "mult":
            weights = torch.abs(last_ious - traj)
            weighted_iou = weights * last_ious
            weighted_iou = torch.nan_to_num(weighted_iou, 0)
            traj = traj + weighted_iou
        elif self.method.lower() == "max":
            traj = torch.max(traj, last_ious)
        else:
            raise ValueError(
                f"`method` must be one of ['mult' or 'max'] got '{self.method.lower()}'"
            )

        state["traj_score"] = traj
        return state


class DistanceWeighting(ProcessingStep):
    """Weight trajectory score by distances between objects across frames.

    This step applies a penalty to the trajectory score based on euclidean distance
    between bounding box centers.

    Expected state keys:
        - traj_score: torch.Tensor (n_query, n_traj)
        - query_boxes_px: torch.Tensor - raw bbox coords of current frame instances
        - last_boxes_px: torch.Tensor - raw bbox coords of instances in context window
        - h: int - height of the image in pixels
        - w: int - width of the image in pixels

    Modified state keys:
        - traj_score: updated with distance-based penalties
    """

    def __init__(self, max_center_dist: float, penalty_multiplier: float = 1.0):
        """Initialize DistanceWeighting step.

        Args:
            max_center_dist: The euclidean distance threshold between bboxes in pixels.
            penalty_multiplier: The multiplier for the penalty.
        """
        super().__init__(name="distance_weighting")
        self.max_center_dist = max_center_dist
        self.penalty_multiplier = penalty_multiplier

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply distance weighting to trajectory score.

        Args:
            state: State dictionary with required keys.

        Returns:
            Modified state with updated traj_score.
        """
        traj = state["traj_score"]
        q_boxes_px = state["query_boxes_px"]
        last_boxes_px = state["last_boxes_px"]
        h, w = state["h"], state["w"]

        assert (
            q_boxes_px is not None
            and last_boxes_px is not None
            and h is not None
            and w is not None
        ), (
            "Need `query_boxes_px`, `last_boxes_px`, and `h`, `w` to weight by `max_center_dist`"
        )
        diag_length = (h**2 + w**2) ** (1 / 2)  # diagonal length of the image in pixels
        max_center_dist_normalized = self.max_center_dist / diag_length
        k_ct = (q_boxes_px[:, :, :2] + q_boxes_px[:, :, 2:]) / 2
        # nonquery boxes are the most recent occurrence of each instance; could be many frames ago
        nonk_ct = (last_boxes_px[:, :, :2] + last_boxes_px[:, :, 2:]) / 2
        # pairwise euclidean distance in units of pixels
        dist = ((k_ct[:, None, :, :] - nonk_ct[None, :, :, :]) ** 2).sum(dim=-1) ** (
            1 / 2
        )  # n_k x n_nonk
        dist = dist.squeeze(-1) / diag_length  # n_k x n_nonk
        while dist.dim() < 2:
            dist = dist.unsqueeze(0)
        asso_scale = torch.abs(traj).mean(dim=1)
        penalty = torch.where(
            dist > max_center_dist_normalized, dist - max_center_dist_normalized, 0
        )  # n_k x n_nonk
        scale = asso_scale / (penalty.mean(dim=1) + 1e-8)
        scaled_penalty = -scale.unsqueeze(-1) * penalty  # n_k x n_nonk
        traj = traj + self.penalty_multiplier * scaled_penalty
        state["traj_score"] = traj
        return state


class OrientationWeighting(ProcessingStep):
    """Weight trajectory score by angle difference between objects' orientations across frames.

    This step applies a penalty based on the angular difference between principal axes
    of instances across frames.

    Expected state keys:
        - traj_score: torch.Tensor (n_query, n_traj)
        - query_principal_axes: torch.Tensor (n_query, 2) - principal axes of current frame
        - last_principal_axes: torch.Tensor (n_traj, 2) - principal axes of last frame instances

    Modified state keys:
        - traj_score: updated with angle-based penalties
    """

    def __init__(self, max_angle_diff_rad: float, penalty_multiplier: float = 1.0):
        """Initialize WeightAngleDiff step.

        Args:
            max_angle_diff_rad: Maximum angle difference threshold in radians.
            penalty_multiplier: The multiplier for the penalty.
        """
        super().__init__(name="weight_angle_diff")
        self.max_angle_diff_rad = max_angle_diff_rad
        self.penalty_multiplier = penalty_multiplier

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply angle difference weighting to trajectory score.

        Args:
            state: State dictionary with required keys.

        Returns:
            Modified state with updated traj_score.
        """
        traj = state["traj_score"]
        query_axes = state["query_principal_axes"]
        last_axes = state["last_principal_axes"]

        # Inline the weight_by_angle_diff logic
        assert query_axes is not None and last_axes is not None, (
            "Need `query_principal_axes`, and `last_principal_axes` to weight by angle difference"
        )
        dot = (query_axes[:, None, :] * last_axes[None, :, :]).sum(dim=-1)  # (q, nq)
        # cross product is scalar in 2D
        cross_z = (
            query_axes[:, None, 0] * last_axes[None, :, 1]
            - query_axes[:, None, 1] * last_axes[None, :, 0]
        )  # (q, nq)
        # product of norms cancels out in arctan so no need to calculate
        angle_diff = torch.abs(torch.atan2(cross_z, dot))
        # wrap angle diff to [0, pi/2] since there is no head/tail disambiguation in general
        angle_diff = torch.where(
            angle_diff > torch.pi / 2, torch.pi - angle_diff, angle_diff
        )
        if angle_diff.dim() == 1:
            angle_diff = angle_diff.unsqueeze(0)
        # already been modified by other post processing so use abs as this only considers scale
        weight = torch.abs(traj).mean(
            dim=1
        )  # row wise aggregation of association scores; used to weight the angle diff
        penalty = torch.where(
            angle_diff > self.max_angle_diff_rad,
            angle_diff - self.max_angle_diff_rad,
            0,
        )  # gracefully handles nans by setting diff to 0
        scale = weight / (penalty.mean(dim=1) + 1e-8)
        # normalize angle difference to [0, 1]
        scaled_penalty = -scale.unsqueeze(-1) * penalty
        traj = traj + self.penalty_multiplier * scaled_penalty
        state["traj_score"] = traj
        return state


class ConfidenceFlagging(ProcessingStep):
    """Flag frames with low confidence based on entropy of association scores.

    This step computes the entropy of the scaled trajectory scores and flags
    frames where instances have high entropy (low confidence).

    Expected state keys:
        - scaled_traj_score: torch.Tensor (n_query, n_traj) - log-softmax scaled scores
        - n_query: int - number of query instances
        - query_frame: Frame - the frame to potentially flag

    Modified state keys:
        - query_frame: may have LOW_CONFIDENCE flag added
    """

    def __init__(self, confidence_threshold: float = 0.0):
        """Initialize ConfidenceFlagging step.

        Args:
            confidence_threshold: Threshold for flagging low confidence frames.
                Set to 0 to disable flagging. Higher values are more strict
                (flag more frames).
        """
        super().__init__(name="confidence_flagging")
        self.confidence_threshold = confidence_threshold

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply confidence flagging to query frame.

        Args:
            state: State dictionary with required keys.

        Returns:
            Modified state with potentially flagged query_frame.
        """

        scaled_traj_score = state["scaled_traj_score"]
        n_query = state["n_query"]
        query_frame = state["query_frame"]

        # Compute entropy for each row
        entropy = -torch.sum(scaled_traj_score * torch.exp(scaled_traj_score), axis=1)
        norm_entropy = entropy / torch.log(torch.tensor(n_query))
        flag_threshold = 1 - self.confidence_threshold

        # Flag rows with high entropy
        flag = norm_entropy > flag_threshold

        # Flag the frame if any instances have high entropy
        if flag.any():
            query_frame.add_flag(FrameFlagCode.LOW_CONFIDENCE)

        return state
