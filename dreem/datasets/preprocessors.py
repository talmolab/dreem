"""Preprocessor classes for dataset preparation steps."""

from __future__ import annotations
from typing import Any, Dict
import logging
from dreem.utils.processors import ProcessingStep
import numpy as np
import torch
from dreem.datasets.data_utils import pairwise_iom, nms
from dreem.inference.boxes import Boxes

logger = logging.getLogger(__name__)


class RemoveExcessDetections(ProcessingStep):
    """Remove excess detections when the number exceeds max_tracks.

    This step mirrors the max_tracks pruning logic from dreem/datasets/sleap_dataset.py
    around line 487. It removes the lowest confidence instances when the total count
    exceeds the maximum allowed tracks.

    Expected state keys:
        - instances: list of dreem.io.Instance objects

    Modified state keys:
        - instances: list with excess low-confidence instances removed
        - removed_detections: int - count of removed instances (if any were removed)
    """

    def __init__(self, max_tracks: int):
        """Initialize RemoveExcessDetections step.

        Args:
            max_tracks: The maximum number of tracks/detections allowed per frame.
        """
        super().__init__(name="remove_excess_detections")
        self.max_tracks = max_tracks

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Remove excess detections based on instance scores.

        Args:
            state: State dictionary with required keys.
                - frame_ind: int - index of the current frame
                - instances: list of dreem.io.Instance objects

        Returns:
            Modified state with excess instances removed.
        """
        frame_ind = state["frame_ind"]
        instances = state["instances"]

        removed = 0
        num_to_remove = len(instances) - self.max_tracks

        for _ in range(num_to_remove):
            lowest = min(instances, key=lambda x: x.instance_score)
            if lowest.instance_score < 1:
                instances.remove(lowest)
                removed += 1
        if removed > 0:
            logger.warning(
                f"Removed {removed} lowest confidence instances from frame {frame_ind} due to excess detections. Parameter 'max_tracks' in the tracker config sets the maximum number of detections per frame."
            )

        state["instances"] = instances
        state["removed_detections"] = removed

        return state


class NonMaxSuppression(ProcessingStep):
    """Apply non-maximum suppression to a list of instances.

    This step applies non-maximum suppression to a list of instances based on their bounding box overlap.

    Expected state keys:
        - instances: list of dreem.io.Instance objects
        - max_detection_overlap: float - the maximum overlap allowed between instances
    """

    def __init__(self, max_detection_overlap: float):
        """Initialize NonMaxSuppression step.

        Args:
            max_detection_overlap: The maximum overlap allowed between instances.
        """
        super().__init__(name="non_max_suppression")
        self.max_detection_overlap = max_detection_overlap

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply non-maximum suppression to a list of instances.

        Args:
            state: State dictionary with required keys.
                - frame_ind: int - index of the current frame
                - instances: list of dreem.io.Instance objects

        Returns:
            Modified state with instances removed.
        """
        instances = state["instances"]
        frame_ind = state["frame_ind"]
        discard = set()
        bboxes = np.stack([instance.bbox.squeeze(0) for instance in instances])
        ioms = pairwise_iom(Boxes(bboxes), Boxes(bboxes))
        high_iom_pairs = nms(ioms, self.max_detection_overlap)
        for pair in high_iom_pairs:
            if pair[0] in discard or pair[1] in discard:
                continue
            # Collect the 'pose' dictionary values for both instances and stack them into tensors
            inst_1_pose_tensor = torch.stack(
                [torch.tensor(v) for v in instances[pair[0]].pose.values()]
            )
            inst_2_pose_tensor = torch.stack(
                [torch.tensor(v) for v in instances[pair[1]].pose.values()]
            )
            inst_1_keypoints = len(
                inst_1_pose_tensor[~inst_1_pose_tensor.isnan().any(dim=1)]
            )
            inst_2_keypoints = len(
                inst_2_pose_tensor[~inst_2_pose_tensor.isnan().any(dim=1)]
            )
            # break ties by keeping the first instance
            if inst_1_keypoints >= inst_2_keypoints:
                id_to_discard = pair[1]
            else:
                id_to_discard = pair[0]
            discard.add(id_to_discard)
        for id in sorted(discard, reverse=True):
            removed = instances.pop(id)
            logger.warning(
                f"Removed instance with bounding box {removed.bbox.squeeze(0)} format [ymin, xmin, ymax, xmax] from frame {frame_ind} due to high bounding box overlap with another instance"
            )
        return state
