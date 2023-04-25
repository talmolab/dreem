from typing import Dict, List, Tuple
import torch


def get_boxes_times(
    instances: List[Dict], device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the bounding boxes and frame indices from the input list of instances.

    Args:
        instances (List[Dict]): List of instance dictionaries

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors containing the
                                            bounding boxes and corresponding frame
                                            indices, respectively.
    """
    boxes, times = [], []
    _, h, w = instances[0]["img_shape"].flatten()

    for fidx, instance in enumerate(instances):
        bbox = instance["bboxes"]

        bbox[:, [0, 2]] /= w
        bbox[:, [1, 3]] /= h

        boxes.append(bbox)
        times.append(torch.full((bbox.shape[0],), fidx))

    boxes = torch.cat(boxes, dim=0)  # N x 4
    times = torch.cat(times, dim=0).to(device)  # N

    return boxes, times
