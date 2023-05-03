from typing import Dict, List, Tuple
import torch


def get_boxes_times(instances: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    times = torch.cat(times, dim=0).to(boxes.device)  # N

    return boxes, times

def softmax_asso(self, asso_output):
    """Applies the softmax activation function on asso_output.
    Args:
        asso_output: Same structure as before. It's a list of tensors.
        An example is shown  below. The shape is modified.
    Returns:
        asso_output: Exactly the same as before but with the softmax applied.
    # ------------------------ An example of asso_output ----------------------- #
    N_i: number of detected instances in i-th frame of window.
    N_t: number of instances in current/query frame (rightmost frame of the window).
    T: length of window.
    asso_output is of shape: (T, N_t, N_i).
    """

    # N_i: number of detected instances in i-th frame of window.
    # N_t: number of instances in current frame (rightmost frame of the window).
    # T: length of window.

    # asso_output: (T, N_t, N_i)

    asso_active = []
    for asso in asso_output:
        # asso: (N_t, N_i)

        # I'm guessing what they are doing here is giving the model a chance to pick "uncertain".
        # If the model doesn't find any high associations between 2 instances within the window,
        # this "uncertain" category will have the highest probability and the other classes/categories
        # will have lower probability.
        asso = torch.cat([asso, asso.new_zeros((asso.shape[0], 1))], dim=1).softmax(
            dim=1
        )[:, :-1]
        asso_active.append(asso)

    return asso_active  # (T, N_t, N_i)