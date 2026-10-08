"""Regression tests for the tracker's box denormalization (max_center_dist axis swap).

`get_boxes` normalizes `(y1, x1, y2, x2)` boxes y-by-height, x-by-width; the tracker
used to denormalize them y-by-WIDTH, x-by-HEIGHT. On a square frame the two agree,
so every test here uses a NON-square 2048x1536 frame (the larvae videos).
"""

from types import SimpleNamespace

import torch

from dreem.inference.post_processing import DistanceWeighting
from dreem.models.model_utils import denormalize_boxes, get_boxes

H, W = 1536, 2048  # rows, cols


def _instance(y1, x1, y2, x2):
    """The two attributes get_boxes reads: bbox (1, n_anchors, 4) and frame.img_shape."""
    return SimpleNamespace(
        bbox=torch.tensor([[[y1, x1, y2, x2]]], dtype=torch.float32),
        frame=SimpleNamespace(img_shape=(3, H, W)),
    )


def test_denormalize_inverts_get_boxes_on_a_non_square_frame():
    insts = [_instance(550, 1812, 678, 1940), _instance(10, 20, 138, 148)]
    raw = torch.cat([i.bbox for i in insts])
    torch.testing.assert_close(denormalize_boxes(get_boxes(insts), H, W), raw)


def test_denormalize_does_not_modify_its_input():
    boxes = get_boxes([_instance(550, 1812, 678, 1940)])
    before = boxes.clone()
    denormalize_boxes(boxes, H, W)
    torch.testing.assert_close(boxes, before)


def _gate(last_boxes_px, query):
    dw = DistanceWeighting(max_center_dist=165, hard=True)
    traj = torch.ones(1, 1)
    state = dw.run(
        {
            "traj_score": traj.clone(),
            "query_boxes_px": query.bbox,
            "last_boxes_px": last_boxes_px,
            "h": H,
            "w": W,
        }
    )
    return state["traj_score"]


def test_a_stationary_animal_passes_the_hard_gate():
    """The larva from the bug report: centred at x=1876, y=614, and not moving."""
    inst = _instance(550, 1812, 678, 1940)
    last_px = denormalize_boxes(get_boxes([inst]), H, W)
    torch.testing.assert_close(_gate(last_px, inst), torch.ones(1, 1))


def test_the_old_swapped_denormalization_is_what_this_test_catches():
    """Guard the guard: the pre-fix arithmetic must FAIL the stationary check."""
    inst = _instance(550, 1812, 678, 1940)
    swapped = get_boxes([inst]).clone()
    swapped[:, :, [0, 2]] *= W
    swapped[:, :, [1, 3]] *= H
    assert not torch.allclose(_gate(swapped, inst), torch.ones(1, 1))
