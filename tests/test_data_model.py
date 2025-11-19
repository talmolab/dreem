"""Tests for Instance, Frame, and AssociationMatrix Objects"""

import numpy as np
import pandas as pd
import pytest
import torch

from dreem.io import AssociationMatrix, Frame, Instance, Track


def test_instance():
    """Test Instance object logic."""
    gt_track_id = 0
    pred_track_id = 0
    bbox = torch.randn((1, 1, 4))
    crop = torch.randn((1, 3, 128, 128))
    features = torch.randn((1, 64))

    instance = Instance(
        gt_track_id=gt_track_id,
        pred_track_id=pred_track_id,
        bbox=bbox,
        crop=crop,
        features=features,
    )

    assert instance.has_gt_track_id()
    assert instance.gt_track_id.item() == gt_track_id
    assert instance.has_pred_track_id()
    assert instance.pred_track_id.item() == pred_track_id
    assert instance.has_bbox()
    assert torch.equal(instance.bbox, bbox)
    assert instance.has_features()
    assert torch.equal(instance.features, features)

    instance.gt_track_id = 1
    instance.pred_track_id = 1
    instance.bbox = torch.randn((1, 4))
    instance.crop = torch.randn((1, 3, 128, 128))
    instance.features = torch.randn((1, 64))

    assert instance.has_gt_track_id()
    assert instance.gt_track_id.item() != gt_track_id
    assert instance.has_pred_track_id()
    assert instance.pred_track_id.item() != pred_track_id
    assert instance.has_bbox()
    assert not torch.equal(instance.bbox, bbox)
    assert instance.has_features()
    assert not torch.equal(instance.features, features)

    instance.gt_track_id = None
    instance.pred_track_id = -1
    instance.bbox = None
    instance.crop = None
    instance.features = None

    assert not instance.has_gt_track_id()
    assert instance.gt_track_id.shape[0] == 0
    assert not instance.has_pred_track_id()
    assert instance.pred_track_id.item() != pred_track_id
    assert not instance.has_bbox()
    assert not torch.equal(instance.bbox, bbox)
    assert not instance.has_features()
    assert not torch.equal(instance.features, features)


def test_frame():
    n_detected = 2
    n_traj = 3
    video_id = 0
    frame_id = 0
    img_shape = torch.Size([3, 1024, 1024])
    traj_score = torch.randn(n_detected, n_traj)
    matches = ([0, 1], [0, 1])

    instances = []
    for i in range(n_detected):
        instances.append(
            Instance(
                gt_track_id=i,
                pred_track_id=i,
                bbox=torch.randn(1, 4),
                crop=torch.randn(1, 3, 64, 64),
                features=torch.randn(1, 64),
            )
        )
    frame = Frame(
        video_id=video_id, frame_id=frame_id, img_shape=img_shape, instances=instances
    )

    assert frame.video_id.item() == video_id
    assert frame.frame_id.item() == frame_id
    assert frame.img_shape == img_shape
    assert frame.num_detected == n_detected
    assert frame.has_instances()
    assert len(frame.instances) == n_detected
    assert frame.has_gt_track_ids()
    assert len(frame.get_gt_track_ids()) == n_detected
    assert frame.has_pred_track_ids()
    assert len(frame.get_pred_track_ids()) == n_detected
    assert not frame.has_matches()
    assert not frame.has_asso_output()
    assert not frame.has_traj_score()

    asso_output = torch.randn(len(instances), len(instances))
    frame.asso_output = AssociationMatrix(asso_output, instances, instances)
    frame.add_traj_score("initial", traj_score)
    frame.matches = matches

    assert frame.has_matches()
    assert frame.matches == matches
    assert frame.has_asso_output()
    assert torch.equal(frame.asso_output.matrix, asso_output)
    assert frame.has_traj_score()
    assert torch.equal(frame.get_traj_score("initial"), traj_score)

    frame.instances = []

    assert frame.video_id.item() == video_id
    assert frame.num_detected == 0
    assert not frame.has_instances()
    assert len(frame.instances) == 0
    assert not frame.has_gt_track_ids()
    assert not len(frame.get_gt_track_ids())
    assert not frame.has_pred_track_ids()
    assert len(frame.get_pred_track_ids()) == 0
    assert frame.has_matches()
    assert frame.has_asso_output()
    assert frame.has_traj_score()


def test_association_matrix():
    n_traj = 2
    total_instances = 32
    n_query = 2

    instances = [
        Instance(gt_track_id=i % n_traj, pred_track_id=i % n_traj)
        for i in range(total_instances)
    ]

    query_instances = instances[-n_query:]
    asso_tensor = np.random.rand(total_instances, total_instances)
    query_tensor = np.random.rand(n_query, total_instances)

    with pytest.raises(ValueError):
        _ = AssociationMatrix(asso_tensor, instances, query_instances)
        _ = AssociationMatrix(asso_tensor, query_instances, query_instances)

    asso_matrix = AssociationMatrix(asso_tensor, instances, instances)

    assert isinstance(asso_matrix.numpy(), np.ndarray)

    asso_lookup = asso_matrix[instances[0], instances[0]]
    assert asso_lookup.item() == asso_tensor[0, 0].item()

    inds = (-1, -1)
    asso_lookup = asso_matrix[inds]
    assert asso_lookup.item() == asso_tensor[-1, -1].item()

    inds = (instances[:2], instances[:-2])
    asso_lookup = asso_matrix[inds]
    assert np.equal(asso_lookup, asso_tensor[:2, :-2]).all()

    inds = ([2, 3], [2, 3])
    asso_lookup = asso_matrix[inds]
    assert np.equal(asso_lookup, asso_tensor[np.array(inds[0])[:, None], inds[1]]).all()

    asso_lookup = asso_matrix[instances[:2], None]
    assert np.equal(asso_lookup, asso_tensor[:2, :]).all()

    with pytest.raises(ValueError):
        _ = AssociationMatrix(query_tensor, instances, instances)
        _ = AssociationMatrix(query_tensor, query_instances, instances)
        _ = AssociationMatrix(query_tensor, query_instances, query_instances)

    query_matrix = AssociationMatrix(query_tensor, instances, query_instances)

    with pytest.raises(ValueError):
        _ = query_matrix[instances[0], instances[0]]

    query_lookup = query_matrix[query_instances[0], instances[0]]
    assert query_lookup.item() == query_tensor[0, 0].item()

    traj_score = pd.concat(
        [
            query_matrix.to_dataframe(row_labels="inst").drop(1, axis=1).sum(1),
            query_matrix.to_dataframe(row_labels="inst").drop(0, axis=1).sum(1),
        ],
        axis=1,
    )
    assert (query_matrix.reduce() == traj_score).all().all()


def test_track():
    instances = [Instance(gt_track_id=0, pred_track_id=0) for i in range(32)]

    track = Track(0, instances=instances)

    assert track.track_id == 0
    assert len(track) == len(instances)

    instance = track[1]

    assert instance is instances[1]
