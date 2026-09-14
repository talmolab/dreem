"""Test the sio.Identity / sio.Track association-label carrier.

sleap-io >=0.9.0 separates `Track` (a within-video trajectory) from `Identity`
(one physical animal, recognizable across videos). `SleapDataset` derives
`gt_track_id` from whichever the file carries -- see
`dreem.datasets.sleap_dataset.resolve_identity_carrier`.
"""

import numpy as np
import pytest
import sleap_io as sio

from dreem.datasets.sleap_dataset import identity_index, resolve_identity_carrier


def _labels(specs):
    """Build a one-frame `Labels` from `(identity_name | None, track_name | None)`."""
    skeleton = sio.Skeleton(["a"])
    instances = [
        sio.Instance.from_numpy(
            np.array([[float(i), 0.0]]),
            skeleton=skeleton,
            track=sio.Track(track) if track else None,
            identity=sio.Identity(identity) if identity else None,
        )
        for i, (identity, track) in enumerate(specs)
    ]
    labels = sio.Labels(
        [
            sio.LabeledFrame(
                video=sio.Video(filename="x.mp4"), frame_idx=0, instances=instances
            )
        ]
    )
    labels.update()  # populate the tracks / identities registries
    return labels


def test_identity_preferred_when_present():
    """An `Identity` is the wider label, so it wins over a co-present `Track`."""
    labels = _labels([("gerbil_1", "track_0"), ("gerbil_2", "track_1")])
    carrier = resolve_identity_carrier(labels)
    assert carrier == "identity"
    ids = [identity_index(i, labels, carrier) for i in labels[0].instances]
    assert sorted(ids) == [0, 1]


def test_track_used_when_no_identity():
    """Legacy track-only files (everything written before sleap-io 0.9) are unchanged."""
    labels = _labels([(None, "track_0"), (None, "track_1")])
    carrier = resolve_identity_carrier(labels)
    assert carrier == "track"
    ids = [identity_index(i, labels, carrier) for i in labels[0].instances]
    assert sorted(ids) == [0, 1]


def test_unlabeled_instances_get_negative_one():
    """No carrier value -> -1. Several such in one frame collapse; the loader warns."""
    labels = _labels([(None, None), (None, None)])
    carrier = resolve_identity_carrier(labels)
    ids = [identity_index(i, labels, carrier) for i in labels[0].instances]
    assert ids == [-1, -1]


def test_mixed_file_does_not_collide_two_animals():
    """The reason the carrier is per-FILE and not per-instance.

    `identities.index(...)` and `tracks.index(...)` are independent index spaces
    that both start at 0. Resolving per instance ("identity if present, else
    track") therefore hands an identity-labeled animal and a track-labeled one the
    SAME `gt_track_id`, asserting two different animals are one. A uniform carrier
    makes that unrepresentable: the instance missing the chosen carrier reads -1
    (and is counted and warned about) instead of aliasing onto a real animal.
    """
    labels = _labels([("gerbil_1", None), (None, "track_0")])
    instances = labels[0].instances

    naive = [
        labels.identities.index(i.identity)
        if i.identity is not None
        else (labels.tracks.index(i.track) if i.track is not None else -1)
        for i in instances
    ]
    assert naive == [0, 0], "expected the per-instance collision this guards against"

    carrier = resolve_identity_carrier(labels)
    uniform = [identity_index(i, labels, carrier) for i in instances]
    assert carrier == "identity"
    assert uniform == [0, -1]
    assert uniform[0] != uniform[1]


@pytest.mark.parametrize("carrier", ["identity", "track"])
def test_index_matches_the_registry_order(carrier):
    """The returned index addresses the matching registry on `Labels`."""
    specs = (
        [("a", None), ("b", None)] if carrier == "identity" else [(None, "a"), (None, "b")]
    )
    labels = _labels(specs)
    assert resolve_identity_carrier(labels) == carrier
    registry = labels.identities if carrier == "identity" else labels.tracks
    for instance in labels[0].instances:
        idx = identity_index(instance, labels, carrier)
        value = instance.identity if carrier == "identity" else instance.track
        assert registry[idx] is value
