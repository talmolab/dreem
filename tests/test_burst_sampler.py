"""Tests for burst chunk sampling (`burst_aug`).

The property that matters is the one the design exists for: a single chunk must
contain BOTH short-range and long-range pairs, so the association loss -- which
scores every pair in the window -- has to tell the two regimes apart inside one
forward pass. An evenly-strided chunk (the older `gap_aug`) fails this: its
smallest pairwise gap equals its stride.
"""

import numpy as np
import pytest
import torch

from dreem.datasets.base_dataset import BaseDataset


class _Sampler(BaseDataset):
    """Minimal harness exposing _burst_chunks without building a real dataset."""

    def __init__(self, n_bursts=4, burst_max_span=900, chunk=True):
        self.n_bursts = n_bursts
        self.burst_max_span = burst_max_span
        self.burst_aug = True
        self.chunk = chunk


def pair_gaps(chunk):
    a = chunk.numpy()
    return np.abs(a[:, None] - a[None, :])[np.triu_indices(len(a), k=1)]


@pytest.fixture
def seg():
    return torch.arange(900)


def test_chunks_have_the_right_length(seg):
    chunks = _Sampler()._burst_chunks(seg, 32)
    full = [c for c in chunks if len(c) == 32]
    assert len(full) >= 20
    assert all(len(c) <= 32 for c in chunks)


def test_indices_are_sorted_distinct_and_in_range(seg):
    for c in _Sampler()._burst_chunks(seg, 32):
        a = c.numpy()
        assert len(np.unique(a)) == len(a), "duplicate frame in a chunk"
        assert (np.diff(a) > 0).all(), "chunk indices not sorted"
        assert a.min() >= 0 and a.max() < len(seg)


def test_chunk_contains_both_short_and_long_pairs(seg):
    """The whole point: both regimes inside one window.

    Without this a burst chunk is just a strided chunk, and the model never has
    to distinguish "position is reliable" from "position is useless".
    """
    np.random.seed(0)
    chunks = [c for c in _Sampler()._burst_chunks(seg, 32) if len(c) == 32]
    spread = [c for c in chunks if pair_gaps(c).max() > 100]
    assert spread, "no chunk spanned a long gap"
    for c in spread:
        g = pair_gaps(c)
        assert g.min() == 1, "no consecutive-frame pair (lost the motion regime)"
        assert g.max() > 100, "no long-range pair (lost the appearance regime)"


def test_bursts_are_runs_of_consecutive_frames(seg):
    """n_bursts contiguous runs, so within-burst gaps really are 1."""
    np.random.seed(1)
    for c in _Sampler(n_bursts=4)._burst_chunks(seg, 32):
        if len(c) != 32:
            continue
        runs = 1 + int((np.diff(c.numpy()) > 1).sum())
        assert 1 <= runs <= 4, f"expected <=4 runs, got {runs}"


def test_span_never_exceeds_the_configured_max(seg):
    np.random.seed(2)
    for c in _Sampler(burst_max_span=200)._burst_chunks(seg, 32):
        if len(c) > 1:
            assert int(c[-1] - c[0]) < 200


def test_gaps_cover_a_wide_range_across_chunks(seg):
    """Log-uniform spans should populate the whole ladder, not cluster."""
    np.random.seed(3)
    mx = [pair_gaps(c).max() for c in _Sampler()._burst_chunks(seg, 32) if len(c) == 32]
    assert min(mx) < 64, "no near-contiguous chunks"
    assert max(mx) > 300, "no widely-spread chunks"


def test_short_segment_falls_back_to_contiguous():
    """A segment shorter than clip_length is returned as-is, not padded."""
    c = _Sampler()._burst_chunks(torch.arange(20), 32)
    assert len(c) == 1 and len(c[0]) == 20


def test_single_burst_is_plain_contiguous(seg):
    """n_bursts=1 degenerates to the old behaviour: one consecutive run."""
    np.random.seed(4)
    for c in _Sampler(n_bursts=1)._burst_chunks(seg, 32):
        if len(c) == 32:
            assert (np.diff(c.numpy()) == 1).all()


def test_more_bursts_than_frames_does_not_crash(seg):
    chunks = _Sampler(n_bursts=64)._burst_chunks(seg, 32)
    assert all(len(np.unique(c.numpy())) == len(c) for c in chunks)
