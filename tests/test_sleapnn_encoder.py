"""Tests for the sleap-nn visual encoder (`encoder_type: sleapnn`).

For the default frozen mode, these assert the three properties that make
"frozen" true, because each fails independently and two of them fail *silently*:

  * no parameter accumulates a gradient;
  * the encoder stays in eval mode even after Lightning calls `.train()`;
  * the weights are bit-identical after a backward pass.

The `trainable_encoder` tests at the end check the fine-tuning mode.

Skipped unless sleap-nn is installed and a checkpoint is available. Point
`DREEM_SLEAPNN_CKPT` / `DREEM_SLEAPNN_CFG` at a run directory's `best.ckpt` and
`training_config.yaml`.
"""

import os

import pytest
import torch

from dreem.models.visual_encoder import create_visual_encoder

pytest.importorskip("sleap_nn", reason="sleap-nn extra not installed")

CKPT = os.environ.get("DREEM_SLEAPNN_CKPT")
CFG = os.environ.get("DREEM_SLEAPNN_CFG")

pytestmark = pytest.mark.skipif(
    not (CKPT and CFG and os.path.exists(CKPT) and os.path.exists(CFG)),
    reason="set DREEM_SLEAPNN_CKPT and DREEM_SLEAPNN_CFG to run",
)


@pytest.fixture(scope="module")
def encoder():
    return create_visual_encoder(
        d_model=128,
        encoder_type="sleapnn",
        encoder_type_args={"checkpoint": CKPT, "config": CFG},
    )


def test_output_shape_and_norm(encoder):
    """Emits (B, d_model) unit vectors, like VisualEncoder does."""
    out = encoder(torch.rand(4, 3, 128, 128))
    assert out.shape == (4, 128)
    torch.testing.assert_close(
        out.norm(dim=1), torch.ones(4), rtol=1e-4, atol=1e-4
    )


def test_accepts_one_or_three_channels(encoder):
    """DREEM may hand either; both must reach the 1-channel encoder."""
    assert encoder(torch.rand(2, 1, 128, 128)).shape == (2, 128)
    assert encoder(torch.rand(2, 3, 128, 128)).shape == (2, 128)


def test_grayscale_conversion_matches_luma(encoder):
    """A grey RGB crop equals the same grey fed as 1 channel.

    Guards the ITU-R 601 weights: any other weighting makes a colour DREEM crop
    land somewhere the encoder was never trained.
    """
    g = torch.rand(2, 1, 128, 128)
    rgb = g.repeat(1, 3, 1, 1)
    torch.testing.assert_close(encoder(rgb), encoder(g), rtol=1e-5, atol=1e-5)


def test_channel_selects_one_channel_of_an_rgb_crop():
    """`channel=k` feeds channel k alone, not a luma mix.

    An encoder trained on sleap-io's `grayscale=True` frames saw channel 0 (red)
    only; giving it luma would ask it about an image it never trained on.
    """
    red = create_visual_encoder(
        d_model=128,
        encoder_type="sleapnn",
        encoder_type_args={"checkpoint": CKPT, "config": CFG, "channel": 0},
    )
    rgb = torch.rand(2, 3, 128, 128)
    torch.testing.assert_close(red(rgb), red(rgb[:, :1]), rtol=1e-5, atol=1e-5)
    # ...and it really is a different input from luma for a colour crop.
    luma = (rgb * torch.tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(
        1, keepdim=True
    )
    assert not torch.allclose(red(rgb), red(luma), atol=1e-3)


def test_channel_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="channel"):
        create_visual_encoder(
            d_model=128,
            encoder_type="sleapnn",
            encoder_type_args={"checkpoint": CKPT, "config": CFG, "channel": 3},
        )


def test_no_gradients_reach_the_encoder(encoder):
    """Fails if requires_grad_(False) or the no_grad forward regressed."""
    out = encoder(torch.rand(2, 3, 128, 128))
    assert not out.requires_grad
    out.sum().backward() if out.requires_grad else None
    assert all(p.grad is None for p in encoder.encoder.parameters())


def test_stays_in_eval_after_train_called(encoder):
    """Lightning calls model.train() each epoch; the encoder must not follow.

    Without the train() override the encoder's norm layers keep updating their
    running statistics and a "frozen" encoder drifts -- with no error anywhere.
    """
    encoder.train()
    assert not encoder.encoder.training


def test_weights_unchanged_by_a_backward_pass(encoder):
    """End-to-end: run a step and prove the frozen weights did not move."""
    before = {k: v.detach().clone() for k, v in encoder.encoder.state_dict().items()}
    head = torch.nn.Linear(128, 1)
    opt = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=1.0)
    opt.zero_grad()
    head(encoder(torch.rand(4, 3, 128, 128))).sum().backward()
    opt.step()
    after = encoder.encoder.state_dict()
    for k, v in before.items():
        torch.testing.assert_close(after[k], v, rtol=0, atol=0,
                                   msg=f"frozen weight changed: {k}")


def test_dim_mismatch_is_an_error_not_a_silent_reshape():
    """d_model != embedding_dim must raise unless a projection is requested."""
    with pytest.raises(ValueError, match="project"):
        create_visual_encoder(
            d_model=256,
            encoder_type="sleapnn",
            encoder_type_args={"checkpoint": CKPT, "config": CFG},
        )


def test_projection_is_trainable_while_encoder_is_not():
    """With project=true only the Linear learns."""
    enc = create_visual_encoder(
        d_model=256,
        encoder_type="sleapnn",
        encoder_type_args={"checkpoint": CKPT, "config": CFG, "project": True},
    )
    out = enc(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 256)
    out.sum().backward()
    assert enc.project.weight.grad is not None
    assert all(p.grad is None for p in enc.encoder.parameters())


def test_missing_paths_raise_clearly():
    with pytest.raises(ValueError, match="checkpoint"):
        create_visual_encoder(d_model=128, encoder_type="sleapnn",
                              encoder_type_args={})


def test_device_bookkeeping_follows_a_parent_move(encoder):
    """Moving a PARENT must update the inner LightningModule's `self.device`.

    Lightning updates `_device` only on a direct `.to()`. `nn.Module._apply`
    relocates a child's weights without it, so the inner forward would then do
    `img.to(self.device)` back to the stale device and the first conv raises
    "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)".
    Simulated here without a GPU by planting a stale device and moving the
    parent.
    """
    encoder.encoder._device = torch.device("cuda", 0)   # stale on purpose
    encoder.to("cpu")
    param = next(encoder.encoder.parameters())
    assert encoder.encoder._device == param.device, (
        "inner module's device did not follow the parent move"
    )
    assert encoder(torch.rand(2, 3, 128, 128)).shape == (2, 128)


@pytest.fixture(scope="module")
def trainable_encoder():
    """The same encoder with `trainable=true`."""
    return create_visual_encoder(
        d_model=128,
        encoder_type="sleapnn",
        encoder_type_args={"checkpoint": CKPT, "config": CFG, "trainable": True},
    )


def test_trainable_encoder_receives_gradients(trainable_encoder):
    """With trainable=true every embedder weight is trainable and gets a grad.

    The train-only SupCon `projection` head is not part of the forward, so it
    must be frozen rather than sit in the optimizer with no gradient.
    """
    trainable_encoder.zero_grad()
    out = trainable_encoder(torch.rand(2, 3, 128, 128))
    assert out.requires_grad
    out.sum().backward()

    embedder = dict(trainable_encoder.encoder.model.named_parameters())
    assert embedder and all(p.requires_grad for p in embedder.values())
    missing = [n for n, p in embedder.items() if p.grad is None]
    assert not missing, f"embedder weights with no gradient: {missing}"

    others = [
        n for n, p in trainable_encoder.encoder.named_parameters()
        if p.requires_grad and not n.startswith("model.")
    ]
    assert not others, f"trainable weights outside the embedder: {others}"


def test_trainable_encoder_follows_train_eval(trainable_encoder):
    """A trainable encoder must follow the parent's mode, unlike a frozen one."""
    trainable_encoder.train()
    assert trainable_encoder.encoder.training
    trainable_encoder.eval()
    assert not trainable_encoder.encoder.training


def test_trainable_weights_move_after_a_step(trainable_encoder):
    """End-to-end: one optimizer step changes the encoder's weights."""
    before = {k: v.detach().clone() for k, v in trainable_encoder.encoder.state_dict().items()}
    opt = torch.optim.SGD(trainable_encoder.parameters(), lr=1.0)
    opt.zero_grad()
    trainable_encoder(torch.rand(4, 3, 128, 128)).sum().backward()
    opt.step()
    after = trainable_encoder.encoder.state_dict()
    assert any(not torch.equal(after[k], v) for k, v in before.items())
