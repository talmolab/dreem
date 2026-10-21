"""Test model modules."""

import pytest
import torch

from dreem.io import Frame, Instance
from dreem.models import (
    Embedding,
    GlobalTrackingTransformer,
    Transformer,
    VisualEncoder,
)
from dreem.models.attention_head import ATTWeightHead
from dreem.models.mlp import MLP
from dreem.models.model_utils import get_boxes, get_times
from dreem.models.transformer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from dreem.models.visual_encoder import GeM


# todo: add named tensor tests
def test_mlp():
    """Test MLP logic."""
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    mlp = MLP(input_dim=f, hidden_dim=f, output_dim=f, num_layers=2, dropout=0.1)

    output_tensor = mlp(torch.rand(size=(b, n, f)))

    assert output_tensor.shape == (b, n, f)


def test_att_weight_head():
    """Test self-attention head logic."""
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    att_weight_head = ATTWeightHead(feature_dim=f, num_layers=2, dropout=0.1)

    q = k = torch.rand(size=(b, n, f))

    attn_weights = att_weight_head(q, k)

    assert attn_weights.shape == (b, n, n)


def test_encoder_timm():
    """Test feature extractor logic using timm backend."""
    b, c, h, w = 1, 1, 100, 100  # batch size, channels, height, width

    features = 512
    input_tensor = torch.rand(b, c, h, w)
    backend = "timm"

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    assert not isinstance(encoder.feature_extractor, torch.nn.Sequential)

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 3
    input_tensor = torch.rand(b, c, h, w)

    features = 128

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 9
    input_tensor = torch.rand(b, c, h, w)

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)


def test_encoder_torch():
    """Test feature extractor logic using torchvision backend."""
    b, c, h, w = 1, 1, 100, 100  # batch size, channels, height, width

    features = 512
    input_tensor = torch.rand(b, c, h, w)
    backend = "torch"

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    assert isinstance(encoder.feature_extractor, torch.nn.Sequential)

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 3
    input_tensor = torch.rand(b, c, h, w)

    features = 128

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 9
    input_tensor = torch.rand(b, c, h, w)

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)


def test_encoder_pooling():
    """Test configurable global pooling (GeM / avg / max) in the visual encoder."""
    # GeM module: collapses spatial dims, and p=1 exactly recovers avg pooling.
    fmap = torch.rand(2, 5, 7, 7)
    assert GeM()(fmap).shape == (2, 5, 1, 1)
    assert torch.allclose(
        GeM(p=1.0)(fmap), torch.nn.AdaptiveAvgPool2d(1)(fmap), atol=1e-5
    )

    b, c, h, w = 2, 3, 64, 64
    x = torch.rand(b, c, h, w)

    # default pooling is avg, matching checkpoints that predate the option (their
    # config has no `pooling` key and their weights have no GeM exponent)
    enc = VisualEncoder(model_name="resnet18", in_chans=c, d_model=128, backend="torch")
    assert enc.pooling == "avg"
    assert not any(key.startswith("pool.") for key in enc.state_dict())

    # GeM has a learnable exponent initialized to 3.
    enc = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=128, backend="torch", pooling="gem"
    )
    assert isinstance(enc.pool, GeM)
    assert enc.pool.p.requires_grad
    assert enc.pool.p.item() == pytest.approx(3.0)

    # each pooling mode yields (b, d_model)
    for pooling in ["gem", "avg", "max"]:
        enc = VisualEncoder(
            model_name="resnet18",
            in_chans=c,
            d_model=128,
            backend="torch",
            pooling=pooling,
        )
        assert enc(x).shape == (b, 128)

    # unknown pooling raises
    with pytest.raises(ValueError):
        VisualEncoder(
            model_name="resnet18",
            in_chans=c,
            d_model=128,
            backend="torch",
            pooling="bogus",
        )


def test_embedding_validity():
    """Test embedding usage."""
    # this would throw assertion since embedding should be "pos"
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="learned", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="fixed", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="temporal", mode="learned", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="fixed", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="pos", mode="learn", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temp", mode="learn", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="pos", mode="fix", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temp", mode="fix", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="learn", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temporal", mode="learn", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="fix", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temporal", mode="learn", features=128)

    _ = Embedding(emb_type="temp", mode="learned", features=128)
    _ = Embedding(emb_type="pos", mode="learned", features=128)

    _ = Embedding(emb_type="pos", mode="learned", features=128)


def test_embedding_basic():
    """Test embedding logic."""
    frames = 32
    objects = 10
    d_model = 256
    n_anchors = 1

    N = frames * objects

    boxes = torch.rand(size=(N, n_anchors, 4))
    times = torch.rand(size=(N,))

    pos_emb = Embedding(
        emb_type="pos",
        mode="fixed",
        features=d_model,
        temperature=objects,
        normalize=True,
        scale=10,
    )

    sine_pos_emb = pos_emb(boxes)

    pos_emb = Embedding(emb_type="pos", mode="learned", features=d_model, emb_num=100)
    learned_pos_emb = pos_emb(boxes)

    temp_emb = Embedding(emb_type="temp", mode="learned", features=d_model, emb_num=16)
    learned_temp_emb = temp_emb(times)

    pos_emb_off = Embedding(emb_type="pos", mode="off", features=d_model)
    off_pos_emb = pos_emb_off(boxes)

    temp_emb_off = Embedding(emb_type="temp", mode="off", features=d_model)
    off_temp_emb = temp_emb_off(times)

    learned_emb_off = Embedding(emb_type="off", mode="learned", features=d_model)
    off_learned_emb_boxes = learned_emb_off(boxes)
    off_learned_emb_times = learned_emb_off(times)

    fixed_emb_off = Embedding(emb_type="off", mode="fixed", features=d_model)
    off_fixed_emb_boxes = fixed_emb_off(boxes)
    off_fixed_emb_times = fixed_emb_off(times)

    off_emb = Embedding(emb_type="off", mode="off", features=d_model)
    off_emb_boxes = off_emb(boxes)
    off_emb_times = off_emb(times)

    assert sine_pos_emb.size() == (N, d_model)
    assert learned_pos_emb.size() == (N, d_model)
    assert learned_temp_emb.size() == (N, d_model)

    assert not torch.equal(sine_pos_emb, learned_pos_emb)
    assert not torch.equal(sine_pos_emb, learned_temp_emb)
    assert not torch.equal(learned_pos_emb, learned_temp_emb)

    assert off_pos_emb.size() == (N, d_model)
    assert off_temp_emb.size() == (N, d_model)
    assert off_learned_emb_boxes.size() == (N, d_model)
    assert off_learned_emb_times.size() == (N, d_model)
    assert off_fixed_emb_boxes.size() == (N, d_model)
    assert off_fixed_emb_times.size() == (N, d_model)
    assert off_emb_boxes.size() == (N, d_model)
    assert off_emb_times.size() == (N, d_model)

    assert not off_pos_emb.any()
    assert not off_temp_emb.any()
    assert not off_learned_emb_boxes.any()
    assert not off_learned_emb_times.any()
    assert not off_fixed_emb_boxes.any()
    assert not off_fixed_emb_times.any()
    assert not off_emb_boxes.any()
    assert not off_emb_times.any()


def test_get_boxes_normalizes_each_axis():
    """Test that y is normalized by the image height and x by the width."""
    frame = Frame(video_id=0, frame_id=0, img_shape=(1, 100, 200))
    instance = Instance(bbox=torch.tensor([[10.0, 20.0, 30.0, 40.0]]))
    frame.instances = [instance]

    boxes = get_boxes([instance])
    assert torch.allclose(boxes, torch.tensor([[[0.1, 0.1, 0.3, 0.2]]]))

    # checkpoints trained before the fix divided y by the width and x by the height
    legacy_boxes = get_boxes([instance], legacy_axes=True)
    assert torch.allclose(legacy_boxes, torch.tensor([[[0.05, 0.2, 0.15, 0.4]]]))


def test_learned_temporal_embedding_rows():
    """Test that each frame offset gets its own row of the learned lookup table."""
    d_model = 32
    emb_num = 16
    offsets = torch.arange(-7, 1)  # signed offsets to the current frame

    temp_emb = Embedding(
        emb_type="temp", mode="learned", features=d_model, emb_num=emb_num
    )
    emb = temp_emb(offsets)
    # the current frame is the last row, earlier frames count back from it
    assert torch.equal(emb, temp_emb.lookup.weight[-8:])
    assert len(torch.unique(emb, dim=0)) == len(offsets)

    # offsets older than the table share its first row
    assert torch.equal(
        temp_emb(torch.tensor([-(emb_num + 4)]))[0], temp_emb.lookup.weight[0]
    )

    # legacy mode (window ranks) collapses every frame after the first onto the
    # last row
    times = torch.arange(8)
    temp_emb.legacy = True
    legacy_emb = temp_emb(times)
    # legacy computes e.g. 97 * row - 96 * row for the clamped frames, so allow
    # the float cancellation error that grows with the frame index
    assert torch.allclose(legacy_emb[0], temp_emb.lookup.weight[0], atol=1e-4)
    assert torch.allclose(
        legacy_emb[1:],
        temp_emb.lookup.weight[-1].expand(len(times) - 1, -1),
        atol=1e-4,
    )


def test_get_times_signed_offsets():
    """Test that times are signed offsets to the current frame, or legacy ranks."""
    frames = [Frame(video_id=0, frame_id=i, img_shape=(1, 8, 8)) for i in (20, 21, 24)]
    instances = []
    for frame in frames:
        frame.instances = [Instance(bbox=torch.rand(1, 4)) for _ in range(2)]
        instances.extend(frame.instances)
    ref, query = instances[:4], instances[4:]

    ref_t, query_t = get_times(ref, query)
    assert ref_t.tolist() == [-4, -4, -3, -3]  # real frame gaps are kept
    assert query_t.tolist() == [0, 0]

    # without queries, the most recent ref frame is the current one
    ref_t, query_t = get_times(instances)
    assert ref_t.tolist() == [-4, -4, -3, -3, 0, 0]
    assert len(query_t) == 0

    # checkpoints trained before signed offsets used each frame's window rank
    ref_t, query_t = get_times(ref, query, legacy=True)
    assert ref_t.tolist() == [0, 0, 1, 1]
    assert query_t.tolist() == [2, 2]


def test_fixed_temporal_embedding_signed_offsets():
    """Test the sinusoidal temporal embedding on signed and legacy times."""
    d_model = 32
    temp_emb = Embedding(emb_type="temp", mode="fixed", features=d_model)

    emb = temp_emb(torch.arange(-3, 1))
    assert emb.shape == (4, d_model)
    assert len(torch.unique(emb, dim=0)) == 4
    # offset 0 (the current frame) is sin(0)=0, cos(0)=1
    assert torch.allclose(emb[-1, 0::2], torch.zeros(d_model // 2))
    assert torch.allclose(emb[-1, 1::2], torch.ones(d_model // 2))


@pytest.mark.parametrize("emb_type", ["pos", "temp"])
def test_rff_embedding(emb_type):
    """Test that the random Fourier features approximate a Gaussian kernel.

    Args:
        emb_type: which embedding to test
    """
    torch.manual_seed(0)
    d_model = 4096  # enough random frequencies for a tight kernel estimate
    sigma = 0.2 if emb_type == "pos" else 4.0
    emb = Embedding(emb_type=emb_type, mode="rff", features=d_model, sigma=sigma)
    omega = "rff_box_omega" if emb_type == "pos" else "rff_omega"
    # the random frequencies are saved with the checkpoint
    assert omega in emb.state_dict()

    if emb_type == "pos":
        x = torch.linspace(0.1, 0.5, 5)
        inputs = torch.stack([x, x, x + 0.05, x + 0.05], dim=-1).unsqueeze(1)
        # squared distance over the 4 box coordinates
        sq_dist = 4 * (x - x[0]) ** 2
    else:
        inputs = -torch.arange(5)
        # offsets are tanh-warped with horizon 3 * sigma before the kernel
        horizon = 3 * sigma
        warped = horizon * torch.tanh(inputs.float() / horizon)
        sq_dist = (warped - warped[0]) ** 2
    out = emb(inputs)
    assert out.shape == (5, d_model)

    expected = torch.exp(-sq_dist / (2 * sigma**2))
    assert torch.allclose(out @ out[0], expected, atol=0.05)

    with pytest.raises(ValueError):
        Embedding(emb_type=emb_type, mode="rff", features=d_model + 1)


def test_transformer_concat_fusion_rff():
    """Test the transformer with concatenated visual/pos/temp RFF embeddings."""
    feats = 64
    num_frames = 3
    num_detected = 4

    frames = []
    for i in range(num_frames):
        instances = [
            Instance(bbox=torch.rand(size=(1, 4)), features=torch.rand(size=(1, feats)))
            for _ in range(num_detected)
        ]
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=(1, 64, 64), instances=instances)
        )
    instances = [instance for frame in frames for instance in frame.instances]

    embedding_meta = {
        "embedding_agg_method": "concatenate",
        "pos": {"mode": "rff", "sigma": 0.2},
        "temp": {"mode": "rff", "sigma": 4.0},
    }
    transformer = Transformer(
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
    )
    assert (transformer.d_visual, transformer.d_pos, transformer.d_temp) == (32, 16, 16)

    asso_preds = transformer(instances[:-num_detected], instances[-num_detected:])
    assert asso_preds[0].matrix.size() == (num_detected, num_detected * 2)


def test_fixed_pos_embedding_normalize():
    """Test that the fixed embedding with `normalize=True` depends on position."""
    d_model = 32
    boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.2]], [[0.6, 0.7, 0.8, 0.9]]])

    pos_emb = Embedding(emb_type="pos", mode="fixed", features=d_model, normalize=True)
    emb = pos_emb(boxes)
    assert not torch.allclose(emb[0], emb[1])

    unnormalized = Embedding(emb_type="pos", mode="fixed", features=d_model)
    assert torch.allclose(emb, unnormalized(boxes * 2 * torch.pi))

    # legacy mode divides by the anchor dim, giving the same embedding everywhere
    pos_emb.legacy = True
    legacy_emb = pos_emb(boxes)
    assert torch.allclose(legacy_emb[0], legacy_emb[1], atol=1e-3)


def test_legacy_embeddings_propagate():
    """Test that the legacy switch reaches both embeddings of the transformer."""
    embedding_meta = {
        "pos": {"mode": "learned", "emb_num": 16},
        "temp": {"mode": "learned", "emb_num": 16},
    }
    model = GlobalTrackingTransformer(
        encoder_cfg={"model_name": "resnet18", "pretrained": False, "in_chans": 3},
        d_model=64,
        nhead=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
    )
    transformer = model.transformer

    assert not model.legacy_embeddings
    assert not transformer.pos_emb.legacy
    assert not transformer.temp_emb.legacy

    model.legacy_embeddings = True
    assert transformer.legacy_embeddings
    assert transformer.pos_emb.legacy
    assert transformer.temp_emb.legacy


def test_embedding_kwargs():
    """Test embedding config logic."""
    frames = 32
    objects = 10

    N = frames * objects
    n_anchors = 1

    boxes = torch.rand(N, n_anchors, 4)

    # sine embedding

    sine_args = {
        "temperature": objects,
        "scale": frames,
        "normalize": True,
    }
    sine_no_args = Embedding("pos", "fixed", 128)
    sine_with_args = Embedding("pos", "fixed", 128, **sine_args)

    assert sine_no_args.temperature != sine_with_args.temperature

    sine_no_args = sine_no_args(boxes)
    sine_with_args = sine_with_args(boxes)

    assert not torch.equal(sine_no_args, sine_with_args)

    # learned pos embedding

    lp_no_args = Embedding("pos", "learned", 128)

    lp_args = {"emb_num": 100, "over_boxes": False}

    lp_with_args = Embedding("pos", "learned", 128, **lp_args)
    assert lp_no_args.lookup.weight.shape != lp_with_args.lookup.weight.shape

    # learned temp embedding

    lt_no_args = Embedding("temp", "learned", 128)

    lt_args = {"emb_num": 100}

    lt_with_args = Embedding("temp", "learned", 128, **lt_args)
    assert lt_no_args.lookup.weight.shape != lt_with_args.lookup.weight.shape


def test_multianchor_embedding():
    """Test multi-anchor embedding computation."""
    frames = 32
    objects = 10
    d_model = 256
    n_anchors = 10
    features = 256

    N = frames * objects

    boxes = torch.rand(size=(N, n_anchors, 4))

    fixed_emb = Embedding(
        "pos",
        "fixed",
        features=features,
        n_points=n_anchors,
        mlp_cfg={"num_layers": 3, "hidden_dim": 2 * d_model},
    )
    learned_emb = Embedding(
        "pos",
        "learned",
        features=features,
        n_points=n_anchors,
        mlp_cfg={"num_layers": 3, "hidden_dim": 2 * d_model},
    )
    assert not isinstance(fixed_emb.mlp, torch.nn.Identity)
    assert not isinstance(learned_emb.mlp, torch.nn.Identity)

    emb = fixed_emb(boxes)
    assert emb.size() == (N, features)

    emb = learned_emb(boxes)
    assert emb.size() == (N, features)

    fixed_emb = Embedding("pos", "fixed", features=features)
    learned_emb = Embedding("pos", "learned", features=features)
    with pytest.raises(RuntimeError):
        _ = fixed_emb(boxes)
    with pytest.raises(RuntimeError):
        _ = learned_emb(boxes)


def test_transformer_encoder():
    """Test transformer encoder layer logic."""
    feats = 256

    transformer_encoder = TransformerEncoderLayer(
        d_model=feats, nhead=1, dim_feedforward=feats, norm=True
    )

    N, B, D = 10, 1, feats

    # no position
    queries = torch.rand(size=(N, B, D))

    encoder_features = transformer_encoder(queries)

    assert encoder_features.size() == queries.size()

    # with position
    pos_emb = torch.ones_like(queries)

    encoder_features = transformer_encoder(queries, pos_emb=pos_emb)

    assert encoder_features.size() == encoder_features.size()


def test_transformer_decoder():
    """Test transformer decoder layer logic."""
    feats = 512

    transformer_decoder = TransformerDecoderLayer(
        d_model=feats,
        nhead=2,
        dim_feedforward=feats,
        dropout=0.2,
        norm=False,
        decoder_self_attn=True,
    )

    N, B, D = 10, 1, feats

    # no position
    decoder_queries = encoder_features = torch.rand(size=(N, B, D))

    decoder_features = transformer_decoder(decoder_queries, encoder_features)

    assert decoder_features.size() == decoder_queries.size()

    # with position
    pos_emb = query_pos_emb = torch.ones_like(encoder_features)

    decoder_features = transformer_decoder(
        decoder_queries,
        encoder_features,
        ref_pos_emb=pos_emb,
        query_pos_emb=query_pos_emb,
    )

    assert decoder_features.size() == decoder_queries.size()


def test_transformer_basic():
    """Test full transformer logic."""
    feats = 256
    num_frames = 32
    num_detected = 10
    img_shape = (1, 100, 100)

    transformer = Transformer(d_model=feats, num_encoder_layers=1, num_decoder_layers=1)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), features=torch.rand(size=(1, feats))
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    instances = [instance for frame in frames for instance in frame.instances]
    asso_preds = transformer(instances)

    assert asso_preds[0].matrix.size() == (num_detected * num_frames,) * 2


def test_transformer_embedding():
    """Test transformer using embedding."""
    feats = 256
    num_frames = 3
    num_detected = 10

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), features=torch.rand(size=(1, feats))
                )
            )
        frames.append(Frame(video_id=0, frame_id=i, instances=instances))

    instances = [instance for frame in frames for instance in frame.instances]

    embedding_meta = {
        "pos": {"mode": "learned", "emb_num": 16, "normalize": True},
        "temp": {"mode": "learned", "emb_num": 16, "normalize": True},
    }

    transformer = Transformer(
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )

    assert transformer.pos_emb.mode == "learned"
    assert transformer.temp_emb.mode == "learned"

    asso_preds = transformer(instances)

    assert asso_preds[0].matrix.size() == (num_detected * num_frames,) * 2

    pos_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )
    temp_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )

    assert pos_emb.size() == (
        len(instances),
        feats,
    ), pos_emb.shape

    assert temp_emb.size() == (
        len(instances),
        feats,
    ), temp_emb.shape


def test_tracking_transformer():
    """Test GTR logic."""
    feats = 512
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), crop=torch.rand(size=(1, 1, 64, 64))
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    embedding_meta = {
        "pos": {
            "mode": "fixed",
            "temperature": num_detected,
            "scale": num_frames,
            "normalize": True,
        },
        "temp": None,
    }

    encoder_cfg = {"model_name": "resnet18", "pretrained": False, "in_chans": 3}

    tracking_transformer = GlobalTrackingTransformer(
        encoder_cfg=encoder_cfg,
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )
    instances = [instance for frame in frames for instance in frame.instances]
    asso_preds = tracking_transformer(instances)

    assert asso_preds[0].matrix.size() == (num_detected * num_frames,) * 2

    pos_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )
    temp_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )

    assert pos_emb.size() == (
        len(instances),
        feats,
    ), pos_emb.shape

    assert temp_emb.size() == (
        len(instances),
        feats,
    ), temp_emb.shape
