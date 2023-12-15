"""Test model modules."""
import pytest
import torch
import numpy as np
from biogtr.data_structures import Frame, Instance
from biogtr.models.attention_head import MLP, ATTWeightHead
from biogtr.models.embeddings.spatial_embedding import SpatialEmbedding
from biogtr.models.embeddings.temporal_embedding import TemporalEmbedding
from biogtr.models.embeddings.relative_positional_embedding import (
    RelativePositionalMask,
)
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.models.transformer import (
    Transformer,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

from biogtr.models.feature_encoders.visual_encoder import VisualEncoder
from biogtr.models.feature_encoders.flow_encoder import FlowEncoder
from biogtr.models.feature_encoders.lsd_encoder import LSDEncoder
from biogtr.models.feature_encoders.feature_encoder import FeatureEncoder
from biogtr.models.feature_encoders import fusion_layers


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


def test_visual_encoder():
    """Test visual feature extractor logic."""
    b, c, h, w = 1, 1, 100, 100  # batch size, channels, height, width

    features = 512
    input_tensor = torch.rand(b, c, h, w)

    for model_name, weights_name in [
        ("resnet18", "ResNet18_Weights.DEFAULT"),
        ("resnet50", "ResNet50_Weights.DEFAULT"),
    ]:
        cfg = {"weights": weights_name}

        encoder = VisualEncoder(model_name, cfg, features)

        output = encoder(input_tensor)

        assert output.shape == (b, features)


def test_flow_encoder():
    b, c, h, w = 1, 2, 100, 100

    features = 512
    input_tensor = torch.rand(b, c, h, w)
    encoder = FlowEncoder(
        {"model_name": "resnet50", "pretrained": False}, d_model=features
    )

    output = encoder(input_tensor)
    assert output.shape == (b, features)


def test_lsd_encoder():
    b, c, d, h, w = 1, 3, 6, 100, 100

    features = 512
    input_tensor = torch.rand(b, d, h, w)
    encoder = LSDEncoder(unet_cfg=None, d_model=features)

    output = encoder(input_tensor)
    assert output.shape == (b, features)

    input_tensor = torch.rand(b, c, h, w)
    with pytest.raises(ValueError):
        encoder(input_tensor)

    input_tensor = torch.rand(b, c, h, w)
    encoder = LSDEncoder(unet_cfg={}, d_model=features)

    output = encoder(input_tensor)
    assert output.shape == (b, features)

    input_tensor = torch.rand(b, d, h, w)
    with pytest.raises(ValueError):
        encoder(input_tensor)


def test_fusion_layers():
    b, d = 1, 512
    vis_feats, flow_feats, lsd_feats = torch.randn(3, b, d)
    fusion_layer = fusion_layers.Cat(d)
    fused_feats = fusion_layer([vis_feats, flow_feats, lsd_feats])
    assert fused_feats.shape == (b, d)

    fusion_layer = fusion_layers.Sum(d)
    fused_feats = fusion_layer([vis_feats, flow_feats, lsd_feats])
    assert fused_feats.shape == (b, d)


def test_feature_encoder():
    b, c, h, w, d = 1, 1, 100, 100, 512

    crop = torch.randn(b, c, h, w)
    flow = torch.randn(b, 2, h, w)
    lsd = torch.randn(b, 6, h, w)

    encoder = FeatureEncoder(
        visual_encoder_cfg={
            "model_name": "resnet50",
            "cfg": {"weights": "ResNet50_Weights.DEFAULT"},
        },
        flow_encoder_cfg={"model_name": "resnet50", "pretrained": False},
        lsd_encoder_cfg={"unet_cfg": None},
        out_dim=d,
        fusion_layer="cat",
    )
    assert isinstance(encoder.out_layer, fusion_layers.Cat)

    feats = encoder(
        crops=crop, flows=flow, lsds=lsd, feats_to_return=("visual", "flow", "lsd")
    )
    assert "visual" in feats and "flow" in feats and "lsd" in feats
    assert feats["combined"].shape == (b, d)

    encoder = FeatureEncoder(
        visual_encoder_cfg=None,
        flow_encoder_cfg={"model_name": "resnet50", "pretrained": False},
        lsd_encoder_cfg={"unet_cfg": None},
        out_dim=d,
        fusion_layer="cat",
    )
    feats = encoder(
        crops=crop, flows=flow, lsds=lsd, feats_to_return=("visual", "flow", "lsd")
    )
    assert "visual" not in feats and "flow" in feats and "lsd" in feats
    assert feats["combined"].shape == (b, d)

    feats = encoder(crops=crop, flows=flow, lsds=lsd, feats_to_return=("flow", "lsd"))
    assert "visual" not in feats and "flow" in feats and "lsd" in feats
    assert feats["combined"].shape == (b, d)

    feats = encoder(crops=crop, flows=flow, lsds=lsd, feats_to_return=("lsd"))
    assert "visual" not in feats and "flow" not in feats and "lsd" in feats
    assert feats["combined"].shape == (b, d)

    feats = encoder(crops=crop, flows=flow, lsds=lsd, feats_to_return=())
    assert "visual" not in feats and "flow" not in feats and "lsd" not in feats
    assert feats["combined"].shape == (b, d)

    feats = encoder(
        crops=None, flows=flow, lsds=lsd, feats_to_return=("visual", "flow", "lsd")
    )
    assert "visual" not in feats and "flow" in feats and "lsd" in feats
    assert feats["combined"].shape == (b, d)

    encoder = FeatureEncoder(
        visual_encoder_cfg=None,
        flow_encoder_cfg=None,
        lsd_encoder_cfg={"unet_cfg": None},
        out_dim=d,
        fusion_layer="cat",
    )
    feats = encoder(
        crops=crop, flows=flow, lsds=lsd, feats_to_return=("visual", "flow", "lsd")
    )
    assert "visual" not in feats and "flow" not in feats and "lsd" in feats
    assert feats["combined"].shape == (b, d)

    feats = encoder(
        crops=None, flows=None, lsds=lsd, feats_to_return=("visual", "flow", "lsd")
    )
    assert "visual" not in feats and "flow" not in feats and "lsd" in feats
    assert feats["combined"].shape == (b, d)

    encoder = FeatureEncoder(
        visual_encoder_cfg=None,
        flow_encoder_cfg=None,
        lsd_encoder_cfg=None,
        out_dim=d,
        fusion_layer="cat",
    )
    feats = encoder(
        crops=crop, flows=flow, lsds=lsd, feats_to_return=("visual", "flow", "lsd")
    )

    assert "visual" not in feats and "flow" not in feats and "lsd" not in feats
    assert feats["combined"].shape == (b, d)


def test_embedding():
    """Test embedding logic."""
    frames = 32
    objects = 10
    d_model = 256
    nhead = 8
    cutoff_spatial = 256
    cutoff_temporal = 32

    pos_emb = SpatialEmbedding(
        features=d_model // 4,
        temperature=objects,
        normalize=True,
        scale=10,
        learn_emb_num=100,
    )
    temp_emb = TemporalEmbedding(features=d_model, learn_emb_num=16)
    rel_pos_mask = RelativePositionalMask(nhead, cutoff_spatial, cutoff_temporal)

    N = frames * objects

    boxes = torch.rand(size=(N, 4))
    coords = torch.stack(
        [
            (boxes[:, -2] + boxes[:, 2]) / 2,
            (boxes[:, -1] - boxes[:, 1]) / 2,
        ],
        dim=-1,
    )
    pred_coords = torch.randn(N, 3)
    times = torch.rand(size=(N,))

    sine_emb = pos_emb(boxes, emb_type="fixed")

    pos_emb = SpatialEmbedding(
        features=d_model,
        temperature=objects,
        normalize=True,
        scale=10,
        learn_emb_num=100,
    )

    learned_pos_emb = pos_emb(boxes, emb_type="learned")

    learned_temp_emb = temp_emb(times, emb_type="learned")

    assert sine_emb.size() == (N, d_model)
    assert learned_pos_emb.size() == (N, d_model)
    assert learned_temp_emb.size() == (N, d_model)
    with pytest.raises(NotImplementedError):
        learned_temp_emb = temp_emb(times, emb_type="fixed")

    mask = rel_pos_mask(coords)
    assert mask.shape == (nhead, N, N)

    query_inds = np.arange(objects)
    mask = rel_pos_mask(coords, query_inds=query_inds)
    assert mask.shape == (nhead, objects, objects)

    mask = rel_pos_mask(coords, query_inds=query_inds, mode="cross")
    assert mask.shape == (nhead, objects, N)


def test_transformer_encoder():
    """Test transformer encoder layer logic."""
    feats = 256

    transformer_encoder = TransformerEncoderLayer(
        d_model=feats, nhead=1, dim_feedforward=feats, norm=True
    )

    N, B, D = 10, 1, feats

    # no position
    src = torch.rand(size=(N, B, D))

    out = transformer_encoder(src)

    assert out.size() == src.size()

    # with position
    emb = torch.ones_like(src)

    out = transformer_encoder(src, emb=emb)

    assert out.size() == src.size()

    transformer_encoder = TransformerEncoderLayer(
        d_model=feats,
        nhead=1,
        dim_feedforward=feats,
        norm=True,
        rel_mask={"cutoff_spatial": 256, "cutoff_temporal": 32},
    )

    coords = torch.randn(N, 3)

    out = transformer_encoder(src, emb=emb, coords=coords)

    assert out.size() == src.size()


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
    tgt = memory = torch.rand(size=(N, B, D))

    out = transformer_decoder(tgt, memory)

    assert out.size() == tgt.size()

    # with position
    emb = tgt_emb = torch.ones_like(memory)

    out = transformer_decoder(tgt, memory, src_emb=emb, tgt_emb=tgt_emb)

    assert out.size() == tgt.size()

    # with relative positions

    transformer_decoder = TransformerDecoderLayer(
        d_model=feats,
        nhead=2,
        dim_feedforward=feats,
        dropout=0.2,
        norm=False,
        decoder_self_attn=True,
        rel_mask={"cutoff_spatial": 256, "cutoff_temporal": 32},
    )

    emb = tgt_emb = torch.ones_like(memory)

    coords = torch.randn(N, 3)

    out = transformer_decoder(tgt, memory, src_emb=emb, tgt_emb=tgt_emb, coords=coords)

    assert out.size() == tgt.size()


def test_transformer_basic():
    """Test full transformer logic."""
    feats = 256
    num_frames = 32
    num_detected = 10
    img_shape = (1, 100, 100)

    transformer = Transformer(
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
    )

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

    asso_preds, _ = transformer(frames)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2


def test_transformer_embedding():
    """Test transformer using embedding."""
    feats = 256
    num_frames = 3
    num_detected = 10
    img_shape = (1, 50, 50)

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
    embedding_meta = {
        "pos": {"type": "learned"},
        "temp": {"type": "learned"},
        "rel": None,
    }

    transformer = Transformer(
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )

    asso_preds, embedding = transformer(frames)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2
    assert embedding.size() == (num_detected * num_frames, 1, feats)


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
                    bbox=torch.rand(size=(1, 4)),
                    crop=torch.rand(size=(1, 1, 64, 64)),
                    flow=torch.rand(size=(1, 2, 64, 64)),
                    lsd=torch.rand(size=(1, 6, 64, 64)),
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    embedding_meta = {
        "pos": {"type": "learned"},
        "temp": {"type": "learned"},
        "rel": None,
    }

    encoder_cfg = {
        "visual_encoder_cfg": {
            "model_name": "resnet50",
            "cfg": {"weights": "ResNet50_Weights.DEFAULT"},
        },
        "flow_encoder_cfg": {"model_name": "resnet50", "pretrained": False},
        "lsd_encoder_cfg": {"unet_cfg": None},
        "fusion_layer": "cat",
    }

    tracking_transformer = GlobalTrackingTransformer(
        feature_encoder_cfg=encoder_cfg,
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )

    asso_preds, embedding = tracking_transformer(frames)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2
    assert embedding.size() == (num_detected * num_frames, 1, feats)
