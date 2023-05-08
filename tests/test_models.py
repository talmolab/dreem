import pytest
import torch
import numpy as np
from biogtr.models.attention_head import MLP, ATTWeightHead
from biogtr.models.embedding import Embedding
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.models.transformer import (
    Transformer,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from biogtr.models.visual_encoder import VisualEncoder
from biogtr.inference.tracker import Tracker

# todo: add named tensor tests
# todo: add fixtures

torch.set_default_device("cpu")


def test_mlp():
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    mlp = MLP(input_dim=f, hidden_dim=f, output_dim=f, num_layers=2, dropout=0.1)

    output_tensor = mlp(torch.rand(size=(b, n, f)))

    assert output_tensor.shape == (b, n, f)


def test_att_weight_head():
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    att_weight_head = ATTWeightHead(feature_dim=f, num_layers=2, dropout=0.1)

    q = k = torch.rand(size=(b, n, f))

    attn_weights = att_weight_head(q, k)

    assert attn_weights.shape == (b, n, n)


def test_encoder():
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


def test_embedding():
    emb = Embedding()

    frames = 32
    objects = 10
    d_model = 256

    N = frames * objects

    boxes = torch.rand(size=(N, 4))
    times = torch.rand(size=(N,))

    sine_emb = emb._sine_box_embedding(
        boxes, features=d_model // 4, temperature=objects, normalize=True, scale=10
    )

    learned_pos_emb = emb._learned_pos_embedding(
        boxes, features=d_model, learn_pos_emb_num=100
    )

    learned_temp_emb = emb._learned_temp_embedding(
        times, features=d_model, learn_temp_emb_num=16
    )

    assert sine_emb.size() == (N, d_model)
    assert learned_pos_emb.size() == (N, d_model)
    assert learned_temp_emb.size() == (N, d_model)


def test_embedding_kwargs():
    emb = Embedding()

    frames = 32
    objects = 10

    N = frames * objects

    boxes = torch.rand(size=(N, 4))
    times = torch.rand(size=(N,))

    # sine embedding

    sine_no_args = emb._sine_box_embedding(boxes)

    sine_args = {
        "temperature": objects,
        "scale": frames,
        "normalize": True,
    }

    sine_with_args = emb._sine_box_embedding(boxes, **sine_args)

    assert not torch.equal(sine_no_args, sine_with_args)

    # learned pos embedding

    lp_no_args = emb._learned_pos_embedding(boxes)

    lp_args = {"learn_pos_emb_num": 100, "over_boxes": False}

    lp_with_args = emb._learned_pos_embedding(boxes, **lp_args)

    assert not torch.equal(lp_no_args, lp_with_args)

    # learned temp embedding

    lt_no_args = emb._learned_temp_embedding(times)

    lt_args = {"learn_temp_emb_num": 100}

    lt_with_args = emb._learned_temp_embedding(times, **lt_args)

    assert not torch.equal(lt_no_args, lt_with_args)


def test_transformer_encoder():
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
    pos = torch.ones_like(src)

    out = transformer_encoder(src, pos=pos)

    assert out.size() == src.size()


def test_transformer_decoder():
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
    pos = tgt_pos = torch.ones_like(memory)

    out = transformer_decoder(tgt, memory, pos=pos, tgt_pos=tgt_pos)

    assert out.size() == pos.size()


def test_transformer_basic():
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

    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "frame_id": torch.tensor(i),
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "bboxes": torch.rand(size=(num_detected, 4)),
                "features": torch.rand(size=(num_detected, feats)),
            }
        )

    asso_preds = transformer(instances)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2


def test_transformer_embedding_validity():
    # use lower feats and single layer for efficiency
    feats = 256

    # this would throw assertion since no "embedding_type" key
    with pytest.raises(Exception):
        _ = Transformer(
            d_model=feats,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=feats,
            feature_dim_attn_head=feats,
            embedding_meta={"type": "learned_pos"},
        )

    # this would throw assertion since "embedding_type" value invalid
    with pytest.raises(Exception):
        _ = Transformer(
            d_model=feats,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=feats,
            feature_dim_attn_head=feats,
            embedding_meta={"embedding_type": "foo"},
        )

    # this would succeed
    _ = Transformer(
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
        embedding_meta={"embedding_type": "learned_pos"},
    )


def test_transformer_embedding():
    feats = 256
    num_frames = 3
    num_detected = 10
    img_shape = (1, 50, 50)

    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "frame_id": torch.tensor(i),
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "bboxes": torch.rand(size=(num_detected, 4)),
                "features": torch.rand(size=(num_detected, feats)),
            }
        )

    embedding_meta = {
        "embedding_type": "learned_pos_temp",
        "kwargs": {
            "learn_pos_emb_num": 16,
            "learn_temp_emb_num": 16,
            "normalize": True,
        },
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

    asso_preds, embedding = transformer(instances)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2
    assert embedding.size() == (num_detected * num_frames, 1, feats)


def test_tracking_transformer():
    feats = 512
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "frame_id": torch.tensor(i),
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "crops": torch.rand(size=(num_detected, 1, 64, 64)),
                "bboxes": torch.rand(size=(num_detected, 4)),
            }
        )

    embedding_meta = {
        "embedding_type": "fixed_pos",
        "kwargs": {"temperature": num_detected, "scale": num_frames, "normalize": True},
    }

    cfg = {"resnet18", "ResNet18_Weights.DEFAULT"}

    tracking_transformer = GlobalTrackingTransformer(
        encoder_model="resnet18",
        encoder_cfg={"weights": "ResNet18_Weights.DEFAULT"},
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )

    asso_preds, embedding = tracking_transformer(instances)

    assert asso_preds[0].size() == (num_detected * num_frames,) * 2
    assert embedding.size() == (num_detected * num_frames, 1, feats)


def test_tracker():
    feats = 512
    num_frames = 5
    num_detected = 2
    img_shape = (1, 128, 128)
    test_frame = 2
    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "frame_id": torch.tensor(i),
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "crops": torch.rand(size=(num_detected, 1, 64, 64)),
                "bboxes": torch.rand(size=(num_detected, 4)),
                "gt_track_ids": torch.arange(num_detected),
                "pred_track_ids": torch.tensor([-1] * num_detected),
            }
        )

    embedding_meta = {
        "embedding_type": "fixed_pos",
        "kwargs": {"temperature": num_detected, "scale": num_frames, "normalize": True},
    }

    tracking_transformer = GlobalTrackingTransformer(
        encoder_model="resnet18",
        encoder_cfg={"weights": "ResNet18_Weights.DEFAULT"},
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
        embedding_meta=embedding_meta,
        return_embedding=False,
    )
    print("testing with no post processing")
    tracking_cfg = {
        "overlap_thresh": 0.01,
        "decay_time": None,
        "iou": None,
        "max_center_dist": None,
    }

    tracker = Tracker(model=tracking_transformer, **tracking_cfg)

    instances_pred = tracker(instances)

    asso_equals = (
        instances_pred[test_frame]["decay_time_traj_score"].to_numpy()
        == instances_pred[test_frame]["with_iou_traj_score"].to_numpy()
    ).all()
    assert asso_equals

    asso_equals = (
        instances_pred[test_frame]["with_iou_traj_score"].to_numpy()
        == instances_pred[test_frame]["max_center_dist_traj_score"].to_numpy()
    ).all()
    assert asso_equals
