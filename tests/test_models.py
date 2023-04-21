import torch
from biogtr.models.attention_head import MLP, ATTWeightHead
from biogtr.models.visual_encoder import VisualEncoder
from biogtr.models.embedding import Embedding

# todo: add named tensor tests
# todo: add fixtures


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
        boxes, features=d_model, temperature=objects, normalize=True, scale=10
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

    _ = emb._sine_box_embedding(boxes)
    sine_no_args_params = emb._get_parameter_values()

    sine_args = {
        "temperature": objects,
        "scale": frames,
        "normalize": True,
    }

    _ = emb._sine_box_embedding(boxes, **sine_args)
    sine_with_args_params = emb._get_parameter_values()

    assert sine_no_args_params["temperature"] != sine_with_args_params["temperature"]

    assert sine_no_args_params["scale"] != sine_with_args_params["scale"]

    assert sine_no_args_params["normalize"] != sine_with_args_params["normalize"]

    # learned pos embedding

    _ = emb._learned_pos_embedding(boxes)
    lp_no_args_params = emb._get_parameter_values()

    lp_args = {"learn_pos_emb_num": 100, "over_boxes": False}

    _ = emb._learned_pos_embedding(boxes, **lp_args)
    lp_with_args_params = emb._get_parameter_values()

    assert (
        lp_no_args_params["learn_pos_emb_num"]
        != lp_with_args_params["learn_pos_emb_num"]
    )

    assert lp_no_args_params["over_boxes"] != lp_with_args_params["over_boxes"]

    # learned temp embedding

    _ = emb._learned_temp_embedding(times)
    lt_no_args_params = emb._get_parameter_values()

    lt_args = {"learn_temp_emb_num": 100}

    _ = emb._learned_temp_embedding(times, **lt_args)
    lt_with_args_params = emb._get_parameter_values()

    assert (
        lt_no_args_params["learn_temp_emb_num"]
        != lt_with_args_params["learn_temp_emb_num"]
    )
