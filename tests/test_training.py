import torch
from biogtr.training.losses import AssoLoss

# todo: add named tensor tests
# todo: add fixtures

torch.set_default_device("cpu")


def test_asso_loss():
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "gt_track_ids": torch.arange(num_detected),
                "bboxes": torch.rand(size=(num_detected, 4)),
            }
        )

    asso_loss = AssoLoss(neg_unmatched=True, asso_weight=10.0)

    asso_preds = torch.rand(size=(1, 100, 100))

    loss = asso_loss(asso_preds, instances)

    assert len(loss.size()) == 0
    assert type(loss.item()) == float
