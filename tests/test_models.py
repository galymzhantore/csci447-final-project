import torch

from models.mlp import build_mlp


def test_mlp_forward_pass():
    model = build_mlp("mlp_small", input_dim=20 * 10, num_classes=3, cfg={"mlp_small": {"hidden_sizes": [16, 8], "dropout": 0.1}, "mlp_activation": "relu"})
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert out.shape == (4, 3)
