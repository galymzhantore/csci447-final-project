from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass
class MLPConfig:
    input_dim: int
    hidden_sizes: List[int]
    num_classes: int
    dropout: float = 0.2
    activation: str = "relu"


def activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class MLPClassifier(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        layers = []
        in_dim = cfg.input_dim
        act = activation_from_name(cfg.activation)
        for hidden in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act)
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten time dimension
        x = x.flatten(1)
        hidden = self.backbone(x)
        return self.classifier(hidden)


def build_mlp(model_name: str, input_dim: int, num_classes: int, cfg: dict) -> MLPClassifier:
    model_cfg = cfg.get(model_name, cfg.get("mlp_small"))
    hidden_sizes = model_cfg.get("hidden_sizes", [128, 64])
    dropout = model_cfg.get("dropout", 0.2)
    activation = cfg.get("mlp_activation", "relu")
    mlp_cfg = MLPConfig(input_dim=input_dim, hidden_sizes=hidden_sizes, num_classes=num_classes, dropout=dropout, activation=activation)
    return MLPClassifier(mlp_cfg)


__all__ = ["build_mlp", "MLPClassifier", "MLPConfig"]
