"""Unimodal EfficientNetB0 classifiers and the BV-enhanced multimodal model.

Two unimodal models are built, one on fundus hemisections and one on blood-vessel
hemisections. The multimodal model concatenates their penultimate features and passes
them through fully connected layers of 512, 128 and 52 nodes with dropout, followed by
a softmax over the two classes (future BRVO occurrence or not).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class TinyBackbone(nn.Module):
    """Small randomly initialized CNN used when tiny=True (no downloads)."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


def build_backbone(cfg: ModelConfig):
    """Return (feature extractor, feature dimension)."""
    if cfg.tiny:
        b = TinyBackbone()
        return b, b.out_dim
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    net = efficientnet_b0(weights=weights)
    dim = net.classifier[1].in_features
    net.classifier = nn.Identity()      # keep features, drop the ImageNet head
    return net, dim


def _head(in_dim: int, cfg: ModelConfig) -> nn.Sequential:
    d1, d2, d3 = cfg.head_dims
    return nn.Sequential(
        nn.Dropout(cfg.dropout), nn.Linear(in_dim, d1), nn.ReLU(inplace=True),
        nn.Dropout(cfg.dropout), nn.Linear(d1, d2), nn.ReLU(inplace=True),
        nn.Dropout(cfg.dropout), nn.Linear(d2, d3), nn.ReLU(inplace=True),
        nn.Linear(d3, cfg.num_classes),
    )


class UnimodalNet(nn.Module):
    """One EfficientNetB0 over a single image domain (fundus or blood vessels)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone, dim = build_backbone(cfg)
        self.head = _head(dim, cfg)
        self.feature_dim = dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class MultimodalNet(nn.Module):
    """BV-enhanced multimodal model: concatenate the two backbones' features.

    The paper builds this from the two already-trained unimodal models, so
    `from_unimodal` reuses their backbones instead of starting from scratch.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fundus_backbone, d1 = build_backbone(cfg)
        self.vessel_backbone, d2 = build_backbone(cfg)
        self.head = _head(d1 + d2, cfg)

    @classmethod
    def from_unimodal(cls, fundus_model: UnimodalNet, vessel_model: UnimodalNet,
                      cfg: ModelConfig) -> "MultimodalNet":
        m = cls(cfg)
        m.fundus_backbone = fundus_model.backbone
        m.vessel_backbone = vessel_model.backbone
        return m

    def forward(self, fundus: torch.Tensor, vessel: torch.Tensor) -> torch.Tensor:
        f = self.fundus_backbone(fundus)
        v = self.vessel_backbone(vessel)
        return self.head(torch.cat([f, v], dim=1))


def build_model(kind: str, cfg: ModelConfig) -> nn.Module:
    if kind in ("fundus", "vessel"):
        return UnimodalNet(cfg)
    if kind == "multimodal":
        return MultimodalNet(cfg)
    raise ValueError(f"unknown model kind: {kind}")
