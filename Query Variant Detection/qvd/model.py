from __future__ import annotations

from torch import nn


class EnvironmentVerifier(nn.Module):
    """MLP head recovered from the final experiment notebook."""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden = [256, 512, 1024, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden:
            layers.extend([nn.Linear(current, width), nn.BatchNorm1d(width), nn.LeakyReLU(0.01)])
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)

