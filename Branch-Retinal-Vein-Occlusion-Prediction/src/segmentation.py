"""U-Net for optic disc and retinal blood-vessel segmentation.

The paper trains one segmentation model per task -- blood vessels on DRIVE and FIVES,
optic disc on REFUGE -- and applies them to the study's fundus photographs. Those
datasets are not redistributed here, so `train_unet` runs on whatever image/mask pairs
it is given.

Segmentation quality is reported with IoU and the Dice coefficient.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SegmentationConfig


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """A compact U-Net. Outputs a single-channel logit map."""

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        chs = [cfg.base_channels * (2 ** i) for i in range(cfg.depth + 1)]
        self.downs = nn.ModuleList()
        cin = cfg.in_channels
        for c in chs[:-1]:
            self.downs.append(_block(cin, c))
            cin = c
        self.bottleneck = _block(cin, chs[-1])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.ups.append(_block(c * 2, c))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips)):
            x = up_conv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([skip, x], dim=1))
        return self.head(x)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """2|A n B| / (|A| + |B|) on binarized masks."""
    p = (torch.sigmoid(pred) > 0.5).float()
    t = (target > 0.5).float()
    inter = (p * t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    p = (torch.sigmoid(pred) > 0.5).float()
    t = (target > 0.5).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + eps) / (union + eps))


def train_unet(images: torch.Tensor, masks: torch.Tensor, cfg: SegmentationConfig,
               device: str = "cpu", out_path: Path | str | None = None) -> UNet:
    """Train the U-Net on (N,3,H,W) images and (N,1,H,W) binary masks."""
    model = UNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(images)
    model.train()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        total, d, j = 0.0, 0.0, 0.0
        nb = 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x, y = images[idx].to(device), masks[idx].to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss); d += dice_coefficient(logits, y); j += iou(logits, y)
            nb += 1
        print(f"[seg] epoch {epoch + 1}/{cfg.epochs}  loss={total / nb:.4f}  "
              f"dice={d / nb:.4f}  iou={j / nb:.4f}")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"[seg] saved -> {out_path}")
    return model


@torch.no_grad()
def predict_mask(model: UNet, image: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Binary mask for a single (3,H,W) image."""
    model.eval().to(device)
    logits = model(image.unsqueeze(0).to(device))
    return (torch.sigmoid(logits)[0, 0] > 0.5).float().cpu()
