"""Split a fundus image into upper and lower hemisections.

The paper divides each fundus photograph along a horizontal line through the centre
of the optic disc, and takes the y-coordinate of that centre from the optic-disc
segmentation. Each half is resized to 224 x 224 and becomes one training datum,
labelled by whether BRVO later occurred in that half.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def optic_disc_center_y(disc_mask: torch.Tensor) -> int:
    """y-coordinate of the optic disc centroid, from a (H,W) binary mask.

    Falls back to the middle of the image when the mask is empty, so a failed
    segmentation degrades to a plain horizontal split instead of crashing.
    """
    h = disc_mask.shape[-2]
    ys, _ = torch.nonzero(disc_mask > 0.5, as_tuple=True)
    if len(ys) == 0:
        return h // 2
    return int(ys.float().mean().round().item())


def split_hemisections(image: torch.Tensor, center_y: int, size: int = 224
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (upper, lower), each resized to (C, size, size).

    The lower half is flipped vertically so that both halves present the vascular
    arcade in the same orientation to the network.
    """
    h = image.shape[-2]
    center_y = max(1, min(h - 1, center_y))
    upper = image[..., :center_y, :]
    lower = image[..., center_y:, :].flip(-2)
    return _resize(upper, size), _resize(lower, size)


def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(0), size=(size, size),
                         mode="bilinear", align_corners=False)[0]


def hemisect_pair(fundus: torch.Tensor, vessels: torch.Tensor, disc_mask: torch.Tensor,
                  size: int = 224) -> dict:
    """Produce matched fundus and blood-vessel hemisections for one eye.

    Returns {"upper": {"fundus", "vessel"}, "lower": {...}, "center_y": int}.
    The two modalities are split at the same y, which is what makes them a matched
    pair for the multimodal model.
    """
    cy = optic_disc_center_y(disc_mask)
    fu, fl = split_hemisections(fundus, cy, size)
    vu, vl = split_hemisections(vessels, cy, size)
    return {"upper": {"fundus": fu, "vessel": vu},
            "lower": {"fundus": fl, "vessel": vl},
            "center_y": cy}
