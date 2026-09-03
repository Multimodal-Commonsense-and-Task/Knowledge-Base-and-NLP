"""Dataset, augmentation, and the toy data generator.

One datum is a hemisection: a fundus half, its matched blood-vessel half, and a label
for whether BRVO later occurred in that half.

The study cohort is 27 BRVO-affected hemisections against 81 unaffected ones
(27 counter halves of the same eye and 54 contralateral halves). Patient images are
not redistributable, so `build_toy_dataset` synthesizes a set with the same shape and
class balance.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import AugmentConfig, IMAGE_SIZE


@dataclass
class Hemisection:
    sample_id: str
    eye_id: str
    patient_id: str
    side: str          # "upper" | "lower"
    origin: str        # "affected" | "counter" | "contralateral"
    label: int         # 1 = BRVO occurred later in this half
    fundus: torch.Tensor    # (3, H, W) in [0, 1]
    vessel: torch.Tensor    # (3, H, W) in [0, 1]


class HemisectionDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[Hemisection], augment: AugmentConfig | None = None):
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        f, v = it.fundus, it.vessel
        if self.augment is not None:
            f, v = apply_augment(f, v, self.augment)
        return f, v, it.label


def apply_augment(fundus: torch.Tensor, vessel: torch.Tensor, cfg: AugmentConfig):
    """Augmentations from the paper, applied identically to both modalities.

    The two images are a matched pair, so any geometric transform has to be shared;
    photometric noise is applied to the fundus only, since the vessel map is a mask.
    """
    if cfg.horizontal_flip and random.random() < 0.5:
        fundus, vessel = fundus.flip(-1), vessel.flip(-1)

    angle = random.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)
    scale = random.uniform(*cfg.scale)
    tx = random.uniform(-cfg.translate, cfg.translate)
    fundus, vessel = _affine(fundus, angle, scale, tx), _affine(vessel, angle, scale, tx)

    if cfg.brightness:
        factor = 1.0 + random.uniform(-cfg.brightness, cfg.brightness)
        fundus = (fundus * factor).clamp(0, 1)
    if cfg.blur and random.random() < 0.3:
        fundus = _box_blur(fundus)
    if cfg.noise_std:
        fundus = (fundus + torch.randn_like(fundus) * cfg.noise_std).clamp(0, 1)
    return fundus, vessel


def _affine(x: torch.Tensor, angle_deg: float, scale: float, tx: float) -> torch.Tensor:
    """Shared rotation + scale + horizontal shift via a sampling grid."""
    theta_r = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta_r) / scale, np.sin(theta_r) / scale
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, 0.0]], dtype=torch.float32)
    grid = F.affine_grid(theta.unsqueeze(0), [1, *x.shape], align_corners=False)
    return F.grid_sample(x.unsqueeze(0), grid, align_corners=False,
                         padding_mode="zeros")[0]


def _box_blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    c = x.shape[0]
    kernel = torch.ones(c, 1, k, k) / (k * k)
    return F.conv2d(x.unsqueeze(0), kernel, padding=k // 2, groups=c)[0]


# --------------------------------------------------------------------- toy data

def _synthetic_fundus(rng: np.random.Generator, size: int, with_crossing: bool):
    """A crude fundus stand-in: a warm disc, a few vessel arcs, an optic disc blob.

    When with_crossing is True an extra artery crosses a vein at a shallow angle --
    the arteriovenous crossing the paper's Grad-CAM maps focus on. That is the only
    signal separating the two classes in the toy set.
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fundus = np.zeros((3, h, w), dtype=np.float32)
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    retina = np.clip(1.0 - r / (0.55 * h), 0, 1)
    fundus[0] = retina * rng.uniform(0.75, 0.95)
    fundus[1] = retina * rng.uniform(0.40, 0.55)
    fundus[2] = retina * rng.uniform(0.20, 0.30)

    vessel = np.zeros((h, w), dtype=np.float32)
    disc_y = int(h * rng.uniform(0.42, 0.58))
    disc_x = int(w * rng.uniform(0.18, 0.28))
    disc = ((yy - disc_y) ** 2 + (xx - disc_x) ** 2) < (0.06 * h) ** 2
    fundus[:, disc] = np.array([[1.0], [0.9], [0.6]])

    for _ in range(rng.integers(4, 7)):
        curve = rng.uniform(-0.5, 0.5)
        slope = rng.uniform(-1.0, 1.0)
        t = np.linspace(0, 1, 400)
        px = disc_x + t * (w - disc_x)
        py = disc_y + slope * t * h * 0.35 + curve * (t ** 2) * h * 0.3
        _draw(vessel, px, py, h, w, width=rng.integers(1, 3))

    if with_crossing:
        t = np.linspace(0, 1, 400)
        px = disc_x + t * (w - disc_x)
        py = disc_y + 0.22 * t * h + 0.02 * h * np.sin(6 * t)
        _draw(vessel, px, py, h, w, width=3)
        py2 = disc_y + 0.30 * t * h - 0.04 * h
        _draw(vessel, px, py2, h, w, width=2)

    fundus = np.clip(fundus - vessel[None] * 0.45, 0, 1)
    disc_mask = disc.astype(np.float32)
    vessel_rgb = np.repeat(vessel[None], 3, axis=0)
    return (torch.from_numpy(fundus), torch.from_numpy(vessel_rgb),
            torch.from_numpy(disc_mask))


def _draw(canvas, px, py, h, w, width=1):
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            xi = np.clip((px + dx).astype(int), 0, w - 1)
            yi = np.clip((py + dy).astype(int), 0, h - 1)
            canvas[yi, xi] = 1.0


def build_toy_dataset(out_dir: Path | str, n_affected: int = 27, size: int = IMAGE_SIZE,
                      seed: int = 42) -> Path:
    """Synthesize a dataset with the paper's shape: 27 affected, 81 unaffected."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items, meta = [], []
    for i in range(n_affected):
        pid, eid = f"p{i:03d}", f"e{i:03d}"
        # affected half (label 1) plus its counter half and two contralateral halves
        for origin, label, side in (("affected", 1, "upper"), ("counter", 0, "lower"),
                                    ("contralateral", 0, "upper"),
                                    ("contralateral", 0, "lower")):
            f, v, _ = _synthetic_fundus(rng, size, with_crossing=bool(label))
            sid = f"{eid}_{origin}_{side}"
            items.append((sid, eid, pid, side, origin, label, f, v))
            meta.append({"sample_id": sid, "eye_id": eid, "patient_id": pid,
                         "side": side, "origin": origin, "label": label})

    torch.save({"fundus": torch.stack([x[6] for x in items]),
                "vessel": torch.stack([x[7] for x in items]),
                "label": torch.tensor([x[5] for x in items]),
                "meta": meta}, out_dir / "hemisections.pt")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_pos = sum(m["label"] for m in meta)
    print(f"[toy] hemisections={len(meta)}  BRVO={n_pos}  non-BRVO={len(meta) - n_pos} "
          f"-> {out_dir}")
    return out_dir


def load_hemisections(path: Path | str) -> list[Hemisection]:
    path = Path(path)
    if path.is_dir():
        path = path / "hemisections.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `main.py toy` first.")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for i, m in enumerate(blob["meta"]):
        out.append(Hemisection(sample_id=m["sample_id"], eye_id=m["eye_id"],
                               patient_id=m["patient_id"], side=m["side"],
                               origin=m["origin"], label=int(blob["label"][i]),
                               fundus=blob["fundus"][i], vessel=blob["vessel"][i]))
    return out
