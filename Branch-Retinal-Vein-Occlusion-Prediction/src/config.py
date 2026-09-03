"""Hyperparameters and paths.

Values follow the Methods section of the paper:
  - hemisection images resized to 224 x 224
  - EfficientNetB0 backbones, ImageNet pretrained (transfer learning)
  - SGD with momentum, lr 1e-4, mini-batch 20, 100 epochs
  - leave-one-out cross-validation
  - Youden's index for the operating threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figures"

SEED = 42

IMAGE_SIZE = 224
CLASSES = ("no_brvo", "brvo")


@dataclass
class SegmentationConfig:
    """U-Net used to segment the optic disc and the retinal blood vessels.

    The paper trains blood-vessel segmentation on DRIVE and FIVES, and optic-disc
    segmentation on the REFUGE challenge set. Neither is redistributed here, so this
    repository trains on whatever masks it is pointed at (`data/seg/`) and falls back
    to the toy set.
    """
    in_channels: int = 3
    base_channels: int = 16
    depth: int = 3
    epochs: int = 5
    batch_size: int = 4
    lr: float = 1e-3
    image_size: int = 256


@dataclass
class ModelConfig:
    """EfficientNetB0 backbones and the concatenation head."""
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    # The paper replaces the last layer with fully connected layers of
    # 512, 128 and 52 nodes with dropout, then a softmax over the two classes.
    head_dims: tuple[int, int, int] = (512, 128, 52)
    dropout: float = 0.3
    num_classes: int = 2
    # tiny=True swaps in a small randomly initialized CNN so the pipeline runs
    # without downloading ImageNet weights. The numbers it gives are meaningless.
    tiny: bool = False


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 20
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    device: str = "auto"
    num_workers: int = 0
    threads: int = 4


@dataclass
class AugmentConfig:
    """Augmentations listed in the paper."""
    horizontal_flip: bool = True
    rotation_degrees: float = 10.0
    blur: bool = True
    brightness: float = 0.2      # brightening and darkening
    noise_std: float = 0.02      # random noise insertion
    translate: float = 0.1       # horizontal movement cropping
    scale: tuple[float, float] = (0.9, 1.1)   # +/- 10% enlargement and reduction


@dataclass
class BRVOConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    seed: int = SEED

    def to_dict(self) -> dict:
        return asdict(self)
