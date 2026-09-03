#!/usr/bin/env python3
"""BRVO prediction from pre-onset fundus hemisection images.

Subcommands follow the paper's pipeline:

    toy        generate a synthetic cohort (no patient data required)
    segment    train the U-Net for optic disc / blood-vessel segmentation
    hemisect   split fundus images into upper and lower halves at the optic disc
    train      train a model on every sample (fundus | vessel | multimodal)
    loocv      leave-one-out cross-validation; AUC, accuracy, sensitivity, specificity
    gradcam    Grad-CAM attention map for one hemisection

Example:
    python main.py toy
    python main.py train --kind fundus --tiny --epochs 2
    python main.py train --kind vessel --tiny --epochs 2
    python main.py loocv --kind multimodal --tiny --epochs 1 --folds 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import config as C
from src.config import BRVOConfig


def _cfg(args) -> BRVOConfig:
    cfg = BRVOConfig()
    cfg.seed = args.seed
    cfg.model.tiny = getattr(args, "tiny", False)
    if getattr(args, "no_pretrained", False):
        cfg.model.pretrained = False
    if getattr(args, "epochs", None):
        cfg.train.epochs = args.epochs
    if getattr(args, "batch_size", None):
        cfg.train.batch_size = args.batch_size
    if getattr(args, "lr", None):
        cfg.train.lr = args.lr
    if getattr(args, "device", None):
        cfg.train.device = args.device
    if getattr(args, "no_augment", False):
        cfg.augment = None
    return cfg


def _dump(obj, path=None):
    print(json.dumps(obj, indent=2))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"[saved] {path}")


def cmd_toy(args) -> int:
    from src.data import build_toy_dataset
    build_toy_dataset(args.data_dir, n_affected=args.n_affected, seed=args.seed)
    return 0


def cmd_segment(args) -> int:
    """Train the U-Net on the toy fundus images, using the vessel maps as masks."""
    import torch
    from src.data import load_hemisections
    from src.segmentation import train_unet
    from src.train import resolve_device

    cfg = _cfg(args)
    items = load_hemisections(args.data_dir)[: args.limit]
    images = torch.stack([it.fundus for it in items])
    masks = torch.stack([it.vessel[:1] for it in items])
    train_unet(images, masks, cfg.segmentation, device=resolve_device(args.device),
               out_path=Path(args.out or C.CKPT_DIR) / "unet.pt")
    return 0


def cmd_hemisect(args) -> int:
    """Demonstrate the hemisection split on one synthetic fundus image."""
    import numpy as np
    from src.data import _synthetic_fundus
    from src.hemisection import hemisect_pair

    rng = np.random.default_rng(args.seed)
    fundus, vessel, disc = _synthetic_fundus(rng, C.IMAGE_SIZE * 2, with_crossing=True)
    out = hemisect_pair(fundus, vessel, disc, size=C.IMAGE_SIZE)
    print(json.dumps({
        "optic_disc_center_y": out["center_y"],
        "upper_fundus": list(out["upper"]["fundus"].shape),
        "upper_vessel": list(out["upper"]["vessel"].shape),
        "lower_fundus": list(out["lower"]["fundus"].shape),
        "lower_vessel": list(out["lower"]["vessel"].shape),
    }, indent=2))
    return 0


def cmd_train(args) -> int:
    from src.data import load_hemisections
    from src.train import build_multimodal_from_checkpoints, train_full

    cfg = _cfg(args)
    items = load_hemisections(args.data_dir)
    out_dir = Path(args.out or C.CKPT_DIR)
    if args.kind == "multimodal" and args.from_unimodal:
        model = build_multimodal_from_checkpoints(cfg, out_dir)
        print("[train] multimodal initialized from the trained unimodal backbones")
        import torch
        torch.save({"state_dict": model.state_dict(), "kind": "multimodal",
                    "config": cfg.to_dict()}, out_dir / "multimodal.pt")
        print(f"[train] saved -> {out_dir / 'multimodal.pt'}")
        return 0
    train_full(items, args.kind, cfg, out_dir=out_dir)
    return 0


def cmd_loocv(args) -> int:
    from src.data import load_hemisections
    from src.train import leave_one_out

    cfg = _cfg(args)
    items = load_hemisections(args.data_dir)
    res = leave_one_out(items, args.kind, cfg,
                        out_dir=Path(args.out or C.CKPT_DIR), limit=args.folds)
    _dump(res)
    return 0


def cmd_gradcam(args) -> int:
    import torch
    from src.data import load_hemisections
    from src.gradcam import grad_cam, save_overlay
    from src.models import build_model
    from src.train import build_multimodal_from_checkpoints, resolve_device

    cfg = _cfg(args)
    items = load_hemisections(args.data_dir)
    item = items[args.index]
    ckpt_dir = Path(args.out or C.CKPT_DIR)
    ckpt = ckpt_dir / f"{args.kind}.pt"

    if ckpt.exists():
        model = build_model(args.kind, cfg.model)
        model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                         weights_only=False)["state_dict"])
        print(f"[gradcam] checkpoint <- {ckpt}")
    elif args.kind == "multimodal":
        model = build_multimodal_from_checkpoints(cfg, ckpt_dir)
    else:
        model = build_model(args.kind, cfg.model)
        print(f"[gradcam] no checkpoint at {ckpt}; using an untrained model")

    cam = grad_cam(model, item.fundus, item.vessel, args.kind,
                   device=resolve_device(args.device))
    save_overlay(item.fundus, cam,
                 Path(args.figdir) / f"gradcam_{args.kind}_{item.sample_id}.png")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, model_args: bool = True):
        sp.add_argument("--data-dir", default=str(C.DATA_DIR / "toy"))
        sp.add_argument("--seed", type=int, default=C.SEED)
        sp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        sp.add_argument("--threads", type=int, default=4,
                        help="cap on CPU threads; much slower if left unset on many-core machines")
        sp.add_argument("--out", default=None, help="checkpoint directory")
        if model_args:
            sp.add_argument("--tiny", action="store_true",
                            help="use a small randomly initialized CNN instead of EfficientNetB0")
            sp.add_argument("--no-pretrained", action="store_true",
                            help="EfficientNetB0 without ImageNet weights")
            sp.add_argument("--epochs", type=int, default=None)
            sp.add_argument("--batch-size", type=int, default=None)
            sp.add_argument("--lr", type=float, default=None)
            sp.add_argument("--no-augment", action="store_true")

    sp = sub.add_parser("toy", help="generate a synthetic cohort")
    common(sp, model_args=False)
    sp.add_argument("--n-affected", type=int, default=27)
    sp.set_defaults(func=cmd_toy)

    sp = sub.add_parser("segment", help="train the U-Net segmentation model")
    common(sp)
    sp.add_argument("--limit", type=int, default=16)
    sp.set_defaults(func=cmd_segment)

    sp = sub.add_parser("hemisect", help="demonstrate the hemisection split")
    common(sp, model_args=False)
    sp.set_defaults(func=cmd_hemisect)

    sp = sub.add_parser("train", help="train on every sample")
    common(sp)
    sp.add_argument("--kind", default="fundus", choices=["fundus", "vessel", "multimodal"])
    sp.add_argument("--from-unimodal", action="store_true",
                    help="assemble the multimodal model from the trained unimodal backbones")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("loocv", help="leave-one-out cross-validation")
    common(sp)
    sp.add_argument("--kind", default="multimodal",
                    choices=["fundus", "vessel", "multimodal"])
    sp.add_argument("--folds", type=int, default=None,
                    help="truncate the number of folds (smoke tests only; not real LOOCV)")
    sp.set_defaults(func=cmd_loocv)

    sp = sub.add_parser("gradcam", help="Grad-CAM attention map")
    common(sp)
    sp.add_argument("--kind", default="fundus", choices=["fundus", "vessel", "multimodal"])
    sp.add_argument("--index", type=int, default=0)
    sp.add_argument("--figdir", default=str(C.FIG_DIR))
    sp.set_defaults(func=cmd_gradcam)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if getattr(args, "threads", None):
        from src.train import cap_threads
        cap_threads(args.threads)
    sys.exit(args.func(args))
