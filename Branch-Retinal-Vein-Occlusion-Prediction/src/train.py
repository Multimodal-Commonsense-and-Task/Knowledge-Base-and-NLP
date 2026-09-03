"""Leave-one-out cross-validation training.

The cohort is small, so the paper evaluates with leave-one-out cross-validation:
build N models, each excluding one sample, score that held-out sample, and pool the
softmax outputs to draw a single ROC curve.

Optimization follows the paper: SGD with momentum, lr 1e-4, mini-batch 20, 100 epochs,
starting from ImageNet-pretrained EfficientNetB0 weights.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import BRVOConfig
from .data import Hemisection, HemisectionDataset
from .evaluate import summarize
from .models import MultimodalNet, UnimodalNet, build_model


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def cap_threads(n: int | None) -> None:
    """Cap CPU threads; small tensors on a many-core node are dominated by overhead."""
    if n and n > 0:
        torch.set_num_threads(n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _forward(model: nn.Module, fundus, vessel, kind: str):
    if kind == "fundus":
        return model(fundus)
    if kind == "vessel":
        return model(vessel)
    return model(fundus, vessel)


def fit_one(items: list[Hemisection], kind: str, cfg: BRVOConfig,
            device: str, progress: bool = False) -> nn.Module:
    """Train a single model on the given items."""
    model = build_model(kind, cfg.model).to(device)
    loader = DataLoader(HemisectionDataset(items, cfg.augment),
                        batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.train.num_workers)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.train.lr,
                          momentum=cfg.train.momentum,
                          weight_decay=cfg.train.weight_decay)
    crit = nn.CrossEntropyLoss()
    model.train()
    epochs = range(cfg.train.epochs)
    if progress:
        epochs = tqdm(epochs, desc=f"train:{kind}", leave=False)
    for _ in epochs:
        for fundus, vessel, label in loader:
            fundus, vessel = fundus.to(device), vessel.to(device)
            label = label.to(device)
            logits = _forward(model, fundus, vessel, kind)
            loss = crit(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_one(model: nn.Module, item: Hemisection, kind: str, device: str) -> float:
    """Softmax probability of the BRVO class for a single hemisection."""
    model.eval()
    f = item.fundus.unsqueeze(0).to(device)
    v = item.vessel.unsqueeze(0).to(device)
    logits = _forward(model, f, v, kind)
    return float(torch.softmax(logits, dim=-1)[0, 1])


def leave_one_out(items: list[Hemisection], kind: str, cfg: BRVOConfig,
                  out_dir: Path | str | None = None, limit: int | None = None) -> dict:
    """Run leave-one-out cross-validation and pool the held-out scores.

    `limit` truncates the number of folds. The full protocol is one fold per sample;
    truncating is only useful for smoke tests, and the result is no longer LOOCV.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    n = len(items) if limit is None else min(limit, len(items))
    print(f"[loocv] kind={kind} device={device} folds={n}/{len(items)} "
          f"epochs={cfg.train.epochs}")
    if limit is not None and limit < len(items):
        print(f"[loocv] NOTE: only {limit} folds requested -- this is not full LOOCV")

    y_true, y_score = [], []
    for i in tqdm(range(n), desc=f"loocv:{kind}"):
        train_items = items[:i] + items[i + 1:]
        model = fit_one(train_items, kind, cfg, device)
        y_score.append(predict_one(model, items[i], kind, device))
        y_true.append(items[i].label)

    res = summarize(y_true, y_score, seed=cfg.seed)
    res["kind"] = kind
    res["folds"] = n
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"loocv_{kind}.json", "w") as f:
            json.dump({**res, "y_true": y_true, "y_score": y_score}, f, indent=2)
        print(f"[loocv] saved -> {out_dir / f'loocv_{kind}.json'}")
    return res


def train_full(items: list[Hemisection], kind: str, cfg: BRVOConfig,
               out_dir: Path | str | None = None) -> nn.Module:
    """Train on every sample and save a checkpoint (used for Grad-CAM)."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    print(f"[train] kind={kind} device={device} n={len(items)} epochs={cfg.train.epochs}")
    model = fit_one(items, kind, cfg, device, progress=True)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "kind": kind,
                    "config": cfg.to_dict()}, out_dir / f"{kind}.pt")
        print(f"[train] saved -> {out_dir / f'{kind}.pt'}")
    return model


def build_multimodal_from_checkpoints(cfg: BRVOConfig, ckpt_dir: Path | str
                                      ) -> MultimodalNet:
    """Assemble the multimodal model from the two trained unimodal checkpoints.

    This mirrors the paper, where the concatenation network is built by replacing the
    last layers of the two already-trained EfficientNetB0 models.
    """
    ckpt_dir = Path(ckpt_dir)
    parts = {}
    for kind in ("fundus", "vessel"):
        p = ckpt_dir / f"{kind}.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Train the unimodal models first "
                f"(`main.py train --kind {kind}`).")
        m = UnimodalNet(cfg.model)
        m.load_state_dict(torch.load(p, map_location="cpu",
                                     weights_only=False)["state_dict"])
        parts[kind] = m
    return MultimodalNet.from_unimodal(parts["fundus"], parts["vessel"], cfg.model)
