#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qvd.features import select_features
from qvd.model import EnvironmentVerifier


def load_features(path: Path, feature_set: str) -> tuple[np.ndarray, np.ndarray, int]:
    archive = np.load(path)
    top_k = int(archive["top_k"])
    features = select_features(archive["features"], feature_set, top_k)
    return features.astype(np.float32), archive["labels"].astype(np.float32), top_k


def metrics(model, loader, device) -> dict[str, float]:
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            logits = model(inputs.to(device))
            predictions.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int))
            labels.extend(batch_labels.numpy().astype(int))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-encoder MLP verifier.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("qq", "qq_qd", "qq_qd_dd"), default="qq_qd_dd")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_x, train_y, train_k = load_features(args.train, args.feature_set)
    test_x, test_y, test_k = load_features(args.test, args.feature_set)
    if train_k != test_k or train_x.shape[1] != test_x.shape[1]:
        raise SystemExit("Train and test feature layouts differ")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnvironmentVerifier(train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.shape[0]
        scheduler.step()
        print(f"epoch={epoch + 1} loss={total_loss / len(train_loader.dataset):.6f}")

    result = metrics(model, test_loader, device)
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train_x.shape[1],
            "feature_set": args.feature_set,
            "top_k": train_k,
            "test_metrics": result,
            "seed": args.seed,
        },
        args.output,
    )


if __name__ == "__main__":
    main()

