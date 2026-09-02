#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from qvd.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL predictions with positive-label F1.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.input) if row.get("decision") is not None]
    if not rows:
        raise SystemExit("No parseable predictions")
    labels = [int(row["label"]) for row in rows]
    decisions = [int(row["decision"]) for row in rows]
    print(
        json.dumps(
            {
                "examples": len(rows),
                "accuracy": accuracy_score(labels, decisions),
                "f1": f1_score(labels, decisions, pos_label=1, zero_division=0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

