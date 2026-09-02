#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from qvd.features import build_feature_vector, texts_for_record
from qvd.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QQ/QD/DD cosine-similarity features.")
    parser.add_argument("--input", type=Path, required=True, help="Search-result JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Compressed NPZ")
    parser.add_argument("--encoder", default="Alibaba-NLP/gte-base-en-v1.5")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--document-field", choices=("text", "snippet", "title"), default="text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    model = SentenceTransformer(args.encoder, trust_remote_code=args.trust_remote_code)
    features, labels, ids = [], [], []
    skipped = 0
    for record in tqdm(read_jsonl(args.input)):
        try:
            texts = texts_for_record(record, args.top_k, args.document_field)
        except ValueError:
            skipped += 1
            continue
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        features.append(build_feature_vector(embeddings, args.top_k))
        labels.append(int(record["label"]))
        ids.append(str(record["id"]))

    if not features:
        raise SystemExit("No records had enough retrieval results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(features),
        labels=np.asarray(labels, dtype=np.int64),
        ids=np.asarray(ids),
        top_k=np.asarray(args.top_k),
        encoder=np.asarray(args.encoder),
    )
    print(f"wrote {len(features)} examples to {args.output}; skipped {skipped}")


if __name__ == "__main__":
    main()

