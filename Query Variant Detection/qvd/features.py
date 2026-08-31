from __future__ import annotations

import numpy as np


def build_feature_vector(embeddings: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Build [QQ, four QD blocks, DD] features in the original notebook order."""
    expected = 2 + 2 * top_k
    if embeddings.shape[0] != expected:
        raise ValueError(f"Expected {expected} embeddings, got {embeddings.shape[0]}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length embedding cannot be cosine-normalized")
    vectors = embeddings / norms

    q1 = vectors[0]
    d1 = vectors[1 : 1 + top_k]
    q2 = vectors[1 + top_k]
    d2 = vectors[2 + top_k :]

    qq = np.array([q1 @ q2])
    qd = np.concatenate([q1 @ d1.T, q1 @ d2.T, q2 @ d1.T, q2 @ d2.T])
    dd = (d1 @ d2.T).reshape(-1)
    return np.concatenate([qq, qd, dd]).astype(np.float32)


def select_features(features: np.ndarray, name: str, top_k: int = 10) -> np.ndarray:
    qd_end = 1 + 4 * top_k
    if name == "qq":
        return features[:, :1]
    if name == "qq_qd":
        return features[:, :qd_end]
    if name == "qq_qd_dd":
        return features
    raise ValueError(f"Unknown feature set: {name}")


def document_text(result: dict, field: str) -> str:
    if field == "text":
        return result.get("text") or result.get("snippet") or result.get("title") or ""
    return result.get(field) or ""


def texts_for_record(record: dict, top_k: int, document_field: str) -> list[str]:
    left = record.get("q1_results", [])[:top_k]
    right = record.get("q2_results", [])[:top_k]
    if len(left) < top_k or len(right) < top_k:
        raise ValueError("Record has fewer than top_k results")
    return [
        record["q1"],
        *(document_text(item, document_field) for item in left),
        record["q2"],
        *(document_text(item, document_field) for item in right),
    ]

