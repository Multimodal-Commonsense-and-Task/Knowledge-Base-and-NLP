"""Corpus encoding, search, and retrieval metrics.

Metrics (Tables 2 and 4 of the paper)
  Top-k hit accuracy : is a passage containing the answer string in the top k (single-hop)
  MRR                : mean reciprocal rank of the first answer-bearing passage
  R@k                : fraction of questions where *all* annotated supporting contexts
                       appear in the top k (multi-hop)
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .data import Passage, QAExample
from .modeling import DualEncoder, passage_text


@torch.no_grad()
def encode_corpus(model: DualEncoder, corpus: list[Passage], batch_size: int = 32,
                  device: str = "cpu") -> torch.Tensor:
    model.eval().to(device)
    embs = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="encode corpus"):
        chunk = corpus[i:i + batch_size]
        embs.append(model.encode_passages([passage_text(p) for p in chunk]).cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def search(model: DualEncoder, questions: list[str], corpus_emb: torch.Tensor,
           top_k: int = 20, batch_size: int = 32, device: str = "cpu"):
    """Return (scores, indices), both shaped (len(questions), top_k)."""
    model.eval().to(device)
    k = min(top_k, corpus_emb.size(0))
    all_s, all_i = [], []
    for i in range(0, len(questions), batch_size):
        q = model.encode_questions(questions[i:i + batch_size]).cpu()
        scores = q @ corpus_emb.t()
        s, idx = scores.topk(k, dim=-1)
        all_s.append(s)
        all_i.append(idx)
    return torch.cat(all_s), torch.cat(all_i)


def _has_answer(passage: Passage, answers: list[str]) -> bool:
    text = passage.text.lower()
    return any(a.lower() in text for a in answers if a)


def evaluate_retrieval(model: DualEncoder, examples: list[QAExample],
                       corpus: list[Passage], ks=(1, 5, 20, 100),
                       device: str = "cpu", batch_size: int = 32) -> dict:
    corpus_emb = encode_corpus(model, corpus, batch_size=batch_size, device=device)
    max_k = min(max(ks), len(corpus))
    _, idx = search(model, [e.question for e in examples], corpus_emb,
                    top_k=max_k, batch_size=batch_size, device=device)

    pid_at = [[corpus[j].pid for j in row.tolist()] for row in idx]
    hits = {k: 0 for k in ks}
    recall = {k: 0 for k in ks}
    rr_sum = 0.0

    for row_i, ex in enumerate(examples):
        ranked = [corpus[j] for j in idx[row_i].tolist()]
        answer_hit = [_has_answer(p, ex.answers) for p in ranked]

        first = next((r + 1 for r, h in enumerate(answer_hit) if h), None)
        rr_sum += 1.0 / first if first else 0.0

        for k in ks:
            kk = min(k, len(ranked))
            if any(answer_hit[:kk]):
                hits[k] += 1
            if ex.supporting_pids:
                # Multi-hop R@k -- every supporting context must be present.
                if set(ex.supporting_pids).issubset(set(pid_at[row_i][:kk])):
                    recall[k] += 1

    n = len(examples)
    out = {f"top{k}_accuracy": hits[k] / n for k in ks}
    out["mrr"] = rr_sum / n
    if any(e.supporting_pids for e in examples):
        out.update({f"R@{k}": recall[k] / n for k in ks})
    out["n_questions"] = n
    out["n_passages"] = len(corpus)
    return out
