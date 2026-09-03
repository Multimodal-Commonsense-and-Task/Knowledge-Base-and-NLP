"""Section 5 analysis -- Answer-Awareness score and the robustness simulation.

AA score (Eq.9)
    Build p' by deleting the answer span, then measure how often the model still
    ranks p+ above p':

        AA = 1 - (1/T) sum_i 1[ <q_i, p+_i> <= <q_i, p'_i> ]

    The paper's observation is that DPR falls well short of the theoretical upper
    bound of 1.0 here.

Robustness
    Inject synthesized distractors into the corpus and measure how much retrieval
    performance degrades.
"""
from __future__ import annotations

import re
from collections import defaultdict

import torch

from .data import Passage, QAExample, mask_answer
from .modeling import DualEncoder, passage_text
from .retrieval import evaluate_retrieval

_WH = ("who", "when", "what", "where", "how", "which", "why")


def question_type(question: str) -> str:
    first = re.findall(r"[a-z]+", question.lower())
    for tok in first[:3]:
        if tok in _WH:
            return tok
    return "other"


@torch.no_grad()
def answer_awareness(model: DualEncoder, examples: list[QAExample],
                     device: str = "cpu", batch_size: int = 16,
                     by_question_type: bool = True) -> dict:
    """Eq.9. Only examples whose answer actually occurs in the passage are used."""
    model.eval().to(device)

    triplets = []
    for ex in examples:
        masked = mask_answer(ex.positive_ctx.text, ex.answers)
        if masked == ex.positive_ctx.text:
            continue                       # answer string absent, so p' cannot be built
        triplets.append((ex, Passage(pid=f"{ex.positive_ctx.pid}::masked",
                                     title=ex.positive_ctx.title, text=masked)))
    if not triplets:
        return {"aa_score": float("nan"), "n_triplets": 0,
                "note": "no example has its answer span present in the passage"}

    wins, per_type = 0, defaultdict(lambda: [0, 0])
    for i in range(0, len(triplets), batch_size):
        chunk = triplets[i:i + batch_size]
        q = model.encode_questions([ex.question for ex, _ in chunk])
        p_pos = model.encode_passages([passage_text(ex.positive_ctx) for ex, _ in chunk])
        p_msk = model.encode_passages([passage_text(m) for _, m in chunk])
        s_pos = (q * p_pos).sum(-1)
        s_msk = (q * p_msk).sum(-1)
        for (ex, _), a, b in zip(chunk, s_pos.tolist(), s_msk.tolist()):
            win = a > b
            wins += int(win)
            t = question_type(ex.question)
            per_type[t][0] += int(win)
            per_type[t][1] += 1

    out = {"aa_score": wins / len(triplets), "n_triplets": len(triplets)}
    if by_question_type:
        out["by_question_type"] = {
            t: {"aa_score": c / n, "n": n} for t, (c, n) in sorted(per_type.items())}
    return out


def synthesize_corpus_distractors(examples: list[QAExample]) -> list[Passage]:
    """Reuse the p* produced by the augment step as corpus contaminants."""
    return [ex.distractor_ctx for ex in examples if ex.distractor_ctx is not None]


def robustness_test(model: DualEncoder, examples: list[QAExample],
                    corpus: list[Passage], ks=(1, 5, 20), device: str = "cpu") -> dict:
    """Compare retrieval performance before and after injecting distractors."""
    injected = synthesize_corpus_distractors(examples)
    before = evaluate_retrieval(model, examples, corpus, ks=ks, device=device)
    after = evaluate_retrieval(model, examples, corpus + injected, ks=ks, device=device)
    delta = {k: after[k] - before[k] for k in before
             if isinstance(before[k], float) and k in after}
    return {"n_injected": len(injected), "before": before, "after": after, "delta": delta}
