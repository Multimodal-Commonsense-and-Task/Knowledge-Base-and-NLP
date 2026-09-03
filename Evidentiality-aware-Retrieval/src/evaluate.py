"""§5 Analysis — Answer-Awareness score 와 robustness 시뮬레이션.

AA score (Eq.9)
    정답 span 을 지운 p' 를 만들고, 모델이 p+ 를 p' 보다 높게 매기는 비율을 잰다.

        AA = 1 - (1/T) Σ_i 1[ <q_i, p+_i> ≤ <q_i, p'_i> ]

    DPR 은 이 값이 이론적 상한(1.0)에 크게 못 미친다는 것이 논문의 관찰이다.

Robustness
    합성 distractor 를 코퍼스에 섞어 넣고 검색 성능이 얼마나 떨어지는지 본다.
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
    """Eq.9. 정답이 본문에 실제로 등장하는 예제만 대상으로 한다."""
    model.eval().to(device)

    triplets = []
    for ex in examples:
        masked = mask_answer(ex.positive_ctx.text, ex.answers)
        if masked == ex.positive_ctx.text:
            continue                       # 정답 문자열이 본문에 없어 p' 를 만들 수 없다
        triplets.append((ex, Passage(pid=f"{ex.positive_ctx.pid}::masked",
                                     title=ex.positive_ctx.title, text=masked)))
    if not triplets:
        return {"aa_score": float("nan"), "n_triplets": 0,
                "note": "정답 span 이 본문에 등장하는 예제가 없다"}

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
    """augment 단계에서 만든 p* 를 코퍼스 오염원으로 재사용한다."""
    return [ex.distractor_ctx for ex in examples if ex.distractor_ctx is not None]


def robustness_test(model: DualEncoder, examples: list[QAExample],
                    corpus: list[Passage], ks=(1, 5, 20), device: str = "cpu") -> dict:
    """distractor 를 코퍼스에 주입하기 전/후의 검색 성능을 비교한다."""
    injected = synthesize_corpus_distractors(examples)
    before = evaluate_retrieval(model, examples, corpus, ks=ks, device=device)
    after = evaluate_retrieval(model, examples, corpus + injected, ks=ks, device=device)
    delta = {k: after[k] - before[k] for k in before
             if isinstance(before[k], float) and k in after}
    return {"n_injected": len(injected), "before": before, "after": after, "delta": delta}
