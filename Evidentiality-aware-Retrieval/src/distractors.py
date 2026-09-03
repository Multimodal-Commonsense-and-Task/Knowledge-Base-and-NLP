"""§3.1 Augmenting Distractor Samples — span 제거 + pseudo-evidence 선택.

Step 1. gold evidence passage p+ = [s_l ; s+ ; s_r] 를 n 개 span 으로 나누고,
        각 span 을 하나씩 빼서 후보 p*_i = [s_l ; s_r] 를 n 개 만든다.
Step 2. 생성형 QA 모델 θ 에 (q, p*_i) 를 넣어 정답의 confidence P_θ(a | q, p*_i) 를 잰다.
        confidence 가 가장 낮은 = perplexity 가 가장 높은 후보를 p* 로 고른다.
        confidence 가 급락했다는 건 그 span 이 답에 기여했다는 뜻이다.

논문은 θ 로 UnifiedQA-T5 를 쓴다.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from tqdm import tqdm

from .config import DistractorConfig
from .data import Passage, QAExample, remove_span, split_spans, write_jsonl


class QAScorer:
    """P_θ(a | q, p) 의 perplexity 를 재는 생성형 QA 모델 래퍼."""

    def __init__(self, cfg: DistractorConfig, device: str = "cpu"):
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                  T5Config, T5ForConditionalGeneration)
        self.cfg = cfg
        self.device = device
        if cfg.tiny:
            # 다운로드 없이 돌리기 위한 소형 랜덤 T5. 점수는 의미가 없고 형태만 같다.
            from .modeling import HashTokenizer
            t5cfg = T5Config(vocab_size=4096, d_model=64, d_ff=128, num_layers=2,
                             num_decoder_layers=2, num_heads=2, d_kv=32,
                             decoder_start_token_id=0, pad_token_id=0, eos_token_id=2)
            self.model = T5ForConditionalGeneration(t5cfg)
            self.tokenizer = HashTokenizer(4096)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.qa_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.qa_model_name)
        self.model.to(device).eval()

    @staticmethod
    def _prompt(question: str, passage: str) -> str:
        # UnifiedQA 입력 규약: "question \n context"
        return f"{question.strip()} \\n {passage.strip()}"

    @torch.no_grad()
    def perplexity(self, questions: list[str], passages: list[str],
                   answers: list[str]) -> list[float]:
        """정답 토큰들의 평균 negative log-likelihood 를 지수화한 값."""
        enc = self.tokenizer([self._prompt(q, p) for q, p in zip(questions, passages)],
                             padding=True, truncation=True, max_length=384,
                             return_tensors="pt")
        lab = self.tokenizer(answers, padding=True, truncation=True, max_length=32,
                             return_tensors="pt")
        labels = lab["input_ids"].clone()
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        labels[labels == pad_id] = -100

        enc = {k: v.to(self.device) for k, v in enc.items()}
        labels = labels.to(self.device)
        logits = self.model(**enc, labels=labels).logits

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        mask = labels != -100
        safe = labels.masked_fill(~mask, 0)
        tok_lp = logprobs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp.masked_fill(~mask, 0.0)
        mean_nll = -(tok_lp.sum(-1) / mask.sum(-1).clamp(min=1))
        return [float(math.exp(min(v, 20.0))) for v in mean_nll.tolist()]


def make_candidates(passage: Passage, cfg: DistractorConfig) -> list[tuple[int, str]]:
    """Step 1 — span 하나씩 제거한 후보들. (제거한 span 인덱스, 본문)."""
    spans = split_spans(passage.text, min_chars=cfg.min_span_chars)
    if len(spans) < 2:
        return []                       # 뺄 span 이 없으면 distractor 를 만들 수 없다
    idxs = list(range(len(spans)))[: cfg.max_candidates]
    return [(i, remove_span(spans, i)) for i in idxs]


def augment_example(ex: QAExample, scorer: QAScorer | None,
                    cfg: DistractorConfig) -> QAExample:
    """한 예제에 p* 를 채워 넣는다."""
    cands = make_candidates(ex.positive_ctx, cfg)
    if not cands:
        return ex

    answer = ex.answers[0] if ex.answers else ""
    if scorer is None or not answer:
        # QA 모델 없이 쓰는 폴백: 가장 긴 span 을 제거한 후보 (정보량이 큰 span 이라는 가정).
        chosen_idx, chosen_text = min(cands, key=lambda c: len(c[1]))
        score = float("nan")
    else:
        ppl = scorer.perplexity([ex.question] * len(cands),
                                [c[1] for c in cands],
                                [answer] * len(cands))
        # confidence 최저 = perplexity 최고
        best = max(range(len(cands)), key=lambda k: ppl[k])
        chosen_idx, chosen_text = cands[best]
        score = ppl[best]

    ex.distractor_ctx = Passage(pid=f"{ex.positive_ctx.pid}::distractor",
                                title=ex.positive_ctx.title, text=chosen_text)
    ex.removed_span_idx = chosen_idx      # 분석용 (직렬화에는 포함되지 않는다)
    ex.distractor_ppl = score
    return ex


def augment_dataset(examples: list[QAExample], cfg: DistractorConfig,
                    device: str = "cpu", out_path: Path | str | None = None,
                    use_qa_model: bool = True) -> list[QAExample]:
    scorer = QAScorer(cfg, device=device) if use_qa_model else None
    if scorer is None:
        print("[augment] QA 모델 없이 길이 휴리스틱으로 distractor 를 고른다")
    out = [augment_example(ex, scorer, cfg) for ex in tqdm(examples, desc="augment")]
    n = sum(1 for e in out if e.distractor_ctx)
    print(f"[augment] distractor 생성 {n}/{len(out)}")
    if out_path:
        write_jsonl([e.to_dict() for e in out], out_path)
        print(f"[augment] saved -> {out_path}")
    return out
