"""Section 3.1, Augmenting Distractor Samples -- span removal + pseudo-evidence.

Step 1. Split the gold evidence passage p+ = [s_l ; s+ ; s_r] into n spans and leave
        one out at a time to form n candidates p*_i = [s_l ; s_r].
Step 2. Feed (q, p*_i) to a generative QA model theta and measure the confidence of
        the gold answer, P_theta(a | q, p*_i). Take the candidate with the lowest
        confidence -- equivalently the highest perplexity -- as p*. A sharp drop in
        confidence means that span was contributing to the answer.

The paper uses UnifiedQA-T5 as theta.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from tqdm import tqdm

from .config import DistractorConfig
from .data import Passage, QAExample, remove_span, split_spans, write_jsonl


class QAScorer:
    """Wrapper around a generative QA model that scores the perplexity of P(a | q, p)."""

    def __init__(self, cfg: DistractorConfig, device: str = "cpu"):
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                  T5Config, T5ForConditionalGeneration)
        self.cfg = cfg
        self.device = device
        if cfg.tiny:
            # A small randomly initialized T5 so this runs without downloads. The
            # scores are meaningless; only the shape of the computation is the same.
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
        # UnifiedQA input convention: "question \n context"
        return f"{question.strip()} \\n {passage.strip()}"

    @torch.no_grad()
    def perplexity(self, questions: list[str], passages: list[str],
                   answers: list[str]) -> list[float]:
        """Exponentiated mean negative log-likelihood of the answer tokens."""
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
    """Step 1 -- one candidate per removed span, as (removed span index, text)."""
    spans = split_spans(passage.text, min_chars=cfg.min_span_chars)
    if len(spans) < 2:
        return []                       # nothing to remove, so no distractor is possible
    idxs = list(range(len(spans)))[: cfg.max_candidates]
    return [(i, remove_span(spans, i)) for i in idxs]


def augment_example(ex: QAExample, scorer: QAScorer | None,
                    cfg: DistractorConfig) -> QAExample:
    """Fill in p* for a single example."""
    cands = make_candidates(ex.positive_ctx, cfg)
    if not cands:
        return ex

    answer = ex.answers[0] if ex.answers else ""
    if scorer is None or not answer:
        # Fallback without a QA model: remove the longest span, assuming it carries
        # the most information.
        chosen_idx, chosen_text = min(cands, key=lambda c: len(c[1]))
        score = float("nan")
    else:
        ppl = scorer.perplexity([ex.question] * len(cands),
                                [c[1] for c in cands],
                                [answer] * len(cands))
        # Lowest confidence == highest perplexity.
        best = max(range(len(cands)), key=lambda k: ppl[k])
        chosen_idx, chosen_text = cands[best]
        score = ppl[best]

    ex.distractor_ctx = Passage(pid=f"{ex.positive_ctx.pid}::distractor",
                                title=ex.positive_ctx.title, text=chosen_text)
    ex.removed_span_idx = chosen_idx      # for analysis; not serialized
    ex.distractor_ppl = score
    return ex


def augment_dataset(examples: list[QAExample], cfg: DistractorConfig,
                    device: str = "cpu", out_path: Path | str | None = None,
                    use_qa_model: bool = True) -> list[QAExample]:
    scorer = QAScorer(cfg, device=device) if use_qa_model else None
    if scorer is None:
        print("[augment] no QA model; selecting distractors by a length heuristic")
    out = [augment_example(ex, scorer, cfg) for ex in tqdm(examples, desc="augment")]
    n = sum(1 for e in out if e.distractor_ctx)
    print(f"[augment] distractors created: {n}/{len(out)}")
    if out_path:
        write_jsonl([e.to_dict() for e in out], out_path)
        print(f"[augment] saved -> {out_path}")
    return out
