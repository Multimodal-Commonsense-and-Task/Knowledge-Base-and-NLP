"""DPR 스타일 dual encoder [f_q, f_p].

논문은 BERT-base 두 개(질문/패시지 분리)를 쓰고 [CLS] 표현을 임베딩으로 삼는다.
관련도는 내적: <q, p> = f_q(q) · f_p(p).

tiny=True 는 사전학습 가중치를 받지 않고 소형 랜덤 BERT 를 만든다.
다운로드 없이 파이프라인을 끝까지 돌려보기 위한 경로다.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

from .config import ModelConfig


class HashTokenizer:
    """오프라인 폴백 토크나이저. 단어를 해시로 vocab 에 사상한다.

    사전학습 토크나이저를 받지 못하는 환경에서도 형태가 같은 배치를 만들기 위한 것이며,
    의미 있는 subword 분절을 하지는 않는다.
    """

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2

    def _ids(self, text: str) -> list[int]:
        out = []
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            out.append(3 + h % (self.vocab_size - 3))
        return out

    def __call__(self, texts, padding=True, truncation=True, max_length=128,
                 return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [[self.cls_token_id] + self._ids(t)[: max_length - 2] + [self.sep_token_id]
                for t in texts]
        n = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), n), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_encoder(cfg: ModelConfig):
    if cfg.tiny:
        bert_cfg = BertConfig(
            vocab_size=cfg.tiny_vocab_size, hidden_size=cfg.tiny_hidden_size,
            num_hidden_layers=cfg.tiny_layers, num_attention_heads=cfg.tiny_heads,
            intermediate_size=cfg.tiny_intermediate, max_position_embeddings=512,
        )
        return BertModel(bert_cfg)
    return AutoModel.from_pretrained(cfg.encoder_name)


def build_tokenizer(cfg: ModelConfig):
    if cfg.tiny:
        return HashTokenizer(cfg.tiny_vocab_size)
    return AutoTokenizer.from_pretrained(cfg.encoder_name)


class DualEncoder(nn.Module):
    """[f_q, f_p]. encode_* 는 [CLS] 벡터를 돌려준다."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q_encoder = _build_encoder(cfg)
        self.p_encoder = self.q_encoder if cfg.share_encoder else _build_encoder(cfg)
        self.tokenizer = build_tokenizer(cfg)

    @property
    def dim(self) -> int:
        return self.q_encoder.config.hidden_size

    def _encode(self, encoder, texts: list[str], max_length: int) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt")
        device = next(encoder.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        out = encoder(**batch)
        return out.last_hidden_state[:, 0]        # [CLS]

    def encode_questions(self, questions: list[str]) -> torch.Tensor:
        return self._encode(self.q_encoder, questions, self.cfg.max_q_len)

    def encode_passages(self, passages: list[str]) -> torch.Tensor:
        return self._encode(self.p_encoder, passages, self.cfg.max_p_len)

    @staticmethod
    def similarity(q_emb: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        """<q_i, p_j> 행렬 (내적)."""
        return q_emb @ p_emb.t()


def passage_text(p) -> str:
    """DPR 관례대로 title 과 text 를 이어 붙인다."""
    title = getattr(p, "title", "") or ""
    text = getattr(p, "text", "") if not isinstance(p, str) else p
    return f"{title} [SEP] {text}".strip() if title else text
