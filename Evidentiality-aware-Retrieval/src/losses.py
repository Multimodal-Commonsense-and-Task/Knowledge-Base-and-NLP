"""EADPR 손실함수 (논문 Eq.3 / Eq.5 / Eq.7 / Eq.8).

기호
  s_pos[i]      = <q_i, p+_i>            질문과 gold evidence
  s_dis[i][j]   = <q_i, p*_j>            질문과 distractor (span 제거본)
  s_neg[i][j]   = <q_i, p-_j>            질문과 negative (in-batch + hard negative)

세 손실이 노리는 순서 관계 (Fig.3):
  <q_i, p+_i>  >  <q_i, p*_i>  >  <q_i, p-_j>
       └ Eq.2 (L_HN)        └ Eq.4 (L_PP)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig


def _logsumexp_cat(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(torch.cat(tensors, dim=-1), dim=-1)


def distractors_as_hard_negatives(s_pos: torch.Tensor, s_dis_self: torch.Tensor
                                  ) -> torch.Tensor:
    """Eq.3 — L_HN.

        -log( exp<q,p+> / (exp<q,p+> + exp<q,p*>) )

    자기 자신의 distractor 하나만 쓰는 이항 softmax 다. p+ 를 p* 위로 밀어 올려
    evidence span 의 인과적 기여를 임베딩에 남긴다.
    """
    logits = torch.stack([s_pos, s_dis_self], dim=-1)       # (B, 2)
    target = torch.zeros(len(s_pos), dtype=torch.long, device=s_pos.device)
    return F.cross_entropy(logits, target)


def distractors_as_pseudo_positives(s_dis_self: torch.Tensor, s_neg: torch.Tensor,
                                    s_dis_other: torch.Tensor) -> torch.Tensor:
    """Eq.5 — L_PP.

        -log( exp<q,p*_i> / (exp<q,p*_i> + Σ_{j≠i} (exp<q,p-_j> + exp<q,p*_j>)) )

    distractor 를 '무관한 패시지들 위' 로 올린다. p* 가 p+ 와 p- 사이의 pivot 이 된다.
    s_neg / s_dis_other 는 이미 대각(자기 자신)이 마스킹돼 들어와야 한다.
    """
    num = s_dis_self                                        # (B,)
    den = _logsumexp_cat(s_dis_self.unsqueeze(-1), s_neg, s_dis_other)
    return (den - num).mean()


def dpr_with_distractor(s_pos: torch.Tensor, s_neg: torch.Tensor,
                        s_dis_self: torch.Tensor, lambda_distractor: float
                        ) -> torch.Tensor:
    """Eq.7 — L_dpr. 표준 DPR(Eq.1)에 자기 distractor 를 λ 가중 음성으로 추가한다.

        -log( exp<q,p+> / (exp<q,p+> + Σ_{j≠i} exp<q,p-_j> + λ·exp<q,p*_i>) )

    λ 는 exp 항에 곱해지므로 log λ 를 로짓에 더하는 것과 같다.
    λ=0 이면 항이 사라지고 원래 DPR 이 된다.
    """
    parts = [s_pos.unsqueeze(-1), s_neg]
    if lambda_distractor > 0:
        shifted = s_dis_self + torch.log(
            torch.tensor(lambda_distractor, device=s_pos.device, dtype=s_pos.dtype))
        parts.append(shifted.unsqueeze(-1))
    den = _logsumexp_cat(*parts)
    return (den - s_pos).mean()


def eadpr_loss(s_pp: torch.Tensor, s_ds: torch.Tensor,
               s_hn: torch.Tensor | None, cfg: LossConfig) -> dict:
    """Eq.8 — L_eadpr = L_dpr + τ1·L_HN + τ2·L_PP.

    인자
      s_pp : (B, B)  <q_i, p+_j>          대각이 gold
      s_ds : (B, B)  <q_i, p*_j>          대각이 자기 distractor
      s_hn : (B, H) or None               명시적 hard negative (BM25 / ANCE)
    """
    B = s_pp.size(0)
    device = s_pp.device
    eye = torch.eye(B, dtype=torch.bool, device=device)

    s_pos = s_pp.diagonal()                                  # <q_i, p+_i>
    s_dis_self = s_ds.diagonal()                             # <q_i, p*_i>

    # in-batch negative: 다른 질문의 gold passage. 자기 대각은 뺀다.
    off = ~eye
    s_neg_inbatch = s_pp.masked_select(off).view(B, B - 1)
    s_neg = s_neg_inbatch if s_hn is None else torch.cat([s_neg_inbatch, s_hn], dim=-1)
    s_dis_other = s_ds.masked_select(off).view(B, B - 1)

    l_dpr = dpr_with_distractor(s_pos, s_neg, s_dis_self, cfg.lambda_distractor)
    out = {"loss_dpr": l_dpr}
    total = l_dpr

    if cfg.use_hn:
        l_hn = distractors_as_hard_negatives(s_pos, s_dis_self)
        out["loss_hn"] = l_hn
        total = total + cfg.tau_hn * l_hn
    if cfg.use_pp:
        l_pp = distractors_as_pseudo_positives(s_dis_self, s_neg, s_dis_other)
        out["loss_pp"] = l_pp
        total = total + cfg.tau_pp * l_pp

    out["loss"] = total
    # 학습 중 순서 제약(Eq.2 / Eq.4)이 실제로 지켜지는 비율
    with torch.no_grad():
        out["acc_pos_over_dis"] = (s_pos > s_dis_self).float().mean()
        out["acc_dis_over_neg"] = (s_dis_self.unsqueeze(-1) > s_neg).float().mean()
    return out
