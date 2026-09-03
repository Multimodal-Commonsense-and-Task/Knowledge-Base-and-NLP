"""EADPR loss functions (Eq.3 / Eq.5 / Eq.7 / Eq.8 of the paper).

Notation
  s_pos[i]      = <q_i, p+_i>            question and gold evidence
  s_dis[i][j]   = <q_i, p*_j>            question and distractor (span removed)
  s_neg[i][j]   = <q_i, p-_j>            question and negative (in-batch + hard negative)

The ordering the three losses are aiming for (Fig.3):
  <q_i, p+_i>  >  <q_i, p*_i>  >  <q_i, p-_j>
       |- Eq.2 (L_HN)       |- Eq.4 (L_PP)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig


def _logsumexp_cat(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(torch.cat(tensors, dim=-1), dim=-1)


def distractors_as_hard_negatives(s_pos: torch.Tensor, s_dis_self: torch.Tensor
                                  ) -> torch.Tensor:
    """Eq.3 -- L_HN.

        -log( exp<q,p+> / (exp<q,p+> + exp<q,p*>) )

    A binary softmax over the example's own distractor only. Pushing p+ above p*
    keeps the causal contribution of the evidence span in the embedding.
    """
    logits = torch.stack([s_pos, s_dis_self], dim=-1)       # (B, 2)
    target = torch.zeros(len(s_pos), dtype=torch.long, device=s_pos.device)
    return F.cross_entropy(logits, target)


def distractors_as_pseudo_positives(s_dis_self: torch.Tensor, s_neg: torch.Tensor,
                                    s_dis_other: torch.Tensor) -> torch.Tensor:
    """Eq.5 -- L_PP.

        -log( exp<q,p*_i> / (exp<q,p*_i> + sum_{j!=i} (exp<q,p-_j> + exp<q,p*_j>)) )

    Lifts the distractor above irrelevant passages, making p* a pivot between p+
    and p-. s_neg and s_dis_other must already have the diagonal masked out.
    """
    num = s_dis_self                                        # (B,)
    den = _logsumexp_cat(s_dis_self.unsqueeze(-1), s_neg, s_dis_other)
    return (den - num).mean()


def dpr_with_distractor(s_pos: torch.Tensor, s_neg: torch.Tensor,
                        s_dis_self: torch.Tensor, lambda_distractor: float
                        ) -> torch.Tensor:
    """Eq.7 -- L_dpr. Standard DPR (Eq.1) plus the own distractor as a weighted negative.

        -log( exp<q,p+> / (exp<q,p+> + sum_{j!=i} exp<q,p-_j> + lambda*exp<q,p*_i>) )

    lambda multiplies an exp term, so it is added to the logit as log(lambda).
    At lambda=0 the term disappears and this is exactly vanilla DPR.
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
    """Eq.8 -- L_eadpr = L_dpr + tau1*L_HN + tau2*L_PP.

    Args
      s_pp : (B, B)  <q_i, p+_j>          gold on the diagonal
      s_ds : (B, B)  <q_i, p*_j>          own distractor on the diagonal
      s_hn : (B, H) or None               explicit hard negatives (BM25 / ANCE)
    """
    B = s_pp.size(0)
    device = s_pp.device
    eye = torch.eye(B, dtype=torch.bool, device=device)

    s_pos = s_pp.diagonal()                                  # <q_i, p+_i>
    s_dis_self = s_ds.diagonal()                             # <q_i, p*_i>

    # In-batch negatives: the gold passages of the other questions, diagonal removed.
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
    # How often the ordering constraints (Eq.2 / Eq.4) actually hold during training.
    with torch.no_grad():
        out["acc_pos_over_dis"] = (s_pos > s_dis_self).float().mean()
        out["acc_dis_over_neg"] = (s_dis_self.unsqueeze(-1) > s_neg).float().mean()
    return out
