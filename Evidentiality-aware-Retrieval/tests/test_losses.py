"""손실함수가 논문 수식(Eq.3 / Eq.5 / Eq.7 / Eq.8)과 맞는지 확인한다.

    python -m pytest tests/ -q      또는      python tests/test_losses.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LossConfig
from src.losses import (distractors_as_hard_negatives, dpr_with_distractor,
                        eadpr_loss)

B = 4


def _scores(pos: float, dis: float, neg: float):
    s_pp = torch.full((B, B), neg); s_pp.fill_diagonal_(pos)
    s_ds = torch.full((B, B), neg); s_ds.fill_diagonal_(dis)
    return s_pp, s_ds


def test_ordering_is_rewarded():
    """<q,p+> > <q,p*> > <q,p-> 인 배치가 뒤집힌 배치보다 손실이 낮아야 한다."""
    cfg = LossConfig()
    good = eadpr_loss(*_scores(6.0, 3.0, 0.0), None, cfg)["loss"]
    bad = eadpr_loss(*_scores(0.0, 3.0, 6.0), None, cfg)["loss"]
    assert good < bad


def test_hn_matches_equation_3():
    """L_HN = -log( e^s+ / (e^s+ + e^s*) )."""
    s_pos = torch.tensor([2.0, 1.0])
    s_dis = torch.tensor([0.5, 3.0])
    manual = -torch.log(torch.exp(s_pos) / (torch.exp(s_pos) + torch.exp(s_dis))).mean()
    assert torch.allclose(distractors_as_hard_negatives(s_pos, s_dis), manual)


def test_dpr_reduces_to_vanilla_when_lambda_zero():
    """λ=0 이면 Eq.7 이 표준 DPR(Eq.1)과 같아진다."""
    s_pos = torch.tensor([2.0, 1.0])
    s_neg = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    s_dis = torch.tensor([5.0, 5.0])
    ref = (torch.logsumexp(torch.cat([s_pos.unsqueeze(-1), s_neg], -1), -1) - s_pos).mean()
    assert torch.allclose(dpr_with_distractor(s_pos, s_neg, s_dis, 0.0), ref)


def test_lambda_monotonic():
    """λ 가 커질수록 distractor 의 음성 효과가 세져 L_dpr 이 커진다."""
    s_pos = torch.tensor([2.0, 1.0])
    s_neg = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    s_dis = torch.tensor([5.0, 5.0])
    vals = [float(dpr_with_distractor(s_pos, s_neg, s_dis, l)) for l in (0.0, 0.1, 0.5, 1.0)]
    assert vals == sorted(vals)


def test_gradients_flow():
    cfg = LossConfig()
    s_pp = torch.randn(B, B, requires_grad=True)
    s_ds = torch.randn(B, B, requires_grad=True)
    eadpr_loss(s_pp, s_ds, None, cfg)["loss"].backward()
    for g in (s_pp.grad, s_ds.grad):
        assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_hard_negatives_enter_pp_denominator():
    """명시적 hard negative 를 넣으면 L_PP 분모가 커져 손실이 증가한다."""
    cfg = LossConfig()
    s_pp, s_ds = _scores(6.0, 3.0, 0.0)
    without = eadpr_loss(s_pp, s_ds, None, cfg)["loss_pp"]
    with_hn = eadpr_loss(s_pp, s_ds, torch.full((B, 2), 2.5), cfg)["loss_pp"]
    assert with_hn > without


def test_ablation_flags():
    """--no-hn / --no-pp 가 해당 항을 실제로 제거한다."""
    s_pp, s_ds = _scores(6.0, 3.0, 0.0)
    full = eadpr_loss(s_pp, s_ds, None, LossConfig())
    only_dpr = eadpr_loss(s_pp, s_ds, None, LossConfig(use_hn=False, use_pp=False))
    assert "loss_hn" not in only_dpr and "loss_pp" not in only_dpr
    assert torch.allclose(only_dpr["loss"], full["loss_dpr"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
