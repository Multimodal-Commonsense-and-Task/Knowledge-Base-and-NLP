"""EADPR 하이퍼파라미터.

값은 논문 Appendix C (Implementation Details) 를 따른다.
  - dual encoder: BERT-base
  - 40 epochs, batch size 16, lr 2e-5, Adam eps 1e-8, betas (0.9, 0.999)
  - λ (Eq.7 의 distractor 가중치) = 1.0,  {0.1,0.2,0.5,0.9,1.0} 중 선택
  - τ1, τ2 (Eq.8) = 1.0, grid search 로 결정
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

SEED = 42


@dataclass
class ModelConfig:
    encoder_name: str = "bert-base-uncased"
    # tiny=True 면 사전학습 가중치를 받지 않고 소형 랜덤 초기화 BERT 를 쓴다.
    # 다운로드 없이 파이프라인 전체를 돌려보기 위한 경로다 (성능은 의미 없다).
    tiny: bool = False
    tiny_vocab_size: int = 4096
    tiny_hidden_size: int = 64
    tiny_layers: int = 2
    tiny_heads: int = 2
    tiny_intermediate: int = 128
    max_q_len: int = 64
    max_p_len: int = 192
    share_encoder: bool = False   # DPR 은 question/passage 인코더를 분리한다


@dataclass
class LossConfig:
    """Eq.7 의 λ, Eq.8 의 τ1 / τ2."""
    lambda_distractor: float = 1.0   # λ < 1 로 두면 distractor 의 음성 효과를 약화
    tau_hn: float = 1.0              # τ1 · L_HN
    tau_pp: float = 1.0              # τ2 · L_PP
    use_hn: bool = True
    use_pp: bool = True


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-5
    adam_eps: float = 1e-8
    adam_betas: tuple[float, float] = (0.9, 0.999)
    warmup_ratio: float = 0.1
    max_grad_norm: float = 2.0
    device: str = "auto"            # auto | cpu | cuda
    log_every: int = 10


@dataclass
class DistractorConfig:
    """§3.1 Augmenting Distractor Samples."""
    qa_model_name: str = "allenai/unifiedqa-t5-base"   # 논문의 UnifiedQA-T5
    tiny: bool = False
    max_candidates: int = 8      # p+ 를 나눈 span 중 후보로 삼을 최대 개수
    min_span_chars: int = 10     # 너무 짧은 span 은 후보에서 제외
    # 논문은 confidence 가 가장 낮은(= perplexity 가 가장 높은) 후보를 distractor 로 고른다.
    selection: str = "highest_perplexity"


@dataclass
class EADPRConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distractor: DistractorConfig = field(default_factory=DistractorConfig)
    seed: int = SEED

    def to_dict(self) -> dict:
        return asdict(self)
