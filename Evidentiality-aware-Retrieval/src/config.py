"""EADPR hyperparameters.

Values follow Appendix C (Implementation Details) of the paper:
  - dual encoder: BERT-base
  - 40 epochs, batch size 16, lr 2e-5, Adam eps 1e-8, betas (0.9, 0.999)
  - lambda (the distractor weight in Eq.7) = 1.0, chosen from {0.1, 0.2, 0.5, 0.9, 1.0}
  - tau1, tau2 (Eq.8) = 1.0, found by grid search
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
    # With tiny=True no pretrained weights are downloaded; a small randomly
    # initialized BERT is used instead. This exists so the whole pipeline can be
    # exercised offline -- the resulting numbers are meaningless.
    tiny: bool = False
    tiny_vocab_size: int = 4096
    tiny_hidden_size: int = 64
    tiny_layers: int = 2
    tiny_heads: int = 2
    tiny_intermediate: int = 128
    max_q_len: int = 64
    max_p_len: int = 192
    share_encoder: bool = False   # DPR keeps the question and passage encoders separate


@dataclass
class LossConfig:
    """lambda from Eq.7 and tau1 / tau2 from Eq.8."""
    lambda_distractor: float = 1.0   # lambda < 1 weakens the distractor's negative effect
    tau_hn: float = 1.0              # tau1 * L_HN
    tau_pp: float = 1.0              # tau2 * L_PP
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
    """Section 3.1, Augmenting Distractor Samples."""
    qa_model_name: str = "allenai/unifiedqa-t5-base"   # UnifiedQA-T5, as in the paper
    tiny: bool = False
    max_candidates: int = 8      # cap on how many spans of p+ become candidates
    min_span_chars: int = 10     # spans shorter than this are not considered
    # The paper picks the candidate with the lowest confidence, i.e. highest perplexity.
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
