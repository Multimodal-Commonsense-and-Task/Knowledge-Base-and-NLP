"""EADPR 학습 루프.

배치마다 (q, p+, p*, [p-]) 를 인코딩해 Eq.8 의 L_eadpr 를 최소화한다.
in-batch negative 를 쓰므로 배치 안의 다른 질문의 p+ 가 p- 역할을 겸한다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from .config import EADPRConfig
from .data import QAExample
from .losses import eadpr_loss
from .modeling import DualEncoder, passage_text


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def cap_threads(n: int | None) -> None:
    """CPU 스레드 상한.

    코어가 많은 노드에서 작은 텐서를 돌리면 스레드 오버헤드가 연산을 압도한다 —
    이 저장소의 tiny 경로에서 52 스레드 대비 4 스레드가 100 배 이상 빨랐다.
    """
    if n and n > 0:
        torch.set_num_threads(n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batches(examples: list[QAExample], batch_size: int, shuffle: bool = True,
                 rng: random.Random | None = None) -> list[list[QAExample]]:
    """distractor 가 없는 예제는 학습에서 제외한다 — L_HN / L_PP 가 정의되지 않는다."""
    usable = [e for e in examples if e.distractor_ctx is not None]
    if shuffle:
        (rng or random).shuffle(usable)
    # in-batch negative 를 쓰려면 배치 크기가 2 이상이어야 한다.
    return [usable[i:i + batch_size] for i in range(0, len(usable), batch_size)
            if len(usable[i:i + batch_size]) >= 2]


def encode_batch(model: DualEncoder, batch: list[QAExample]):
    q = model.encode_questions([e.question for e in batch])
    p_pos = model.encode_passages([passage_text(e.positive_ctx) for e in batch])
    p_dis = model.encode_passages([passage_text(e.distractor_ctx) for e in batch])

    s_pp = DualEncoder.similarity(q, p_pos)
    s_ds = DualEncoder.similarity(q, p_dis)

    # 명시적 hard negative (BM25 / ANCE). 질문마다 1개씩 쓰고, 배치 전체가 공유한다.
    hns = [e.hard_negative_ctxs[0] for e in batch if e.hard_negative_ctxs]
    s_hn = None
    if len(hns) == len(batch) and hns:
        p_hn = model.encode_passages([passage_text(p) for p in hns])
        s_hn = DualEncoder.similarity(q, p_hn)
    return s_pp, s_ds, s_hn


def train(examples: list[QAExample], cfg: EADPRConfig,
          out_dir: Path | str | None = None) -> DualEncoder:
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    print(f"[train] device={device} tiny={cfg.model.tiny}")

    model = DualEncoder(cfg.model).to(device)
    model.train()

    batches = make_batches(examples, cfg.train.batch_size, shuffle=False)
    if not batches:
        raise ValueError(
            "학습 가능한 배치가 없다. distractor 가 채워졌는지(`main.py augment`), "
            "배치 크기가 2 이상인지 확인할 것.")

    opt = AdamW(model.parameters(), lr=cfg.train.lr, eps=cfg.train.adam_eps,
                betas=cfg.train.adam_betas)
    total_steps = cfg.train.epochs * len(batches)
    warmup = int(total_steps * cfg.train.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        left = total_steps - warmup
        return max(0.0, (total_steps - step) / max(left, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = random.Random(cfg.seed)

    history = []
    step = 0
    for epoch in range(cfg.train.epochs):
        epoch_batches = make_batches(examples, cfg.train.batch_size, shuffle=True, rng=rng)
        agg = {}
        bar = tqdm(epoch_batches, desc=f"epoch {epoch + 1}/{cfg.train.epochs}", leave=False)
        for batch in bar:
            s_pp, s_ds, s_hn = encode_batch(model, batch)
            out = eadpr_loss(s_pp, s_ds, s_hn, cfg.loss)

            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            opt.step()
            sched.step()
            step += 1

            for k, v in out.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            if step % cfg.train.log_every == 0:
                bar.set_postfix(loss=f"{float(out['loss']):.4f}")

        line = {k: v / len(epoch_batches) for k, v in agg.items()}
        line["epoch"] = epoch + 1
        history.append(line)
        print(f"[epoch {epoch + 1}] " + "  ".join(
            f"{k}={v:.4f}" for k, v in line.items() if k != "epoch"))

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()},
                   out_dir / "eadpr.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"[train] saved -> {out_dir / 'eadpr.pt'}")
    return model


def load_checkpoint(path: Path | str, cfg: EADPRConfig | None = None) -> DualEncoder:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if cfg is None:
        from .config import EADPRConfig as _C, ModelConfig, LossConfig, TrainConfig, DistractorConfig
        d = ckpt["config"]
        cfg = _C(model=ModelConfig(**d["model"]), loss=LossConfig(**d["loss"]),
                 train=TrainConfig(**d["train"]),
                 distractor=DistractorConfig(**d["distractor"]), seed=d["seed"])
    model = DualEncoder(cfg.model)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
