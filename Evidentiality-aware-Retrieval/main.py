#!/usr/bin/env python3
"""EADPR — Evidentiality-Aware Dense Passage Retrieval (EACL 2024 Findings) 재현 구현.

파이프라인은 논문의 절 구성을 그대로 따른다:

    toy        데모용 합성 데이터 생성 (다운로드 없이 전체 흐름을 돌려보기 위함)
    augment    §3.1  gold passage 에서 span 을 빼 distractor p* 를 만든다
    train      §3.2  L_eadpr = L_dpr + τ1·L_HN + τ2·L_PP 로 dual encoder 학습
    retrieve   §4    Top-k accuracy · MRR · R@k
    aa         §5    Answer-Awareness score (Eq.9)
    robustness §5    코퍼스에 distractor 를 주입했을 때의 성능 변화

예:
    python main.py toy
    python main.py augment --tiny --split train
    python main.py train   --tiny --epochs 3
    python main.py aa      --tiny
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import config as C
from src.config import EADPRConfig
from src.data import build_toy_dataset, load_corpus, load_examples, write_jsonl


def _cfg_from_args(args) -> EADPRConfig:
    cfg = EADPRConfig()
    cfg.seed = args.seed
    cfg.model.tiny = args.tiny
    cfg.distractor.tiny = args.tiny
    if getattr(args, "encoder", None):
        cfg.model.encoder_name = args.encoder
    if getattr(args, "epochs", None):
        cfg.train.epochs = args.epochs
    if getattr(args, "batch_size", None):
        cfg.train.batch_size = args.batch_size
    if getattr(args, "lr", None):
        cfg.train.lr = args.lr
    if getattr(args, "device", None):
        cfg.train.device = args.device
    for name in ("lambda_distractor", "tau_hn", "tau_pp"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg.loss, name, v)
    if getattr(args, "no_hn", False):
        cfg.loss.use_hn = False
    if getattr(args, "no_pp", False):
        cfg.loss.use_pp = False
    return cfg


def _dump(obj, path: Path | None):
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"[saved] {path}")


def cmd_toy(args) -> int:
    build_toy_dataset(args.data_dir, n_train=args.n_train, n_dev=args.n_dev, seed=args.seed)
    return 0


def cmd_augment(args) -> int:
    from src.distractors import augment_dataset
    from src.train import resolve_device

    cfg = _cfg_from_args(args)
    src_path = Path(args.data_dir) / f"{args.split}.jsonl"
    out_path = args.out or Path(args.data_dir) / f"{args.split}.augmented.jsonl"
    examples = load_examples(src_path)
    augment_dataset(examples, cfg.distractor, device=resolve_device(args.device),
                    out_path=out_path, use_qa_model=not args.no_qa_model)
    return 0


def cmd_train(args) -> int:
    from src.train import train

    cfg = _cfg_from_args(args)
    path = args.train_file or Path(args.data_dir) / "train.augmented.jsonl"
    examples = load_examples(path)
    train(examples, cfg, out_dir=args.out or C.CKPT_DIR)
    return 0


def _load_model(args):
    from src.train import load_checkpoint
    from src.modeling import DualEncoder

    ckpt = Path(args.checkpoint) if args.checkpoint else C.CKPT_DIR / "eadpr.pt"
    if ckpt.exists():
        print(f"[model] checkpoint <- {ckpt}")
        return load_checkpoint(ckpt)
    print(f"[model] checkpoint 가 없어 학습하지 않은 인코더를 쓴다 ({ckpt})")
    return DualEncoder(_cfg_from_args(args).model)


def cmd_retrieve(args) -> int:
    from src.retrieval import evaluate_retrieval
    from src.train import resolve_device

    model = _load_model(args)
    examples = load_examples(args.eval_file or Path(args.data_dir) / "dev.jsonl")
    corpus = load_corpus(Path(args.data_dir) / "corpus.jsonl")
    res = evaluate_retrieval(model, examples, corpus, ks=tuple(args.ks),
                             device=resolve_device(args.device))
    _dump(res, args.out)
    return 0


def cmd_aa(args) -> int:
    from src.evaluate import answer_awareness
    from src.train import resolve_device

    model = _load_model(args)
    examples = load_examples(args.eval_file or Path(args.data_dir) / "dev.jsonl")
    res = answer_awareness(model, examples, device=resolve_device(args.device))
    _dump(res, args.out)
    return 0


def cmd_robustness(args) -> int:
    from src.evaluate import robustness_test
    from src.train import resolve_device

    model = _load_model(args)
    path = args.eval_file or Path(args.data_dir) / "dev.augmented.jsonl"
    if not Path(path).exists():
        sys.exit(f"{path} 가 없다. 먼저 `main.py augment --split dev` 를 돌릴 것.")
    examples = load_examples(path)
    corpus = load_corpus(Path(args.data_dir) / "corpus.jsonl")
    res = robustness_test(model, examples, corpus, ks=tuple(args.ks),
                          device=resolve_device(args.device))
    _dump(res, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, model_args: bool = True):
        sp.add_argument("--data-dir", default=str(C.DATA_DIR / "toy"))
        sp.add_argument("--seed", type=int, default=C.SEED)
        sp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        sp.add_argument("--threads", type=int, default=4,
                        help="CPU 스레드 상한. 코어 많은 노드에서 0 으로 두면 크게 느려진다")
        sp.add_argument("--out", default=None)
        if model_args:
            sp.add_argument("--tiny", action="store_true",
                            help="사전학습 가중치를 받지 않고 소형 랜덤 모델로 돌린다")
            sp.add_argument("--encoder", default=None, help="기본 bert-base-uncased")

    sp = sub.add_parser("toy", help="합성 데모 데이터 생성")
    common(sp, model_args=False)
    sp.add_argument("--n-train", type=int, default=48)
    sp.add_argument("--n-dev", type=int, default=16)
    sp.set_defaults(func=cmd_toy)

    sp = sub.add_parser("augment", help="§3.1 distractor 증강")
    common(sp)
    sp.add_argument("--split", default="train")
    sp.add_argument("--no-qa-model", action="store_true",
                    help="QA 모델 없이 길이 휴리스틱으로 고른다")
    sp.set_defaults(func=cmd_augment)

    sp = sub.add_parser("train", help="§3.2 EADPR 학습")
    common(sp)
    sp.add_argument("--train-file", default=None)
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--lr", type=float, default=None)
    sp.add_argument("--lambda-distractor", type=float, default=None, help="Eq.7 의 λ")
    sp.add_argument("--tau-hn", type=float, default=None, help="Eq.8 의 τ1")
    sp.add_argument("--tau-pp", type=float, default=None, help="Eq.8 의 τ2")
    sp.add_argument("--no-hn", action="store_true", help="L_HN 제거 (ablation)")
    sp.add_argument("--no-pp", action="store_true", help="L_PP 제거 (ablation)")
    sp.set_defaults(func=cmd_train)

    for name, fn, helptext in (("retrieve", cmd_retrieve, "§4 검색 성능"),
                               ("aa", cmd_aa, "§5 Answer-Awareness (Eq.9)"),
                               ("robustness", cmd_robustness, "§5 distractor 주입 실험")):
        sp = sub.add_parser(name, help=helptext)
        common(sp)
        sp.add_argument("--checkpoint", default=None)
        sp.add_argument("--eval-file", default=None)
        sp.add_argument("--ks", type=int, nargs="*", default=[1, 5, 20])
        sp.set_defaults(func=fn)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if getattr(args, "threads", None):
        from src.train import cap_threads
        cap_threads(args.threads)
    sys.exit(args.func(args))
