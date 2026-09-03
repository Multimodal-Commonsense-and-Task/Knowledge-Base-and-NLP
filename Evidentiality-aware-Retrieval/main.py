#!/usr/bin/env python3
"""EADPR -- Evidentiality-Aware Dense Passage Retrieval (Findings of EACL 2024).

The subcommands mirror the structure of the paper:

    toy        generate synthetic demo data (runs the whole flow with no downloads)
    augment    Sec. 3.1  remove a span from the gold passage to build a distractor p*
    train      Sec. 3.2  train the dual encoder with L_eadpr = L_dpr + t1*L_HN + t2*L_PP
    retrieve   Sec. 4    Top-k accuracy, MRR, R@k
    aa         Sec. 5    Answer-Awareness score (Eq.9)
    robustness Sec. 5    effect of injecting distractors into the corpus

Example:
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
    print(f"[model] no checkpoint at {ckpt}; using an untrained encoder")
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
        sys.exit(f"{path} does not exist. Run `main.py augment --split dev` first.")
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
                        help="cap on CPU threads; leaving this at 0 is much slower on many-core machines")
        sp.add_argument("--out", default=None)
        if model_args:
            sp.add_argument("--tiny", action="store_true",
                            help="skip pretrained weights and run a small randomly initialized model")
            sp.add_argument("--encoder", default=None, help="defaults to bert-base-uncased")

    sp = sub.add_parser("toy", help="generate synthetic demo data")
    common(sp, model_args=False)
    sp.add_argument("--n-train", type=int, default=48)
    sp.add_argument("--n-dev", type=int, default=16)
    sp.set_defaults(func=cmd_toy)

    sp = sub.add_parser("augment", help="Sec. 3.1 distractor augmentation")
    common(sp)
    sp.add_argument("--split", default="train")
    sp.add_argument("--no-qa-model", action="store_true",
                    help="select distractors by a length heuristic instead of a QA model")
    sp.set_defaults(func=cmd_augment)

    sp = sub.add_parser("train", help="Sec. 3.2 EADPR training")
    common(sp)
    sp.add_argument("--train-file", default=None)
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--lr", type=float, default=None)
    sp.add_argument("--lambda-distractor", type=float, default=None, help="lambda in Eq.7")
    sp.add_argument("--tau-hn", type=float, default=None, help="tau1 in Eq.8")
    sp.add_argument("--tau-pp", type=float, default=None, help="tau2 in Eq.8")
    sp.add_argument("--no-hn", action="store_true", help="drop L_HN (ablation)")
    sp.add_argument("--no-pp", action="store_true", help="drop L_PP (ablation)")
    sp.set_defaults(func=cmd_train)

    for name, fn, helptext in (("retrieve", cmd_retrieve, "Sec. 4 retrieval metrics"),
                               ("aa", cmd_aa, "Sec. 5 Answer-Awareness (Eq.9)"),
                               ("robustness", cmd_robustness, "Sec. 5 distractor injection")):
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
