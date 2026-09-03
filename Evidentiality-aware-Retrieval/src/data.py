"""Data schema, loaders and the toy dataset builder.

The format follows the DPR convention: one line per question.

    {"qid", "question", "answers": [...],
     "positive_ctx":   {"pid", "title", "text"},           # p+  (gold evidence)
     "distractor_ctx": {"pid", "title", "text"},           # p*  (filled in by `augment`)
     "hard_negative_ctxs": [{"pid","title","text"}, ...],  # p-  (BM25 / ANCE mining)
     "supporting_pids": [...]}                             # multi-hop R@k (optional)

corpus.jsonl holds one {"pid", "title", "text"} per line.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Passage:
    pid: str
    title: str
    text: str

    def to_dict(self) -> dict:
        return {"pid": self.pid, "title": self.title, "text": self.text}

    @staticmethod
    def from_dict(d: dict) -> "Passage":
        return Passage(pid=str(d["pid"]), title=d.get("title", ""), text=d["text"])


@dataclass
class QAExample:
    qid: str
    question: str
    answers: list[str]
    positive_ctx: Passage
    distractor_ctx: Passage | None = None
    hard_negative_ctxs: list[Passage] = field(default_factory=list)
    supporting_pids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "question": self.question, "answers": self.answers,
            "positive_ctx": self.positive_ctx.to_dict(),
            "distractor_ctx": self.distractor_ctx.to_dict() if self.distractor_ctx else None,
            "hard_negative_ctxs": [c.to_dict() for c in self.hard_negative_ctxs],
            "supporting_pids": self.supporting_pids,
        }

    @staticmethod
    def from_dict(d: dict) -> "QAExample":
        return QAExample(
            qid=str(d["qid"]), question=d["question"], answers=list(d.get("answers", [])),
            positive_ctx=Passage.from_dict(d["positive_ctx"]),
            distractor_ctx=Passage.from_dict(d["distractor_ctx"]) if d.get("distractor_ctx") else None,
            hard_negative_ctxs=[Passage.from_dict(c) for c in d.get("hard_negative_ctxs", [])],
            supporting_pids=[str(p) for p in d.get("supporting_pids", [])],
        )


# --------------------------------------------------------------------------- io

def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `main.py toy` to generate a toy dataset.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_examples(path: Path | str) -> list[QAExample]:
    return [QAExample.from_dict(d) for d in read_jsonl(path)]


def load_corpus(path: Path | str) -> list[Passage]:
    return [Passage.from_dict(d) for d in read_jsonl(path)]


# ------------------------------------------------------------------ span splitting

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_spans(text: str, min_chars: int = 0) -> list[str]:
    """Split p+ into n discrete spans (the paper assumes sentence granularity)."""
    spans = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if min_chars:
        spans = [s for s in spans if len(s) >= min_chars] or spans
    return spans


def remove_span(spans: list[str], idx: int) -> str:
    """p* = [s_l ; s_r] -- rejoin every span except the i-th one (Section 3.1)."""
    return " ".join(s for k, s in enumerate(spans) if k != idx).strip()


def mask_answer(text: str, answers: list[str]) -> str:
    """p' for the AA score (Eq.9) -- delete the answer string verbatim."""
    out = text
    for a in sorted(answers, key=len, reverse=True):
        if not a:
            continue
        out = re.sub(re.escape(a), " ", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


# -------------------------------------------------------------------- toy dataset

_TOPICS = [
    ("Easter Bunny", "German Lutherans",
     "where does the origin of the easter bunny come from"),
    ("Mount Everest", "8848 metres", "how tall is mount everest"),
    ("Great Barrier Reef", "Coral Sea", "where is the great barrier reef located"),
    ("Alan Turing", "1912", "when was alan turing born"),
    ("Amazon River", "South America", "which continent is the amazon river on"),
    ("Penicillin", "Alexander Fleming", "who discovered penicillin"),
    ("Mona Lisa", "Louvre", "which museum holds the mona lisa"),
    ("Photosynthesis", "chloroplasts", "where does photosynthesis take place"),
    ("Halley's Comet", "76 years", "how often does halleys comet appear"),
    ("Esperanto", "L. L. Zamenhof", "who created the esperanto language"),
]

_FILLER = [
    "The subject has been discussed in a wide range of popular sources.",
    "Several encyclopedias describe the topic in some detail.",
    "Its cultural significance is frequently noted by commentators.",
    "Later accounts expand on the same material with minor variation.",
]


def build_toy_dataset(out_dir: Path | str, n_train: int = 48, n_dev: int = 16,
                      n_distractor_docs: int = 60, seed: int = 42):
    """A small synthetic dataset for exercising the pipeline without downloads.

    Each gold passage is one answer-bearing sentence plus a few filler sentences,
    so span-removal distractor augmentation (Section 3.1) has something to work on.
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    corpus: list[dict] = []
    examples: list[dict] = []

    def make_gold(topic, answer, pid):
        evidence = f"The record states that {topic} is associated with {answer}."
        body = rng.sample(_FILLER, k=2)
        pos = rng.randint(0, len(body))
        sents = body[:pos] + [evidence] + body[pos:]
        return {"pid": pid, "title": topic, "text": " ".join(sents)}

    n_total = n_train + n_dev
    for i in range(n_total):
        topic, answer, qtmpl = _TOPICS[i % len(_TOPICS)]
        topic_i = topic if i < len(_TOPICS) else f"{topic} ({i // len(_TOPICS)})"
        pid = f"gold-{i}"
        gold = make_gold(topic_i, answer, pid)
        corpus.append(gold)
        examples.append({
            "qid": f"q-{i}", "question": qtmpl, "answers": [answer],
            "positive_ctx": gold, "distractor_ctx": None,
            "hard_negative_ctxs": [], "supporting_pids": [pid],
        })

    for j in range(n_distractor_docs):
        topic, _, _ = _TOPICS[j % len(_TOPICS)]
        corpus.append({"pid": f"noise-{j}", "title": f"{topic} in culture",
                       "text": " ".join(rng.sample(_FILLER, k=3))})

    # Stand-in for BM25 / ANCE mining: attach a noise document on the same topic.
    by_title = {}
    for c in corpus:
        by_title.setdefault(c["title"].split(" (")[0], []).append(c)
    for ex in examples:
        base = ex["positive_ctx"]["title"].split(" (")[0]
        cands = [c for c in by_title.get(base, []) if c["pid"] != ex["positive_ctx"]["pid"]]
        ex["hard_negative_ctxs"] = rng.sample(cands, k=min(1, len(cands)))

    rng.shuffle(examples)
    train, dev = examples[:n_train], examples[n_train:]
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(dev, out_dir / "dev.jsonl")
    write_jsonl(corpus, out_dir / "corpus.jsonl")
    print(f"[toy] train={len(train)} dev={len(dev)} corpus={len(corpus)} -> {out_dir}")
    return out_dir
