# Evidentiality-Aware Dense Passage Retrieval (EADPR)

Reimplementation of [*Evidentiality-aware Retrieval for Overcoming Abstractiveness in Open-Domain Question Answering*](https://aclanthology.org/2024.findings-eacl.130/) (Findings of EACL 2024).

Yongho Song\*, Dahyun Lee\*, Myungha Jang, Seung-won Hwang, Kyungjae Lee, Dongha Lee, Jinyoung Yeo

> ⚠ This repository ports the paper's **method** into runnable code. It is not the authors'
> official implementation, and it does not reproduce the reported numbers — see [Scope](#scope).

## Overview

In abstractive ODQA, **relevance and answerability come apart.** A retriever readily finds
passages that are semantically close to the question, but closeness is no guarantee that a
passage carries the evidence the answer rests on. Standard IR datasets annotate only
relevance, so they supply a weak supervision signal for evidentiality.

The usual fix is **model-centric**: run an iterative pipeline that annotates answerability
using signals from the reader. That works, but it is expensive. EADPR takes a
**data-centric** route instead — it synthesizes **distractors** by cutting the evidence span
out of a gold passage, and turns those distractors into training signal.

The key idea is to treat a distractor as a **pivot between the positive and the negative**.
A distractor is still relevant to the question; what it lacks is the evidence:

```
        <q, p+>   >   <q, p*>   >   <q, p->
        evidence      distractor    irrelevant
                 └ Eq.2            └ Eq.4
```

Each inequality becomes a loss — the distractor serves as a hard negative and as a
pseudo-positive at the same time — and both sit on top of the standard DPR objective.

## Method

### §3.1 Distractor augmentation — `src/distractors.py`

Removing the evidence span `s+` from a gold passage `p+ = [s_l ; s+ ; s_r]` leaves
`p* = [s_l ; s_r]`. The catch is that most datasets carry no evidence-span annotation, so
the paper works around it with **pseudo-evidence**:

1. Split `p+` into n spans and leave one out at a time to form candidates `{p*_i}`.
2. Feed `(q, p*_i)` to a generative QA model θ (UnifiedQA-T5 in the paper) and measure the
   confidence of the gold answer, `P_θ(a | q, p*_i)`.
3. Pick the candidate with the **lowest confidence — equivalently, the highest perplexity**
   — as `p*`. A sharp drop in confidence means that span was doing the answering.

### §3.2 Evidentiality-aware learning — `src/losses.py`

A weighted sum of three losses (Eq.8):

```
L_eadpr = L_dpr + τ1·L_HN + τ2·L_PP
```

| Loss | Formula | Role |
|---|---|---|
| `L_HN` (Eq.3) | `-log( e^<q,p+> / (e^<q,p+> + e^<q,p*>) )` | Distractor as a **hard negative** — pushes `p+` above `p*` |
| `L_PP` (Eq.5) | `-log( e^<q,p*ᵢ> / (e^<q,p*ᵢ> + Σ_{j≠i}(e^<q,p⁻ⱼ> + e^<q,p*ⱼ>)) )` | Distractor as a **pseudo-positive** — lifts it above irrelevant passages |
| `L_dpr` (Eq.7) | Standard DPR denominator plus `λ·e^<q,p*ᵢ>` | The example's own distractor as a λ-weighted negative |

At `λ=0`, `L_dpr` collapses exactly to vanilla DPR (Eq.1) — asserted in the tests.

### §5 Answer-Awareness — `src/evaluate.py`

Build `p'` by deleting the answer span, then measure how often the model still scores `p+`
above `p'` (Eq.9):

```
AA = 1 - (1/T) Σ 1[ <q,p+> ≤ <q,p'> ]
```

The paper's observation is that vanilla DPR falls well short of the theoretical upper bound
here, and is weakest on who/what/where questions whose answers are named entities. This
implementation also breaks the score down by question type.

## Repository Structure

```
.
├── main.py                # CLI — toy / augment / train / retrieve / aa / robustness
├── requirements.txt
├── tests/test_losses.py   # Checks the losses against the paper's equations
└── src/
    ├── config.py          # Hyperparameters from Appendix C
    ├── data.py            # Schema, JSONL io, span splitting, toy data builder
    ├── distractors.py     # §3.1 span removal + pseudo-evidence via QA perplexity
    ├── modeling.py        # DPR dual encoder [f_q, f_p] + offline fallback
    ├── losses.py          # Eq.3 / Eq.5 / Eq.7 / Eq.8
    ├── train.py           # Training loop (in-batch negatives)
    ├── retrieval.py       # Corpus encoding, search, Top-k / MRR / R@k
    └── evaluate.py        # AA score (Eq.9), robustness simulation
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Synthetic demo data (no downloads)
python main.py toy

# 2. §3.1 distractor augmentation
python main.py augment --tiny --split train
python main.py augment --tiny --split dev

# 3. §3.2 EADPR training
python main.py train --tiny --epochs 5

# 4. Evaluation
python main.py retrieve   --tiny        # Top-k accuracy, MRR, R@k
python main.py aa         --tiny        # Answer-Awareness (Eq.9)
python main.py robustness --tiny        # Inject distractors into the corpus

# Verify the loss implementations
python tests/test_losses.py
```

`--tiny` skips pretrained weights and runs a **small randomly initialized model**. It exists
to check that the pipeline runs end to end; the numbers it produces mean nothing. Drop the
flag for the real setup, which downloads BERT-base and UnifiedQA-T5:

```bash
python main.py augment --split train --device cuda
python main.py train   --device cuda --epochs 40
```

### Data format

To use your own data, place three JSONL files under `--data-dir`.

`train.jsonl` / `dev.jsonl` — one question per line:

```json
{"qid": "q-0", "question": "...", "answers": ["..."],
 "positive_ctx": {"pid": "...", "title": "...", "text": "..."},
 "hard_negative_ctxs": [{"pid": "...", "title": "...", "text": "..."}],
 "supporting_pids": ["..."]}
```

`corpus.jsonl` — one `{"pid", "title", "text"}` per line.
`distractor_ctx` is filled in by `augment`. `supporting_pids` is used only for multi-hop R@k.

### Key options

| Option | Default | Corresponds to |
|---|---|---|
| `--lambda-distractor` | 1.0 | λ in Eq.7; the paper picks 1.0 from `{0.1, 0.2, 0.5, 0.9, 1.0}` |
| `--tau-hn` / `--tau-pp` | 1.0 / 1.0 | τ1 / τ2 in Eq.8, chosen by grid search |
| `--no-hn` / `--no-pp` | — | Drop that term (ablation) |
| `--no-qa-model` | — | Select distractors by a length heuristic instead of a QA model |
| `--threads` | 4 | CPU thread cap — **see the note below** |

Defaults follow Appendix C: BERT-base, 40 epochs, batch size 16, lr 2e-5,
Adam eps 1e-8 and betas (0.9, 0.999).

> **A note on `--threads`.** On a machine with many cores, thread overhead swamps the actual
> computation for small tensors. On the `--tiny` path here, 4 threads ran **over 100× faster**
> than 52 (about 14 s per batch down to well under 0.1 s). Keep the default at 4 and raise it
> only when you move to larger models.

## Scope

**Implemented** — §3.1 distractor augmentation (span removal plus QA-perplexity selection),
all three losses of §3.2, the DPR dual-encoder training loop, Top-k / MRR / R@k,
the AA score, the distractor-injection robustness test, and `--no-hn` / `--no-pp` ablations.

**Not implemented** — the following appear in the paper but not here:

- **Readers.** No end-to-end QA with the DPR extractive reader, FiD, or ELECTRA
  (Tables 3 and 5). This repository stops at the retriever.
- **Negative mining.** No BM25 or ANCE pipeline for mining hard negatives. The
  `hard_negative_ctxs` field is **consumed** if present, never produced.
- **MDR.** No multi-hop dense retrieval (Tables 4 and 5); only R@k over `supporting_pids`.
- **Real benchmarks.** NQ, TriviaQA, TREC, HotpotQA and the 21M-passage Wikipedia corpus are
  not included. The only data shipped here is the synthetic toy set.
- **FAISS index.** Search is exhaustive inner product — fine at toy scale, inadequate for a
  real corpus.

So this repository **does not reproduce the paper's numbers.** Its scope is the logical
structure of the method and the fact that it runs.

## Correctness

Since performance is not the goal, what is pinned down instead is that **the code matches the
equations** (`tests/test_losses.py`, 7 cases passing):

- A correctly ordered batch (`p+ > p* > p-`) incurs a lower loss than an inverted one
- `L_HN` matches the binary softmax of Eq.3 numerically
- At `λ=0`, `L_dpr` reduces exactly to vanilla DPR (Eq.1)
- Increasing λ increases `L_dpr` monotonically
- Gradients flow through all three losses (finite and non-zero)
- Adding hard negatives enlarges the `L_PP` denominator and raises the loss
- `--no-hn --no-pp` returns `L_eadpr` to exactly `L_dpr`

During training, the two inequalities are also logged every epoch as the fraction of the
batch that satisfies them: `acc_pos_over_dis` (Eq.2) and `acc_dis_over_neg` (Eq.4).

## Citation

```bibtex
@inproceedings{song-etal-2024-evidentiality,
    title = "Evidentiality-aware Retrieval for Overcoming Abstractiveness in Open-Domain Question Answering",
    author = "Song, Yongho and Lee, Dahyun and Jang, Myungha and Hwang, Seung-won and
              Lee, Kyungjae and Lee, Dongha and Yeo, Jinyoung",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    year = "2024",
    pages = "1930--1943",
}
```
