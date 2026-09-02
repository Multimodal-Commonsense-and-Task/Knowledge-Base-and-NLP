# Query Variant Detection Using Retriever as Environment (NAACL 2025 industry)

## Overview

This paper addresses the challenge of detecting query variants—pairs of queries with identical intents. One application in commercial search engines is reformulating user queries with its variant online. While measuring pairwise query similarity has been an established standard, it often falls short of capturing semantic equivalence when word forms or order differ. We propose leveraging the retrieval as an environment feedback (EF), based on the premise that desirable retrieval outcomes from equivalent queries should be interchangeable. Experimental results on both proprietary and public datasets demonstrate the efficacy of the proposed method, both with and without LLM calls.


## Included

- `scripts/build_paws_qqp.py`: reconstruct PAWS-QQP from QQP and the PAWS index
- `scripts/rewrite_queries.py`: GPT query rewriting
- `scripts/retrieve_documents.py`: Google Custom Search + `trafilatura`
- `scripts/build_features.py`: SBERT/GTE QQ, QD, DD features
- `scripts/train_verifier.py`: lightweight MLP training and evaluation
- `scripts/run_llm_verifier.py`: LLM-only and LLM+EF experiments
- `scripts/evaluate_predictions.py`: accuracy and positive-label F1
- `data/manual/`: released 50k train / 50k test manual annotations
- `data/paws_indices/dev_and_test.tsv`: PAWS-QQP evaluation index

The proprietary search corpus, click logs, and internal outputs are not included.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Set credentials only as environment variables:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_CSE_API_KEY="..."
export GOOGLE_CSE_ID="..."
```

## PAWS-QQP data

Download `quora_duplicate_questions.tsv` yourself from the original
[Quora Question Pairs release](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
and place it at `data/qqp/quora_duplicate_questions.tsv`.

Then build the evaluation split:

```bash
python scripts/build_paws_qqp.py \
  --original-qqp data/qqp/quora_duplicate_questions.tsv \
  --paws-index data/paws_indices/dev_and_test.tsv \
  --output data/paws_qqp/dev_and_test.tsv
```

The copied workspace did not contain the PAWS-QQP train index. To retrain the
PAWS lightweight model, place the official index at
`data/paws_indices/train.tsv` and run the same command with that path.

## Public experiment flow

```bash
python scripts/rewrite_queries.py \
  --input data/paws_qqp/dev_and_test.tsv \
  --output artifacts/test.rewritten.jsonl \
  --model gpt-4o

python scripts/retrieve_documents.py \
  --input artifacts/test.rewritten.jsonl \
  --output artifacts/test.search.jsonl

python scripts/build_features.py \
  --input artifacts/test.search.jsonl \
  --output artifacts/test.features.npz \
  --trust-remote-code
```

After producing train and test features:

```bash
python scripts/train_verifier.py \
  --train artifacts/train.features.npz \
  --test artifacts/test.features.npz \
  --feature-set qq_qd_dd \
  --output artifacts/verifier.pt
```

LLM experiments:

```bash
python scripts/run_llm_verifier.py \
  --input data/paws_qqp/dev_and_test.tsv \
  --output artifacts/llm_only.jsonl \
  --mode plain --model gpt-4o-mini

python scripts/run_llm_verifier.py \
  --input artifacts/test.search.jsonl \
  --output artifacts/llm_with_ef.jsonl \
  --mode ef --model gpt-4o-mini

python scripts/evaluate_predictions.py --input artifacts/llm_with_ef.jsonl
```

The feature layout is `1 QQ + 40 QD + 100 DD` for top-10 retrieval. Training
defaults match the paper: Adam, learning rate `1e-4`, weight decay `1e-4`,
StepLR `(10, 0.5)`, 100 epochs, and batch size 2048.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (No. RS-2024-00414981), and by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
