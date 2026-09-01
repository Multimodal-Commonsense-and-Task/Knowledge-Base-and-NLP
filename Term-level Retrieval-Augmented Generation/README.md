# tRAG: Term-level Retrieval-Augmented Generation for Domain-Adaptive Retrieval (NAACL 2025)

## Overview

[NAACL 2025 paper](https://aclanthology.org/2025.naacl-long.334/)

tRAG generates a domain-level keyword vocabulary, retrieves and verifies terms relevant to each document, and uses the selected terms for domain-adaptive pseudo-query generation.

## Code

- `keyword-generation/`: domain keyword generation and prompts.
- `method/`: pseudo-query generation, cross-encoder filtering, collective verification, and GPL export.
- `baselines/llm-pqg/`: LLM pseudo-query generation baseline.
- `analysis/`: seen-term bias, unseen-term recall, and keyword hallucination analysis.

## Data

BEIR datasets are loaded as `beir/<dataset>` through `hamu_tool.dataset.DataLoader`.

Each component reads prompts and writes intermediate files under its own `data/<dataset>/` directory. Prompt files are provided for FiQA, NFCorpus, SciDocs, SciFact, Robust04, and TREC-COVID.

API credentials are read from `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY`.

## Acknowledgments

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. RS-2024-00414981).
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
