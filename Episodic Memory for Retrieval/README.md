# Beyond Markovian Forgetfulness: Episodic Memory for Reasoning-Intensive Retrieval (ACL 2026)

## Overview

Reasoning-intensive information retrieval uses large language models to solve complex queries via multi-step reasoning. However, existing methods have critical limitations. Chain-of-Thought (CoT) approaches suffer from inefficiency, while state-based methods, despite better token efficiency, often fall into reasoning cycles that trap the query refinement process. To address these issues, we propose Episodic Memory for Retrieval (EMR), which enhances the state-based framework with an episodic memory. This module stores the full history of prior states for a query, allowing the model to avoid repetition of such cycles. Experiments on the BRIGHT benchmark show that EMR consistently outperforms both CoT and state-based baselines. Moreover, it is highly token-efficient, reducing token usage by 72% on average. Our results show that episodic memory is an effective and token-efficient mechanism for reasoning-intensive retrieval. The gains also generalize across different base models and stay efficient in terms of end-to-end latency.

## Instructions

This repository now ships a small suite of retrieval runners built on top of the BRIGHT benchmark assets:

- `bm25`: A single-step sparse baseline powered by Pyserini BM25.
- `smr`: Sequential Memory Retrieval, an iterative agent without memory compression.
- `emr`: Episodic Memory Retrieval, which augments each step with minified document memories.

All methods share the same retriever, callback, and LLM infrastructure, and they can be mixed with BM25 or ReasonIR retrieval, OpenAI or local vLLM backends, and built-in pytrec_eval scoring.

## Requirements
- Python 3.12+
- GPU recommended for dense retrieval (ReasonIR) and vLLM. BM25 works on CPU.
- Java 11+ for Pyserini/Lucene (required for indexing and LuceneSearcher).
- If using OpenAI: an API key at `.env/openai_api_key.txt` (file must contain only the key).
- If using vLLM: a running server (default `http://localhost:8000`).

## Installation
You can use uv (recommended) or pip. The project includes `uv.lock` for reproducible installs.

spaCy English transformer model required by EMR (sentence splitting & NER)
```bash
python -m spacy download en_core_web_trf
```

## LLM backends
- OpenAI: Place your key in `.env/openai_api_key.txt`.
- vLLM: tested with Qwen models. Start a server on port 8000 (or change `--port`). Example:
```bash
# Example (adjust GPU flags as needed)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
	--model Qwen/Qwen3-32B-FP8 \
	--host 0.0.0.0 --port 8000
```

## Data and Indexes
This project uses the [BRIGHT benchmark](https://brightbenchmark.github.io/), a collection of information-seeking queries across various domains. The preprocessing script, `src/prepro.py`, handles downloading the necessary data and building the retrieval indexes.

### Download Datasets
First, use the preprocessing script to download the query files for a specific domain. The script will save the data to the `data/<domain>/` directory.

```bash
# Example: Download the dataset for the 'biology' domain
python -m src.prepro --dataset biology
```

The following domains are available for download and use:
  * `aops`
  * `biology`
  * `earth_science`
  * `economics`
  * `leetcode`
  * `pony`
  * `psychology`
  * `robotics`
  * `stackoverflow`
  * `sustainable_living`
  * `theoremqa_questions`
  * `theoremqa_theorems`

### Build Indexes
After downloading the data, the preprocessing script will **automatically build** the necessary indexes for retrieval. Both a BM25 (sparse) and a ReasonIR (dense) index will be created and stored under the `index/<domain>/` directory.

Since the indexing process is integrated into the preprocessing script, you do not need to run any separate commands. Simply running the download command above will prepare the data and the indexes required to run the EMR agent.

## Running the runners
The main entrypoint is `src/run.py`. It reads queries from `data/<dataset>/query.<type>.jsonl`, runs the selected method, writes agent histories and evaluation outputs, and aggregates metrics.

### Methods at a glance
- `bm25`: Retrieves once with Pyserini BM25 and logs the ranked list. Iterative options (`--max_steps`, etc.) are ignored.
- `smr`: Iterative agent that refines queries or reorders documents while keeping full document texts in the system prompt.
- `emr`: Iterative agent that compresses document memories by extracting salient sentences before feeding them back into the prompt.

### Common flags
- `--dataset`: Dataset to use (default: `biology`).
- `--query_type`: Variant of queries to use (default: `original`).
- `--method`: One of `bm25`, `smr`, `emr` (default: `emr`).
- `--retriever`: Sparse `bm25` or dense `reasonir` backends (default: `bm25`). Ignored by the `bm25` method.
- `--llm`: LLM name to hand off to `util.llm` (default: `qwen3`). Required for `smr` and `emr`.
- `--max_steps`: Maximum number of reasoning steps (default: 16). Only used by `smr` and `emr`.
- `--doc_topk`: Number of top documents to retrieve before filtering exclusions (default: 10).
- `--sent_topk`: Top sentences kept per document for EMR memory compression (default: 5). Only used by `emr`.
- `--init_temp`: Initial sampling temperature for the agent LLM (default: 0.1).
- `--bm25_k1` / `--bm25_b`: BM25 hyper-parameters for both the standalone `bm25` runner and the BM25 retriever option (defaults: 0.9 / 0.2).
- `--idx`: Index of the current run, used to de-duplicate log filenames (default: 0).
- `--port`: Port for the local vLLM/OpenAI-compatible endpoint (default: 8000).

### Examples
#### BM25 baseline
```bash
python -m src.run \
    --dataset biology \
    --query_type original \
    --method bm25 \
    --doc_topk 10 \
    --bm25_k1 0.9 --bm25_b 0.2 \
    --idx 0
```

#### SMR with local vLLM + BM25
```bash
python -m src.run \
    --dataset biology \
    --query_type original \
    --method smr \
    --retriever bm25 \
    --llm qwen3 \
    --doc_topk 10 \
    --idx 0 --port 8000
```

#### EMR with OpenAI + ReasonIR

```bash
python -m src.run \
    --dataset biology \
    --query_type gpt4 \
    --method emr \
    --retriever reasonir \
    --llm gpt-4o \
    --doc_topk 10 --sent_topk 5 \
    --idx 1
```

## Notes
- spaCy model: `en_core_web_trf` is required for sentence segmentation. Install once via `python -m spacy download en_core_web_trf`.
- Java requirement: Pyserini needs a working Java 11+ runtime for Lucene search and indexing.
- Dense retriever: `reasonir/ReasonIR-8B` and Faiss indices require significant memory and benefit from GPU. For CPU‑only environments stick to `bm25`.
- OpenAI security: keep your key only in `.env/openai_api_key.txt`. Never commit it.

## Troubleshooting
- ModuleNotFoundError: en_core_web_trf
	- Run: `python -m spacy download en_core_web_trf`
- Java/Pyserini errors (NoClassDefFoundError, etc.)
	- Install Java 11+ and ensure `java` is on PATH.
- Connection refused to `localhost:8000`
	- Start vLLM server or switch to an OpenAI model.
- CUDA/FAISS issues on dense indexing
	- Use `bm25` retriever or install a CPU Faiss build; the default dep is GPU‑oriented (`faiss-gpu-cu12`).
- Duplicate run index error
	- Change `--idx` (the runner prevents overwriting logs using this index).

## License
This project is provided for research purposes.

## Dependencies
- BRIGHT benchmark (xlangai/BRIGHT)
- Pyserini/Lucene
- Sentence-Transformers and Cross-Encoders
- Faiss
- vLLM
- OpenAI API

## Acknowledgments

This work was supported by the InnoCORE program of the Ministry of Science and ICT (N10250156), and by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
