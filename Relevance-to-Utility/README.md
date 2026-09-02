
<!-- <h1 align="center">Relevance to Utility: Process-Supervised<br>Rewriting for RAG</a></h1> -->
<h1 align="center" style="display: flex; align-items: center; justify-content: center; gap: 12px;">
  <img src="figures/bridge_icon.png" alt="bridge_icon" height="40"/>
    <span>
    Relevance to Utility: Process-Supervised Rewrite for RAG
  </span>
</h1>


## 💡 Overview
**TL;DR:**  
We propose a new bridging method for RAG that rewrites retrieved documents to maximize answer generation utility, using LLM-guided process supervision and scalable distillation. Our method outperforms existing baselines across multiple QA benchmarks.



## 🔧 Installation

### 1. Environment Setup
```bash
# Create conda environment
conda create -n rtou python=3.9
conda activate rtou

# Install requirements
pip install -r requirements.txt
```

## 🏃 Quick Start

### Data Preparation

Use the code in `notebook/{dataset_name}.ipynb` to preprocess each dataset into our standardized JSON format. In this work, we utilize the datasets as follows:

- **Multi-hop QA:** HotpotQA, 2WikiMultihopQA, MuSiQue
- **Disambiguation QA:** AmbigQA
- Web corpus
  - **Single-hop QA:** MS MARCO
  - **Comprehensive QA:** CRAG

**Custom tasks:**  
For other generation tasks (e.g., QA, math, code), format your data as follows:

```json
{
  "Question": "your question here",
  "answer": ["answer1", "answer2"]
}
```

Also modify these scripts to support your task:
- `scripts/evaluate.py`: for task-specific evaluation
- `scripts/prompts.py`: to customize prompts for your task
- `scripts/run_xxx_xxx.py`: to define the end-to-end pipeline


### Model Inference

Please make sure the required file paths are correct before running the script.

1. **RAG**

- please give correct ```search_cache_name```.

```
bash runs/run_naive_rag.sh
```


## Training

1. **Generating Bridging Document Distribution**

```
bash runs/run_rewrite_docs.sh
```


2. **Training Student Model**

```
bash runs/convert_cache_to_train.sh
cd train
conda activate test
bash train/bash/run_train_rewriting.sh
cd ..
conda activate rtou
bash runs/convert_train_to_cache.sh
```

3. **Preference learning**
```
cd dpo-train
bash run.sh
cd ..
conda activate rtou
bash runs/convert_train_to_cache.sh
```

5. **Using the trained model to write docs**

```
bash runs/run_rewrite_docs_fromtrain.sh
```


## **baselines**
- Please check files in ```reranker/``` and ```runs/baselines/```


## Acknowlegements

We acknowledge this repository is based on [Search-o1](https://search-o1.github.io/).

This work was supported by Naver, Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)].
