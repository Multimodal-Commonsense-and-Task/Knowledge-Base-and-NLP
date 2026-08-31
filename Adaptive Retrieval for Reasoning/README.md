# Adaptive Retrieval for Reasoning (ACL 2026)

## Overview
We study leveraging adaptive retrieval to ensure sufficient “bridge” documents are retrieved for reasoning-intensive retrieval. Bridge documents are those that contribute to the reasoning process yet are not directly relevant to the initial query. While existing reasoning-based reranker pipelines attempt to surface these documents in ranking, they suffer from bounded recall. Naive solution with adaptive retrieval into these pipelines often leads to planning error propagation. To address this, we propose REPAIR, a framework that bridges this gap by repurposing reasoning plans as dense feedback signals for adaptive retrieval. Our key distinction is enabling mid-course correction during reranking through selective adaptive retrieval, retrieving documents that support the pivotal plan. Experimental results on reasoning-intensive retrieval and complex QA tasks demonstrate that our method outperforms existing baselines by 5.6%pt.

## Setup
- For inference,
```
pip install -r requirements.txt
```

- For training,
```
cd shortcut_reranker/train
pip install -r requirements.txt
```


## Training
- Train data generation
```
bash bash/train_sft.sh
```

## Graph construction

- Graph construction
```
bash bash/construct_graph.sh
```

## Inference

- Inference (e.g. BRIGHT)
```
bash bash/rank_bright.sh
```

## Evaluation
- Evaluation (e.g. BRIGHT)
```
bash bash/eval_bright.sh
```

## Acknowledgments
This work was supported by LG AI Research, Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)].
