# UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval (ACL Findings 2026)

## Overview

Unsupervised domain adaptation generalizes neural retrievers to an unseen domain by generating pseudo queries on target domain documents. The quality and efficiency of this adaptation critically depend on which documents are selected for pseudo query generation. The existing document sampling method focuses on diversity but fails to capture model uncertainty. In contrast, we propose Uncertainty-based Iterative Document Sampling (UnIte), addressing these limitations by (1) filtering documents with high aleatoric uncertainty and (2) prioritizing those with high epistemic uncertainty, maximizing the learning utility of the current model. We conducted extensive experiments on a large corpus of BEIR with small and large models, showing significant gains of +2.45 and +3.49 nDCG@10 with a smaller training sample size, 4k on average.

## Instructions

### Scripts folder contains the required preparation & train pipeline.
- bm25_entropy_auto.sh performs aleatoric uncertainty calculation for given corpus using bm25 index
- run_pipeline-{model}.sh performs the ours pipeline after aleatoric uncertainty calculation
- run_epochs-{model}.sh perform the train for specific case of model

### Example pipeline running for TREC-COVID dataset using DPR model
```
./scripts/bm25_entropy_auto.sh trec-covid
# This calculate the document level aleatoric uncertainty and outputs the best-k with automatic elbow detection, as reported the k is 3 for all dataset reported

./scripts/run_pipeline-dpr.sh trec-covid 3
# It runs the iterative sampling-training pipeline by calling run_pipeline-dpr-iter.sh
# At each iteration, calculating the embedding, epistemic uncertainty estimation, epistemic uncertainty sampling, query generation, hard negative mining, train the model is performed
# Also, it checks the early stopping criteria, if detected pipeline scripts quits.
```

## Acknowledgments

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. RS-2024-00414981), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-000077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data), and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)].
