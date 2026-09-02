# RaDA: Retrieval-augmented Web Agent Planning with LLMs (ACL Findings 2024)

## Overview

Agents powered by large language models (LLMs) inherit important limitations, such as the restricted context length, dependency on human-engineered exemplars (e.g., for task decomposition), and insufficient generalization. To address these challenges, we propose RaDA, a novel planning method for Web agents that does not require manual exemplars, efficiently leverages the LLMs’ context, and enhances generalization. RaDA disentangles planning into two stages: for a new given task, during Retrieval-augmented Task Decomposition (RaD), it decomposes tasks into high-level subtasks; next, during Retrieval-augmented Action Generation (RaA), it traverses the trajectory obtained with RaD to iteratively synthesize actions based on dynamically retrieved exemplars. We compare RaDA with strong baselines covering a broad space of design choices, using both GPT-3.5 and GPT-4 as backbones; and we find consistent improvements over previous SOTA in two challenging benchmarks, CompWoB and Mind2Web, covering settings with different complexities. We show the contributions of RaDA via ablation studies and qualitative analysis; and we discuss the structural benefits of our more compositional design.

## Acknowledgments

This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) [NO. 2021-0-01343, Artificial Intelligence Graduate School Program (Seoul National University)] and Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) [NO. 2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data].

This codebase is largely based on Synapse (https://github.com/ltzheng/Synapse).