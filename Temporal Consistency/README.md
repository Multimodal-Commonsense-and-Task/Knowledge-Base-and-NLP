# Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models (ACL 2025 short)

## Overview

Despite the advanced capabilities of large language models (LLMs), their temporal reasoning ability remains underdeveloped. Prior works have highlighted this limitation, particularly in maintaining temporal consistency when understanding event relations. For example, models often confuse mutually exclusive temporal relations like “before” and “after” between events and make inconsistent predictions. In this work, we tackle the issue of temporal inconsistency in LLMs by proposing a novel counterfactual prompting approach. Our method generates counterfactual questions and enforces collective constraints, enhancing the model’s consistency. We evaluate our method on multiple datasets, demonstrating significant improvements in event ordering for explicit and implicit events and temporal commonsense understanding, by effectively addressing temporal inconsistencies.

## Instructions

Please refer to `README-timeqa.md` for instructions on running the code.

## Acknowledgments

This work was supported by Naver, the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. RS-2024-00414981), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data), and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)].
