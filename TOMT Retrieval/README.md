# Intended Target Identification for Anomia Patients with Gradient-based Selective Augmentation (EMNLP Findings 2024)

## Overview
In this study, we investigate the potential of language models (LMs) in aiding patients experiencing anomia, a difficulty identifying the names of items. Identifying the intended target item from patient’s circumlocution involves the two challenges of term failure and error. (1) The terms relevant to identifying the item remain unseen. (2) What makes the challenge unique is inherent perturbed terms by semantic paraphasia, which are not exactly related to the target item, hindering the identification process. To address each, we propose robustifying the model from semantically paraphasic errors and enhancing the model with unseen terms with gradient-based selective augmentation (GradSelect). Specifically, the gradient value controls augmented data quality amid semantic errors, while the gradient variance guides the inclusion of unseen but relevant terms. Due to limited domain-specific datasets, we evaluate the model on the Tip of the Tongue dataset as an intermediary task and then apply our findings to real patient data from AphasiaBank. Our results demonstrate strong performance against baselines, aiding anomia patients by addressing the outlined challenges.


## Results

### ours
#### BOOK
recall_1: 0.133, (0.3396)
recall_10: 0.3562, (0.4789)
recip_rank: 0.2141, (0.3388)

#### MOVIE
recall_1: 0.1114, (0.3147)
recall_10: 0.2883, (0.453)
recip_rank: 0.1707, (0.3166)


#### MOVIE w/ plot in query
recall_1: 0.1171, (0.3216)
recall_10: 0.3036, (0.4598)
recip_rank: 0.1795, (0.3223)


#### MOVIE w/o plot in query
recall_1: 0.033, (0.1786)
recall_10: 0.0769, (0.2665)
recip_rank: 0.0498, (0.1849)


### reported
BOOKS 0.1974 (0.3981) 0.4206 (0.4937) 0.2783 (0.3835)
MOVIES 0.1285 (0.3347) 0.3180 (0.4657) 0.1938 (0.3343)


## Acknowledgments
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)], and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).

This repository is largely based on https://github.com/samarthbhargav/tomt-data.