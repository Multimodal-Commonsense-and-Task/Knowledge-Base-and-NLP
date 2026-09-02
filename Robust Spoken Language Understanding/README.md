# Interventional Speech Noise Injection for ASR Generalizable Spoken Language Understanding (EMNLP 2024)

## Overview

Recently, pre-trained language models (PLMs) have been increasingly adopted in spoken language understanding (SLU). However, automatic speech recognition (ASR) systems frequently produce inaccurate transcriptions, leading to noisy inputs for SLU models, which can significantly degrade their performance. To address this, our objective is to train SLU models to withstand ASR errors by exposing them to noises commonly observed in ASR systems, referred to as ASR-plausible noises. Speech noise injection (SNI) methods have pursued this objective by introducing ASR-plausible noises, but we argue that these methods are inherently biased towards specific ASR systems, or ASR-specific noises. In this work, we propose a novel and less biased augmentation method of introducing the noises that are plausible to any ASR system, by cutting off the non-causal effect of noises. Experimental results and analyses demonstrate the effectiveness of our proposed methods in enhancing the robustness and generalizability of SLU models against unseen ASR systems by introducing more diverse and plausible ASR noises in advance.

## Instructions

This is the code and instruction for the paper Interventional Speech Noise Injection for ASR Generalizable Spoken Language Understanding.
Our code is based on the code of the paper ASR error correction with constrained decoding on operation prediction. 

1. Clone the code from the github https://github.com/yangjingyuan/ConstDecoder.
2. Place the codes in this directory identical to the cloned code.
3. Make phoneme label with `phoneme_softlabel_clean.ipynb`
4. Train SNI model with train_constrained_noisygen_phoneme.py
5. Run generation with `phoneme_random_phoneme_generate.py`

## Acknowledgments

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)] and Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT)(No. 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
