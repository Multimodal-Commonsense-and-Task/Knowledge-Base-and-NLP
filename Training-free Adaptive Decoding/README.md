# Smarter, Not Harder: Training-Free Adaptive Computation for Transformers (ACL Findings 2025)

## Overview

Adaptive Computation in Transformers (ACT) has been pursued in two directions: efficiency- and performance-focused. We study performance-focused ACT, or PACT, which invests more computation on hard steps to improve performance, such as by adding forward passes. We first discuss beam search and hesitation-based methods as PACT and their limitations. While the hesitation-based approach outperforms beam search by perturbing input embeddings, it suffers from inefficiency due to invalidating KVCache and exhibits instability due to its reliance on randomness. To address this, we propose IMPACT, a novel PACT method that perturbs network weights rather than input embeddings. This approach enables the reuse of KVCache, offers deterministic predictions, and significantly improves memory and computational efficiency. By achieving a better balance between performance and efficiency, IMPACT makes PACT accessible to communities with consumer-grade hardware.

## Prerequisites

```
pip install -e .
pip install wandb sentencepiece bitsandbytes datasets flash-attn

git clone https://github.com/FasterDecoding/TEAL.git
cd TEAL
pip install -e .
CUDA_VISIBLE_DEVICES=0 python teal/grab_acts.py \  
  --model_name ../Meta-Llama-3.1-8B-Instruct \ 
  --output_path Meta-Llama-3.1-8B-Instruct-TEAL
cd ..

```

## Evaluate original
```
lm_eval --model hf     --model_args pretrained=../Meta-Llama-3.1-8B-Instruct,dtype=float16,attn_implementation=flash_attention_2     --tasks gsm8k     --batch_size 1
```

## Evaluate HARP
```
lm_eval --model hf     --model_args pretrained=../Meta-Llama-3.1-8B-Instruct,harp_dropout_rate=0.2,dtype=float16,attn_implementation=flash_attention_2     --tasks gsm8k     --batch_size 1
```

## Evaluate HARP+ (w/ contextual sparsity)

```
lm_eval --model hf --model_args pretrained=../Meta-Llama-3.1-8B-Instruct,harp_last_token_only=True,dtype=float16,attn_implementation=flash_attention_2,teal_path=../teal/Meta-Llama-3.1-8B-Instruct-TEAL,harp_sparsity=0.3,harp_sparsify_mlp=True,harp_sparsify_attn=True,harp_entropy_threshold=1 --tasks gsm8k --batch_size 1
```

## Evaluate Beam Search
```
lm_eval --model hf     --model_args pretrained=../Meta-Llama-3.1-8B-Instruct,dtype=float16,attn_implementation=flash_attention_2     --tasks gsm8k --gen_kwargs length_penalty=0.6,num_beams=3     --batch_size 1
```

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00414981), and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
