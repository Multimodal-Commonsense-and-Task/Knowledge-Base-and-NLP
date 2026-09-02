# Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity

**Yeongbin Seo,  Gayoung Kim, Jaehyung kim †, Jinyoung Yeo †**      ( †  indicates advisors)

Presented at **ICLR 2026** main track

---

### abstract

As large language models (LLMs) are pretrained on massive web corpora, careful selection of data becomes essential to ensure effective and efficient learning. While perplexity (PPL)-based filtering has shown strong performance, it suffers from drawbacks: substantial time costs and inherent unreliability of the model when handling noisy or out-of-distribution samples. In this work, we propose a simple yet powerful alternative: a \textbf{prior-based data filtering} method that estimates token priors using corpus-level term frequency statistics, inspired by linguistic insights on word roles and lexical density. Our approach filters documents based on the mean and standard deviation of token priors, serving as a fast proxy to PPL while requiring no model inference. Despite its simplicity, the prior-based filter achieves the highest average performance across 21 downstream benchmarks, while reducing time cost by over \textbf{1000×} compared to PPL-based filtering. We further demonstrate its applicability to symbolic languages such as code and math, and its dynamic adaptability to multilingual corpora without supervision.

---

### Issue

**Clarification on the meaning of “document”**  (2026.05.24)

A reader pointed out that the term “document” in the paper may be misleading. In the paper, the term “document” was intended to mean a text data point $d$, as defined in Section 2.1. More precisely, a data point is a fixed-length token sequence $d = (x_1, x_2, \ldots, x_N)$, $d \in D$, where each $x_i$ is a token and $D$ is the dataset. This reflects the common practice in the pretraining stage of using fixed-length text blocks as data points. Our method aims to filter out noisy data points from $D$ before training. 

---

## Installation

```coq

$ conda create --name prior python=3.9
$ conda activate prior
$ pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

$ pip install datasets==2.18.0 einops==0.7.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 

$ pip install notebook==7.1.1 nvitop==1.3.2 omegaconf==2.3.0 packaging==23.2 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 timm==0.9.16 transformers==4.38.2 triton==2.2.0 wandb==0.13.5 

$ pip install bitsandbytes==0.42.0 git+https://github.com/huggingface/peft.git 
```

---

### Step1. Prepare datasets

You can donwload **Dolma** from  https://huggingface.co/datasets/allenai/dolma

After downloading, tokenize and split dataset via the following command.

```bash
python s1_tokenize_chunk.py
```

### Step2. Assess token priors. Calculate $\mu_d, \sigma_d$ , PPL  for datapoints.

Assess token priors and $\mu_d, \sigma_d$ with the following command.

```bash
python s2_prior.py
```

Assess PPL with the following command. The reference model should be trained using the  `s4_pretrain.py` , with the arguments similar to that described in Step 4.  As explained in the paper, the model size is 137M (equivalent to the gpt2-small architecture), and the ``no_filter'' indices must be used.

```bash
python do4_ppl.py \
dataset=dolma_flatten-train \
dataset_path=dataset/dolma_flatten/dolma_flatten-train_train_bs512_wrapped.dat \
batch_size=32 \
+model_name=outputs/gpt2_small_lr2e-4_dolma_no_filter/cp_ep1_gstep20000
```

### Step3. Build filtered indices

Build filtered indices in the following ipynb scripts. The result files are boolean type torch file, where *True* indicates selected.

`s3_prior_filter.ipynb`

### Step4. Pretrain models

Pretrain with dataset and the filtered indices.

```bash
python s4_pretrain.py \
data.train=dolma_flatten-train \
data.valid=dolma_flatten-valid \
model.length=512 \
wandb.name=gpt2_xl_lr2e-4_dolma_merge \
optim.lr=2e-4 \
loader.global_batch_size=256 \
lr_scheduler.num_warmup_steps=500 \
loader.batch_size=1 \
trainer.val_check_interval=5000 \
data.cache_dir=./dataset/dolma_flatten \
+base_model_name=gpt2-xl \
+output_dir=./outputs \
+filter_file=./filtered_indices/prior_filter/dolma_flatten-train_merged.pt
```

You can choose filter with the argument `+filter_file` 

### Step5. Evaluation

We use Mosaic gauntlet for evaluation.  (https://github.com/mosaicml/llm-foundry/blob/main/scripts/eval/local_data/EVAL_GAUNTLET.md )

---

## Other baselines

We also provide code for other baselines, e.g., DSIR.

### DSIR

The DSIR code is a modified version of the code provided in [1] (https://github.com/p-lambda/dsir).

Run the following script.

```bash
python run_dsir.py \
  --dataset_name dolma_flatten-train \
  --num_shards 129 \
  --raw_path_template "/dataset/dolma/dolma_flatten-train/data-{subset}-of-00129.arrow" \
  --target_path "path/to/wiki+pile/target.arrow" \
  --cache_dir "./cache" \
  --mid_output_dir "./resampled" \
  --final_output_path "./filtered_indices/dolma_flatten-train_DSIR.pt" \
  --num_to_sample 10000000
```

---

## References

[1] Xie, Sang Michael, et al. "Data selection for language models via importance resampling." *Advances in Neural Information Processing Systems* 36 (2023)
