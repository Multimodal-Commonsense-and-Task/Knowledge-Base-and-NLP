# Training

## Installation

```
conda create -n test python=3.10
pip install torch torchvision torchaudio —index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Training

To train the rewriter model, run:
```
train/bash/run_train_rewriting.sh
```



## Acknowlegement

We acknowledge this repository is based on the paper [FIRST: Faster Improved Listwise Reranking with Single Token Decoding](https://arxiv.org/pdf/2406.15657)