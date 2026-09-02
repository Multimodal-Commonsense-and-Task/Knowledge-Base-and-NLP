# Usage: sh 1_train_compressor.sh [task] [total_steps]
# Trains the dual-encoder compressor on the evidentiality labels:
#   positive_ctxs = strong evidence, hard_negative_ctxs = weak evidence,
#   negative_ctxs = distractor.
task=${1:-NQ}
total_steps=${2:-15812}   # 15812 for NQ, 5832 for TQA

python train_compressor.py \
    --model_path facebook/contriever \
    --train_data ../data/compressor/$task/train.json \
    --eval_data ../data/compressor/$task/dev.json \
    --eval_normalize_text \
    --per_gpu_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --eval_freq 100 \
    --save_freq 100 \
    --output_dir ../checkpoints/compressor/$task \
    --do_train \
    --lr 5e-5 \
    --total_steps $total_steps \
    --negative_ctxs 7 \
    --negative_hard_ratio 0.15
