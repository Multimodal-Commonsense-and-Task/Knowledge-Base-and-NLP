# Usage: sh 1_train_evaluator.sh [task]
# Distills the evidentiality labels into Flan-T5-large: <EVI> for strong evidence,
# <NOT> otherwise. --hardneg takes the top-scoring negatives so the evaluator learns to
# separate genuine evidence from sentences that merely look relevant.
task=${1:-NQ}

python train_evaluator.py \
    --train_file ../data/evaluator/$task/train.json \
    --val_file ../data/evaluator/$task/dev.json \
    --save_path ../checkpoints/evaluator/$task \
    --model_path google/flan-t5-large \
    --batch_size 5 \
    --num_epochs 4 \
    --lr 1e-5 \
    --seed 42 \
    --pos_key positive_ctxs \
    --neg_key negative_ctxs \
    --pos_cnt 4 \
    --neg_cnt 12 \
    --eval_steps 1000 \
    --max_length 1024 \
    --gradient_accumulation_steps 8 \
    --hardneg
