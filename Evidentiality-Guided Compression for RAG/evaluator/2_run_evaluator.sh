# Usage: sh 2_run_evaluator.sh [task] [checkpoint]
# Evidentiality reflection: start from the top-ranked evidence and keep adding until the
# evaluator judges the collective evidential. max_iters x sents_per_iter is the token
# limit that stops the loop (5 x 4 = 20 sentences in the paper).
task=${1:-NQ}
ckpt=${2:-../checkpoints/evaluator/$task}

mkdir -p ../data/reader/$task

python run_evaluator.py \
    --eval_data ../data/evaluator/$task/test.json \
    --output_path ../data/reader/$task/test.json \
    --weight_path $ckpt \
    --max_iters 5 \
    --sents_per_iter 4 \
    --threshold 0.7
