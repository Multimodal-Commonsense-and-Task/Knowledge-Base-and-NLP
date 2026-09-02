# Usage: sh 3_score_labeled_data.sh [task]
# Adds the compressor's r_score to the labeled train/dev data and sorts by it, so the
# evaluator is trained on the ranking it will see at inference time.
task=${1:-NQ}
ckpt=../checkpoints/compressor/$task/checkpoint/latest/checkpoint.pth

mkdir -p ../data/evaluator/$task

for split in train dev; do
    python score_labeled_data.py \
        --model_path facebook/contriever \
        --weight_path $ckpt \
        --input_data ../data/compressor/$task/$split.json \
        --output_path ../data/evaluator/$task/$split.json
done
