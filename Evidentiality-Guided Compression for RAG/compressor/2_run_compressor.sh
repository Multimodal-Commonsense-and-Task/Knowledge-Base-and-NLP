# Usage: sh 2_run_compressor.sh [task]
# Scores every sentence of the top-100 retrieved documents and sorts them by
# evidentiality, producing the ordered evidence d'_1, ..., d'_|D| for the test set.
task=${1:-NQ}
ckpt=../checkpoints/compressor/$task/checkpoint/latest/checkpoint.pth

mkdir -p ../data/evaluator/$task

python run_compressor.py \
    --model_path facebook/contriever \
    --weight_path $ckpt \
    --eval_data ../data/compressor/$task/test.json \
    --eval_normalize_text \
    --per_gpu_eval_batch_size 16 \
    --do_eval \
    --output_score_path ./output/$task/test

# wrap the scores back into the data and sort each question's sentences
python aggregate_compressor_scores.py \
    --input_score_path ./output/$task/test/score.json \
    --input_data_path ../data/compressor/$task/test.json \
    --output_path ../data/evaluator/$task/test.json
