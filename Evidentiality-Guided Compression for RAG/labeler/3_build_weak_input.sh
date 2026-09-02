# Usage: sh 3_build_weak_input.sh [task]
task=${1:-NQ}

data_dir=../data/labeler/$task

python build_weak_input.py \
    --per_sentence $data_dir/train_per_sentence.json \
    --flat $data_dir/train_flat.json \
    --closed_book_result $data_dir/signal/closed_book \
    --strong_result $data_dir/signal/strong \
    --output_dir $data_dir \
    --split train \
    --num_distractors 4
