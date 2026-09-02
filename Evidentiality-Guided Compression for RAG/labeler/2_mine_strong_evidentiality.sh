# Usage: sh 2_mine_strong_evidentiality.sh [task] [num_shards]
# Condition 1: the LLM fails without the sentence but succeeds with it.
task=${1:-NQ}
num_shards=${2:-1}

data_dir=../data/labeler/$task

# (a) closed-book: does the LLM already know the answer without any evidence?
for shard in $(seq 0 $((num_shards - 1))); do
    CUDA_VISIBLE_DEVICES=$shard python run_llm.py \
        --eval_data $data_dir/train_per_sentence.json \
        --output_path $data_dir/signal/closed_book \
        --n_context 0 \
        --shard_id $shard \
        --num_shards $num_shards &
done
wait

# (b) one sentence at a time: does this single sentence let the LLM answer?
for shard in $(seq 0 $((num_shards - 1))); do
    CUDA_VISIBLE_DEVICES=$shard python run_llm.py \
        --eval_data $data_dir/train_flat.json \
        --output_path $data_dir/signal/strong \
        --n_context 1 \
        --shard_id $shard \
        --num_shards $num_shards &
done
wait
