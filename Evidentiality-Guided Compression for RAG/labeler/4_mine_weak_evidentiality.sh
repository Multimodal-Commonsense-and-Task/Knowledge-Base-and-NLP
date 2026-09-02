# Usage: sh 4_mine_weak_evidentiality.sh [task] [num_shards]
# Condition 2: does the sentence interfere with the evidence?
task=${1:-NQ}
num_shards=${2:-1}

data_dir=../data/labeler/$task

for shard in $(seq 0 $((num_shards - 1))); do
    CUDA_VISIBLE_DEVICES=$shard python run_llm.py \
        --eval_data $data_dir/train_perturb.json \
        --output_path $data_dir/signal/weak \
        --n_context 5 \
        --shard_id $shard \
        --num_shards $num_shards &
done
wait
