# Usage: sh 5_build_compressor_data.sh [task]
task=${1:-NQ}

data_dir=../data/labeler/$task

python build_compressor_data.py \
    --candidates $data_dir/train_candidates.json \
    --perturb $data_dir/train_perturb.json \
    --perturb_result $data_dir/signal/weak \
    --output_dir ../data/compressor/$task \
    --min_negatives 17 \
    --dev_ratio 0.2
