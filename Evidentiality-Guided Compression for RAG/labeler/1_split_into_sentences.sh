# Usage: sh 1_split_into_sentences.sh [task]
task=${1:-NQ}

data_dir=../data/labeler/$task

# training split: keep only answer-containing sentences, they are the label candidates.
# (there is no separate dev retrieval file; dev is split off in step 5)
python split_into_sentences.py \
    --input_data $data_dir/train.json \
    --output_dir $data_dir \
    --split train \
    --mode label \
    --max_sents_per_question 16

# test split: keep every sentence, this is what the compressor scores at inference time
python split_into_sentences.py \
    --input_data $data_dir/test.json \
    --output_dir $data_dir \
    --split test \
    --mode compress

mkdir -p ../data/compressor/$task
cp $data_dir/test_flat.json ../data/compressor/$task/test.json
