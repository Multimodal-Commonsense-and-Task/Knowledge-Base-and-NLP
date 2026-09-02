#!/bin/bash
set -e

# train_path=$1
# output_path=$2
# batch_size=${3:-32}
# save_steps=${4:-20000}
# epochs=${5:-1}

ckpt="Qwen/Qwen3-Embedding-4B"
dataset=$1
embedding_dir="qwen3_trained/${dataset}/zs/embeddings_beir"

# prepare the query prefix according to the dataset
if [ "$dataset" == "trec-news" ]; then
  query_prefix="Instruct: Classify the fine-grained category of the given news title.\nTitle:"
elif [ "$dataset" == "robust04" ]; then
  query_prefix="Instruct: Retrieval the relevant passage for the given query.\nQuery:"
elif [ "$dataset" == "trec-covid" ]; then
  query_prefix="Instruct: Given a query on COVID-19, retrieve documents that answer the query.\nQuery:"
elif [ "$dataset" == "quora" ]; then
  query_prefix="Instruct: Given a question, retrieve questions that are semantically equivalent to the given question.\nQuestion:"
elif [ "$dataset" == "hotpotqa" ]; then
  query_prefix="Instruct: Given a multi-hop question, retrieve documents that can help answer the question.\nQuestion:"
else
  echo "Unknown dataset: $dataset"
  exit 1
fi

    # --dataset_name Tevatron/beir-corpus \
    # --dataset_config ${dataset} \
    # --dataset_split train \
mkdir -p $embedding_dir
pids=()
for s in $(seq -f "%02g" 0 3)
do
gpunum=$(($s))
if [ -f $embedding_dir/corpus_${dataset}.${s}.pkl ]; then
    echo "$embedding_dir/corpus_${dataset}.${s}.pkl already exists, skipping encoding."
    continue
fi
if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
  export CUDA_VISIBLE_DEVICES=${gpunum}; python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path ${ckpt} \
    --bf16 \
    --per_device_eval_batch_size 32 \
    --passage_max_len 512 \
    --pooling last \
    --padding_side left \
    --query_prefix "" \
    --dataset_path "datasets/${dataset}/corpus.jsonl" \
    --encode_output_path $embedding_dir/corpus_${dataset}.${s}.pkl \
    --dataset_number_of_shards 4 \
    --dataset_shard_index ${s} &
  pids+=($!)
else
  export CUDA_VISIBLE_DEVICES=${gpunum}; python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path ${ckpt} \
    --bf16 \
    --per_device_eval_batch_size 32 \
    --passage_max_len 512 \
    --pooling last \
    --padding_side left \
    --query_prefix "" \
    --dataset_name Tevatron/beir-corpus \
    --dataset_config ${dataset} \
    --dataset_split train \
    --encode_output_path $embedding_dir/corpus_${dataset}.${s}.pkl \
    --dataset_number_of_shards 4 \
    --dataset_shard_index ${s} &
  pids+=($!)
fi
done

exit_status=0
for pid in "${pids[@]}"; do
    wait "$pid"
    if [ $? -ne 0 ]; then
        exit_status=1
    fi
done

if [ $exit_status -ne 0 ]; then
    echo "One or more encoding processes failed. Exiting."
    exit 1
fi

if [ -f $embedding_dir/query_${dataset}.pkl ]; then
    echo "$embedding_dir/query_${dataset}.pkl already exists, skipping encoding."
else
  if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
    export CUDA_VISIBLE_DEVICES=${gpunum}; python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --model_name_or_path ${ckpt} \
      --bf16 \
      --per_device_eval_batch_size 32 \
      --normalize \
      --pooling last \
      --padding_side left \
      --query_prefix "$query_prefix" \
      --dataset_path "datasets/${dataset}/queries.jsonl" \
      --encode_output_path $embedding_dir/query_${dataset}.pkl \
      --query_max_len 512 \
      --encode_is_query
  else
    export CUDA_VISIBLE_DEVICES=${gpunum}; python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --model_name_or_path ${ckpt} \
      --bf16 \
      --per_device_eval_batch_size 32 \
      --normalize \
      --pooling last \
      --padding_side left \
      --query_prefix "$query_prefix" \
      --dataset_name Tevatron/beir \
      --dataset_config ${dataset} \
      --dataset_split test \
      --encode_output_path $embedding_dir/query_${dataset}.pkl \
      --query_max_len 512 \
      --encode_is_query
  fi
fi

if [ -f $embedding_dir/rank.${dataset}.txt ]; then
  echo "$embedding_dir/rank.${dataset}.txt already exists, skipping retrieval."
else
  echo "Starting retrieval for $dataset..."
  set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
      --query_reps $embedding_dir/query_${dataset}.pkl \
      --passage_reps $embedding_dir/corpus_${dataset}.*.pkl \
      --depth 100 \
      --batch_size 32 \
      --save_text \
      --save_ranking_to $embedding_dir/rank.${dataset}.txt
fi

if [ -f $embedding_dir/rank.${dataset}.trec ]; then
  echo "$embedding_dir/rank.${dataset}.trec already exists, skipping evaluation."
else
  python -m tevatron.utils.format.convert_result_to_trec --input $embedding_dir/rank.${dataset}.txt \
    --output $embedding_dir/rank.${dataset}.trec \
    --remove_query
fi

if [ -f $embedding_dir/eval.${dataset}.txt ]; then
  echo "$embedding_dir/eval.${dataset}.txt already exists, skipping evaluation."
else
  echo "Starting evaluation for $dataset..."
  python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset}-test $embedding_dir/rank.${dataset}.trec > $embedding_dir/eval.${dataset}.txt
fi

cat $embedding_dir/eval.${dataset}.txt