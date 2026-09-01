#!/bin/bash
set -e
ckpt=$1
dataset=$2
embedding_dir=${ckpt}/embeddings_beir
run_name=$3

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

export WANDB_PROJECT=${WANDB_PROJECT:-"unite"}

    # --dataset_name Tevatron/beir-corpus \
    # --dataset_config ${dataset} \
    # --dataset_split train \
mkdir -p $embedding_dir
pids=()
num_gpu=8
for s in $(seq -f "%02g" 0 7)
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
    --tokenizer_name "Qwen/Qwen3-Embedding-4B" \
    --bf16 \
    --per_device_eval_batch_size 32 \
    --passage_max_len 512 \
    --pooling last \
    --padding_side left \
    --query_prefix "" \
    --dataset_path "datasets/${dataset}/corpus.jsonl" \
    --encode_output_path $embedding_dir/corpus_${dataset}.${s}.pkl \
    --dataset_number_of_shards $num_gpu \
    --dataset_shard_index ${s} &
  pids+=($!)
else
  export CUDA_VISIBLE_DEVICES=${gpunum}; python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path ${ckpt} \
    --tokenizer_name "Qwen/Qwen3-Embedding-4B" \
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
    --dataset_number_of_shards $num_gpu \
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
      --tokenizer_name "Qwen/Qwen3-Embedding-4B" \
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
      --tokenizer_name "Qwen/Qwen3-Embedding-4B" \
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
      --batch_size 64 \
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

if [ -f $embedding_dir/metrics.json ]; then
  echo "Evaluation file $embedding_dir/metrics.json exist. Exiting."
else
  # Method 1: CLI를 사용한 WandB 로깅 (wandb 명령어 사용)
  echo "Uploading results to WandB..."

  # 결과를 파싱하여 JSON 형태로 변환
  recall_100=$(grep "recall_100" $embedding_dir/eval.${dataset}.txt | awk '{print $3}')
  ndcg_cut_10=$(grep "ndcg_cut_10" $embedding_dir/eval.${dataset}.txt | awk '{print $3}')

  # WandB에 메트릭을 로깅 (CLI 방식)
  echo "{\"${dataset}_recall_100\": $recall_100, \"${dataset}_ndcg_10\": $ndcg_cut_10}" > $embedding_dir/metrics.json

  # WandB API를 통한 업로드 (Python 한 줄 실행)
  python -c "
import wandb
import json

with open('$embedding_dir/metrics.json', 'r') as f:
    metrics = json.load(f)

run = wandb.init(
    project='$WANDB_PROJECT', 
    name='$run_name',
    resume='allow',
    id='$run_name'
)
wandb.log(metrics)
wandb.log({'dataset': '$dataset', 'model_checkpoint': '$ckpt'})
wandb.finish()
print('✅ Successfully uploaded metrics to WandB!')
"

  cat $embedding_dir/metrics.json
fi
