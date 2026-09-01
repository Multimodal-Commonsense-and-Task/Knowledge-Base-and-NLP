#!/bin/bash
set -e

ckpt="Luyu/co-condenser-marco-retriever"
dataset=$1
tokenizer="Luyu/co-condenser-marco-retriever"
embedding_dir="cocondenser_trained/${dataset}/zs/embeddings_beir"

mkdir -p $embedding_dir
for s in $(seq -f "%02g" 0 7)
do
if [ -f $embedding_dir/corpus_${dataset}.${s}.pkl ]; then
    echo "$embedding_dir/corpus_${dataset}.${s}.pkl already exists, skipping encoding."
    continue
fi

# if dataset is closed set then provide the local path to the corpus
if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
  python -m tevatron.driver.encode \
    --output_dir=temp \
    --model_name_or_path ${ckpt} \
    --tokenizer_name ${tokenizer} \
    --fp16 \
    --per_device_eval_batch_size 64 \
    --p_max_len 512 \
    --dataset_name Tevatron/beir-corpus:${dataset} \
    --corpus_path "datasets/${dataset}/corpus.jsonl" \
    --encoded_save_path $embedding_dir/corpus_${dataset}.${s}.pkl \
    --encode_num_shard 8 \
    --encode_shard_index ${s}
else
  python -m tevatron.driver.encode \
    --output_dir=temp \
    --model_name_or_path ${ckpt} \
    --tokenizer_name ${tokenizer} \
    --fp16 \
    --per_device_eval_batch_size 64 \
    --p_max_len 512 \
    --dataset_name Tevatron/beir-corpus:${dataset} \
    --encoded_save_path $embedding_dir/corpus_${dataset}.${s}.pkl \
    --encode_num_shard 8 \
    --encode_shard_index ${s}
fi
done

if [ -f $embedding_dir/query_${dataset}.pkl ]; then
    echo "$embedding_dir/query_${dataset}.pkl already exists, skipping encoding."
else
  if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
    python -m tevatron.driver.encode \
      --output_dir=temp \
      --model_name_or_path ${ckpt} \
      --tokenizer_name ${tokenizer} \
      --fp16 \
      --per_device_eval_batch_size 64 \
      --dataset_name Tevatron/beir:${dataset}/test \
      --query_path "datasets/${dataset}/queries.jsonl" \
      --encoded_save_path $embedding_dir/query_${dataset}.pkl \
      --q_max_len 512 \
      --encode_is_qry
  else
  python -m tevatron.driver.encode \
    --output_dir=temp \
    --model_name_or_path ${ckpt} \
    --tokenizer_name ${tokenizer} \
    --fp16 \
    --per_device_eval_batch_size 64 \
    --dataset_name Tevatron/beir:${dataset}/test \
    --encoded_save_path $embedding_dir/query_${dataset}.pkl \
    --q_max_len 512 \
    --encode_is_qry
  fi
fi

if [ -f $embedding_dir/rank.${dataset}.txt ]; then
  echo "$embedding_dir/rank.${dataset}.txt already exists, skipping retrieval."
else
  echo "Starting retrieval for $dataset..."
  set -f && OMP_NUM_THREADS=12 python -m tevatron.faiss_retriever \
      --query_reps $embedding_dir/query_${dataset}.pkl \
      --passage_reps $embedding_dir/corpus_${dataset}.*.pkl \
      --depth 1000 \
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

cat $embedding_dir/eval.${dataset}.txt