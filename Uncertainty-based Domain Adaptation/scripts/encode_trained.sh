#!/bin/bash
script_root=$(dirname "$0")
dataset=$1
model=$2
pooling=$3
checkpoint_dir=$4

if [[ $model == "colbert" ]]; then
    # Set the path to the ColBERT index script
    if [ ! -d "${checkpoint_dir}/embeddings/index" ]; then
        bash $script_root/target_representation/colbert_encode/index_trained.sh $dataset $pooling $checkpoint_dir
    else
        echo "intermediates/${dataset}/colbert embedding already exists"
    fi
elif [[ $model == "monot5" ]]; then
    # Set the path to the Monot5 index script
    if [ ! -f "${checkpoint_dir}/embeddings.pt" ]; then
        if [ ! -d "${checkpoint_dir}/embeddings" ]; then
            python $script_root/target_representation/monot5_encode/encode.py \
                --base_modename_or_path $checkpoint_dir \
                --cache_dir "/data/.cache" \
                --clean_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
                --document_embedding_file "${checkpoint_dir}/embeddings.pt"
        else
            echo "monot5 splitted embedding already exists"
        fi
    else
        echo "monot5 embedding already exists"
    fi
elif [[ $model == "dpr" ]]; then
    if [ ! -f "${checkpoint_dir}/embeddings.pt" ]; then
        if [ ! -d "${checkpoint_dir}/embeddings" ]; then
            python $script_root/target_representation/dpr_encode/encode.py \
                --base_modename_or_path "${checkpoint_dir}/ctx_encoder" \
                --cache_dir "/data/.cache" \
                --clean_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
                --document_embedding_file "${checkpoint_dir}/embeddings.pt"
        else
            echo "intermediates/${dataset}/dpr splitted embedding already exists"
        fi
    else
        echo "intermediates/${dataset}/dpr embedding already exists"
    fi
elif [[ $model == "cocondenser" ]]; then
    if [ ! -d "${checkpoint_dir}/embeddings_beir" ]; then
        mkdir -p ${checkpoint_dir}/embeddings_beir
        for s in $(seq -f "%02g" 0 7)
        do
            if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "${checkpoint_dir}" \
                    --tokenizer_name "Luyu/co-condenser-marco-retriever" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --corpus_path "datasets/${dataset}/corpus.jsonl" \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --encoded_save_path "${checkpoint_dir}/embeddings_beir/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            else
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "${checkpoint_dir}" \
                    --tokenizer_name "Luyu/co-condenser-marco-retriever" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --encoded_save_path "${checkpoint_dir}/embeddings_beir/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            fi
        done
    else
        echo "intermediates/${dataset}/cocondenser embedding already exists"
    fi
elif [[ $model == "cocodr" ]]; then
    if [ ! -d "${checkpoint_dir}/embeddings_beir" ]; then
        mkdir -p ${checkpoint_dir}/embeddings_beir
        for s in $(seq -f "%02g" 0 7)
        do
            if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "${checkpoint_dir}" \
                    --tokenizer_name "OpenMatch/cocodr-base-msmarco" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --corpus_path "datasets/${dataset}/corpus.jsonl" \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --encoded_save_path "${checkpoint_dir}/embeddings_beir/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            else
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "${checkpoint_dir}" \
                    --tokenizer_name "OpenMatch/cocodr-base-msmarco" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --encoded_save_path "${checkpoint_dir}/embeddings_beir/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            fi
        done
    fi

elif [[ $model == "qwen3" ]]; then
    embedding_dir="${checkpoint_dir}/embeddings_beir"
    if [ ! -d "$embedding_dir" ]; then
        mkdir -p "$embedding_dir"
        pids=()
        num_shards=${QWEN3_ENCODE_SHARDS:-8}
        num_gpus=${QWEN3_NUM_GPUS:-8}
        for s in $(seq -f "%02g" 0 $((num_shards - 1)))
        do
            gpunum=$((10#$s % num_gpus))
            if [ -f "$embedding_dir/corpus_${dataset}.${s}.pkl" ]; then
                echo "$embedding_dir/corpus_${dataset}.${s}.pkl already exists, skipping encoding."
                continue
            fi
            if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
                CUDA_VISIBLE_DEVICES=${gpunum} python -m tevatron.retriever.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "$checkpoint_dir" \
                    --tokenizer_name "Qwen/Qwen3-Embedding-4B" \
                    --bf16 \
                    --per_device_eval_batch_size 64 \
                    --passage_max_len 512 \
                    --pooling last \
                    --padding_side left \
                    --query_prefix "" \
                    --dataset_path "datasets/${dataset}/corpus.jsonl" \
                    --encode_output_path "$embedding_dir/corpus_${dataset}.${s}.pkl" \
                    --dataset_number_of_shards $num_shards \
                    --dataset_shard_index ${s} &
            else
                CUDA_VISIBLE_DEVICES=${gpunum} python -m tevatron.retriever.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "$checkpoint_dir" \
                    --tokenizer_name "Qwen/Qwen3-Embedding-4B" \
                    --bf16 \
                    --per_device_eval_batch_size 64 \
                    --passage_max_len 512 \
                    --pooling last \
                    --padding_side left \
                    --query_prefix "" \
                    --dataset_name Tevatron/beir-corpus \
                    --dataset_config ${dataset} \
                    --dataset_split train \
                    --encode_output_path "$embedding_dir/corpus_${dataset}.${s}.pkl" \
                    --dataset_number_of_shards $num_shards \
                    --dataset_shard_index ${s} &
            fi
            pids+=($!)
        done
        exit_status=0
        for pid in "${pids[@]}"; do
            wait "$pid" || exit_status=1
        done
        if [ $exit_status -ne 0 ]; then
            echo "One or more encoding processes failed. Exiting."
            exit 1
        fi
    else
        echo "$embedding_dir already exists"
    fi
else
    echo "Unknown model: $model"
fi