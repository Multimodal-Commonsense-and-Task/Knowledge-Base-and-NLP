#!/bin/bash
script_root=$(dirname "$0")

dataset=$1
model=$2
pooling=$3
partition_idx=${4:0}

if [[ $model == "colbert" ]]; then
    # Set the path to the ColBERT index script
    if [ ! -d "intermediates/${dataset}/colbert/${pooling}" ]; then
        python $script_root/target_representation/colbert_encode/data_prep.py \
            --dataset $dataset \
            --clean_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --collection "intermediates/${dataset}/colbert/collection.tsv"
        bash $script_root/target_representation/colbert_encode/index.sh $dataset $pooling
    else
        echo "intermediates/${dataset}/colbert embedding already exists"
    fi
elif [[ $model == "monot5" ]]; then
    # Set the path to the Monot5 index script
    if [ ! -f "intermediates/${dataset}/document_embeddings_monot5.pt" ]; then
        python $script_root/target_representation/monot5_encode/encode.py \
            --base_modename_or_path "castorini/monot5-base-msmarco-10k" \
            --cache_dir "/data/.cache" \
            --clean_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --document_embedding_file "intermediates/${dataset}/document_embeddings_monot5.pt"
    else
        echo "intermediates/${dataset}/monot5 embedding already exists"
    fi
elif [[ $model == "dpr" ]]; then
    if [ ! -f "intermediates/${dataset}/document_embeddings_dpr.pt" ]; then
        python $script_root/target_representation/dpr_encode/encode.py \
            --base_modename_or_path "sentence-transformers/facebook-dpr-ctx_encoder-multiset-base" \
            --cache_dir "/data/.cache" \
            --clean_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --document_embedding_file "intermediates/${dataset}/document_embeddings_dpr.pt"
    else
        echo "intermediates/${dataset}/dpr embedding already exists"
    fi
elif [[ $model == "cocondenser" ]]; then
    if [ ! -d "intermediates/${dataset}/document_embeddings_cocondenser" ]; then
        mkdir -p intermediates/${dataset}/document_embeddings_cocondenser
        for s in $(seq -f "%02g" 0 7)
        do
            if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "Luyu/co-condenser-marco-retriever" \
                    --tokenizer_name "Luyu/co-condenser-marco-retriever" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --corpus_path "datasets/${dataset}/corpus.jsonl" \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --encoded_save_path "intermediates/${dataset}/document_embeddings_cocondenser/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            else
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "Luyu/co-condenser-marco-retriever" \
                    --tokenizer_name "Luyu/co-condenser-marco-retriever" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --encoded_save_path "intermediates/${dataset}/document_embeddings_cocondenser/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            fi
        done
    else
        echo "intermediates/${dataset}/cocondenser embedding already exists"
    fi
elif [[ $model == "cocodr" ]]; then
    if [ ! -d "intermediates/${dataset}/document_embeddings_cocodr" ]; then
        mkdir -p intermediates/${dataset}/document_embeddings_cocodr
        for s in $(seq -f "%02g" 0 7)
        do
            if [ "$dataset" == "trec-news" ] || [ "$dataset" == "robust04" ]; then
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "OpenMatch/cocodr-base-msmarco" \
                    --tokenizer_name "OpenMatch/cocodr-base-msmarco" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --corpus_path "datasets/${dataset}/corpus.jsonl" \
                    --encoded_save_path "intermediates/${dataset}/document_embeddings_cocodr/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            else
                python -m tevatron.driver.encode \
                    --output_dir=temp \
                    --model_name_or_path "OpenMatch/cocodr-base-msmarco" \
                    --tokenizer_name "OpenMatch/cocodr-base-msmarco" \
                    --fp16 \
                    --per_device_eval_batch_size 64 \
                    --p_max_len 512 \
                    --dataset_name Tevatron/beir-corpus:${dataset} \
                    --encoded_save_path "intermediates/${dataset}/document_embeddings_cocodr/corpus${s}.pkl" \
                    --encode_num_shard 8 \
                    --encode_shard_index ${s}
            fi
        done
    else
        echo "intermediates/${dataset}/cocodr embedding already exists"
    fi

elif [[ $model == "qwen3" ]]; then
    embedding_dir="intermediates/${dataset}/document_embeddings_qwen3"
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
                    --model_name_or_path "Qwen/Qwen3-Embedding-4B" \
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
                    --model_name_or_path "Qwen/Qwen3-Embedding-4B" \
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