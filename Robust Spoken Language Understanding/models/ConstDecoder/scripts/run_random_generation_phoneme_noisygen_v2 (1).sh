
PRETRAINED_MODEL=bert-base-uncased

GPU_ID=3
base_path=../../../datasets/dstc_eval
task=rg
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ../phoneme_random_generate.py \
    --base_model ${PRETRAINED_MODEL} \
    --tokenizer_name ${PRETRAINED_MODEL} \
    --model_path ./models.noisygen.phoneme_v2/best.pt  \
    --test_data_path  ${base_path}/${task}.json \
    --output_path ${base_path}/random_phoneme_v2_augmented_logitsum_${task}.json \
    --device 0 \
    --tag_pdrop 0.2 \
    --lm_weight 0.5 \
    --decoder_proj_pdrop 0.2 \
    --tag_hidden_size 768 \
    --tag_size 3 \
    --alpha 3.0 \
    --change_weight 1.5 \
    --vocab_size 30522 \
    --pad_token_id 0 \
    --error_rate 0.21 \
    --max_add_len 10 >random_augmented_phoneme_v2_noisygen_lowlm_${task}.log 2>&1 &
