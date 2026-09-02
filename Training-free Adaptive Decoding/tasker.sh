TASK_NAME=$1
SCREEN_NAME=$2
WINDOW_NUMBER=$3

# This file will run the task over the 5 evaluations in a given screen window
# It will run the commands using GPU of the same screen window
# The output of each command will be saved in a file with the name of the task and the evaluation number
# The commands run sequentially

# Create output directory
mkdir -p ./outs/$TASK_NAME

# Execute
# screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=float16,attn_implementation=flash_attention_2 --tasks $TASK_NAME --batch_size 1 &> ./outs/$TASK_NAME/1.txt\n"
# screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,harp_dropout_rate=0.2,harp_theta=1.2,dtype=float16,attn_implementation=flash_attention_2 --tasks $TASK_NAME --batch_size 1 &> ./outs/$TASK_NAME/2.txt\n"
# screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,harp_dropout_rate=0.2,harp_theta=1.2,dtype=float16,attn_implementation=flash_attention_2 --tasks $TASK_NAME --batch_size 1 --seed 42 &> ./outs/$TASK_NAME/2-42.txt\n"
# screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,harp_dropout_rate=0.2,harp_theta=1.2,dtype=float16,attn_implementation=flash_attention_2 --tasks $TASK_NAME --batch_size 1 --seed 9999 &> ./outs/$TASK_NAME/2-9999.txt\n"
# screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,harp_dropout_rate=0.2,harp_theta=1.2,dtype=float16,attn_implementation=flash_attention_2 --tasks $TASK_NAME --batch_size 1 --seed 1234 &> ./outs/$TASK_NAME/2-1234.txt\n"
# screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,harp_last_token_only=True,dtype=float16,attn_implementation=flash_attention_2,teal_path=./TEAL/Meta-Llama-3.1-8B-Instruct-TEAL,harp_sparsity=0.3,harp_sparsify_mlp=True,harp_sparsify_attn=True,harp_entropy_threshold=1 --tasks $TASK_NAME --batch_size 1 &> ./outs/$TASK_NAME/3.txt\n"
screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=float16,attn_implementation=flash_attention_2 --tasks $TASK_NAME --gen_kwargs length_penalty=0.6,num_beams=3 --batch_size 1 &> ./outs/$TASK_NAME/4.txt\n"
screen -S $SCREEN_NAME -p $WINDOW_NUMBER -X stuff "CUDA_VISIBLE_DEVICES=$WINDOW_NUMBER lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=float16,attn_implementation=flash_attention_2 --tasks $TASK_NAME --gen_kwargs length_penalty=0.6,num_beams=5 --batch_size 1 &> ./outs/$TASK_NAME/5.txt\n"