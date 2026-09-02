# hyperparams
n_samples=912
top_k_elements=30
previous_top_k_elements=30
see_previous_k=5

# Exp name
#exp_name=exp_ours_comp_v6_oracle_3
#exp_name=exp_ours_comp_v6_oracle_4

# v5+
#exp_name=exp_ours_comp_v6_oracle_5_topk$top_k_elements
exp_name=exp_ours_comp_v6_oracle_7_${n_samples}samples_topk${top_k_elements}_prevtopk${previous_top_k_elements}_prevk${see_previous_k}_completiontrajcheck_complexity

exp_name=debugrm3
exp_name=gpt4exp

exp_name=decomp

## api
# azure
api=azure1
model=gpt-35-turbo-mnskim
model=gpt-35-turbo-16k-mnskim

#api=openai
#model=gpt-4-0613

# Paths
base_dir=/home/mnskim/workspace/web/Synapse
data_dir=$base_dir/data
log_dir=$base_dir/results/mind2web/$exp_name

# data split
#split="test_task"
split="test_website"
split="test_domain"

python run_mind2web_orig.py --data_dir $data_dir \
                            --benchmark $split \
                            --log_dir $log_dir \
                            --n_samples $n_samples \
                            --api $api \
                            --model $model \
                            --plan \
                            --mind2web_oracle \
                            --top_k_elements $top_k_elements \
                            --previous_top_k_elements $previous_top_k_elements \
                            --see_previous_k $see_previous_k \
                            --order_by_complexity    
                            #--start_idx 35
