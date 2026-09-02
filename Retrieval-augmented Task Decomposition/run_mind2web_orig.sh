# hyperparams
n_samples=100
top_k_elements=5

# Exp name
exp_name=exp_base_3_synapsenew_${n_samples}samples_topk${top_k_elements}_complexity

base_dir=/home/mnskim/workspace/web/Synapse
data_dir=$base_dir/data

log_dir=$base_dir/results/mind2web/$exp_name

n_samples=100

#split="test_task"
#split="test_website"
split="test_domain"

# api
api='azure1'
model='gpt-35-turbo-16k-mnskim'

python run_mind2web_orig.py --data_dir $data_dir \
                       --benchmark $split \
                       --log_dir $log_dir \
                       --n_samples $n_samples \
                       --api $api \
                       --model $model \
                       --top_k_elements $top_k_elements \
                       --order_by_complexity    
