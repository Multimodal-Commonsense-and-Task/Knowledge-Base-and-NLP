# Add to PYTHONPATH
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/miniwob-plusplus/python'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/miniwob-plusplus'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/miniconda3/envs/synapse_comp/lib/python3.10/site-packages/'

# Dataset
dataset=compwob_compositional
#dataset=compwob_miniwob56

# exp id
exp_name=debug_synapse_update_5

# hparams
n_episodes_per_task=10
prompting_strat=first
exp_name=${exp_name}_${dataset}_${prompting_strat}
heuristic_termination=3

## api
# openai
api=openai
model=gpt-3.5-turbo-0301
model=gpt-3.5-turbo-16k-0613

# azure
#api=azure1
#model=gpt-35-turbo-16k-mnskim

# Paths
base_dir=/Users/minsookim/Workspace/web/m2w2
#data_dir=$base_dir/data
log_dir=$base_dir/results/miniwob/$exp_name

python run_miniwob_experiments.py --seed 0 \
                                  --api $api \
                                  --model $model \
                                  --num_episodes $n_episodes_per_task \
                                  --dataset $dataset \
                                  --log_dir $log_dir \
                                  --heuristic_termination $heuristic_termination \
                                  --compwob_prompting_strategy $prompting_strat \
                                  --headless
