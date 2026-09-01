# Add to PYTHONPATH
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/miniwob-plusplus/python'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/miniwob-plusplus'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/miniconda3/envs/synapse_comp/lib/python3.10/site-packages/'

# Dataset
dataset=compwob_compositional
#dataset=compwob_miniwob56

# exp id
exp_name=ours_v2_planner_2
exp_name=ours_v2_planner_3_reverse_reorder
#exp_name=ours_v2_planner_2_reorder

# debugging name
#exp_name=ours_v2_planner_2_debug
#exp_name=debug_ours_v2_planner_2_reverse


# hparams
n_episodes_per_task=10
prompting_strat=first
heuristic_termination=3
planner_type=planning

## api
# openai
api=openai
model=gpt-3.5-turbo-0301
model=gpt-3.5-turbo-0613
#model=gpt-3.5-turbo-16k-0613

# azure
api=azure1
#model=gpt-35-turbo-16k-mnskim
model=gpt-35-turbo-mnskim

# LOG
exp_name=${exp_name}_${dataset}_${prompting_strat}_${planner_type}

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
                                  --comp_planner \
                                  --subtask \
                                  --reduce_if_fail \
                                  --planner_type $planner_type \
                                  --headless \
                                  --reverse
                                  #--headless
