# Exp name
exp_name=exp_parity_baseline_2
#memory_annotation_name=mem_annot_debug

exp_name=exp_parity_baseline_3
#
# Paths
base_dir=/home/mnskim/workspace/web/Synapse
data_dir=$base_dir/data
log_dir=$base_dir/results/mind2web/$exp_name
annotated_memory_path=$base_dir/memory/annotated/$memory_annotation_name

# data split
split="test_task"
split="test_website"
split="test_domain"

# hyperparams
n_samples=50

python run_mind2web_orig.py --data_dir $data_dir \
                            --benchmark $split \
                            --log_dir $log_dir \
                            --n_samples $n_samples \
                            #--narrate \
