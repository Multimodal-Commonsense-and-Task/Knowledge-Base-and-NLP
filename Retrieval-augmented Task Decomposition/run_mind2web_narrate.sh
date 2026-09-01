#log_dir=$base_dir/results/mind2web/exp_base2
log_dir=$base_dir/results/mind2web/debugging
log_dir=$base_dir/results/mind2web/debugging_nar2_lookahead4

log_dir=$base_dir/results/mind2web/exp_lookahead2
log_dir=$base_dir/results/mind2web/exp_baseline2
log_dir=$base_dir/results/mind2web/exp_narrate2
log_dir=$base_dir/results/mind2web/exp_plannarrate2

log_dir=$base_dir/results/mind2web/exp_react2

log_dir=$base_dir/results/mind2web/exp_react_3

# Exp name
exp_name=exp_react_comp_v2_5_oracle

memory_annotation_name=mem_annot_1_bugfix

# Paths
base_dir=/home/mnskim/workspace/web/Synapse
data_dir=$base_dir/data
log_dir=$base_dir/results/mind2web/$exp_name
annotated_memory_path=$base_dir/memory/annotated/$memory_annotation_name

# data split
#split="test_task"
split="test_website"
#split="test_domain"

# hyperparams
n_samples=50

python run_mind2web_orig.py --data_dir $data_dir \
                            --benchmark $split \
                            --log_dir $log_dir \
                            --n_samples $n_samples \
                            --comp \
                            --use_memory_annotation \
                            --annotated_memory_path $annotated_memory_path \
                            --mind2web_oracle
                            #--start_idx 35
