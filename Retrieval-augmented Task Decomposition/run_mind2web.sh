base_dir=/home/mnskim/workspace/web/Synapse
data_dir=$base_dir/data

#log_dir=$base_dir/results/mind2web/exp_base2
log_dir=$base_dir/results/mind2web/debugging
log_dir=$base_dir/results/mind2web/debugging_nar


n_samples=20

split="test_task"
#split="test_website"
#split="test_domain"

python run_mind2web.py --data_dir $data_dir \
                       --benchmark $split \
                       --log_dir $log_dir \
                       --n_samples $n_samples
