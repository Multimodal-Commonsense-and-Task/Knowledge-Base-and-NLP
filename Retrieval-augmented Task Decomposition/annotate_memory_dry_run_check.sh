
# Exp name
exp_name=mem_annot_1_bugfix

# Paths
base_dir=/home/mnskim/workspace/web/Synapse
#data_dir=$base_dir/data
#log_dir=$base_dir/results/mind2web/$exp_name
memory_input_path=$base_dir/memory

memory_output_path=$base_dir/memory/annotated/$exp_name

# data split
#split="test_task"
#split="test_website"
#split="test_domain"


python synapse/memory/mind2web/annotate_memory.py --memory_input_path $memory_input_path \
                                                  --memory_output_path $memory_output_path \
                                                  --do_check \
                                                  --dry_run
                                    
                            
