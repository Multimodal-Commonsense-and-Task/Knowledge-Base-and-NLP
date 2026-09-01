# Add to PYTHONPATH
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/miniwob-plusplus/python'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/miniwob-plusplus'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/Workspace/web/compwob/'
#export PYTHONPATH=$PYTHONPATH:'/Users/minsookim/miniconda3/envs/synapse_comp/lib/python3.10/site-packages/'


base_dir=/Users/minsookim/Workspace/web/m2w2
data_dir=$base_dir/data

log_dir=$base_dir/results/miniwob/debugging_1


n_samples=20

subdomain=book-flight
subdomain=compositional.click-button_click-link
#subdomain=click-button_click-link
#subdomain=compositional.click-link_click-button_click-checkboxes_click-option_click-dialog-reverse

python run_miniwob.py --env_name $subdomain --seed 0 --num_episodes 50
#python run_miniwob.py --env_name $subdomain --no_memory --no_filter --seed 0 --num_episodes 50

#python run_mind2web.py --data_dir $data_dir \
#                       --benchmark $split \
#                       --log_dir $log_dir \
#                       --n_samples $n_samples
