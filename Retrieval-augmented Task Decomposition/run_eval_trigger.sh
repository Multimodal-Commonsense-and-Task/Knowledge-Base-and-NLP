path=./results/mind2web/debugging_nar2_lookahead3/gpt-3.5-turbo-0613/test_task/
#path=./results/mind2web/debugging_nar2_lookahead4/gpt-3.5-turbo-0613/test_task/

#path=results/mind2web/exp_base1/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/

path=results/mind2web/exp_lookahead2/gpt-3.5-turbo-0613/test_task/
path=results/mind2web/exp_baseline2/gpt-3.5-turbo-0613/test_task/
path=results/mind2web/exp_narrate2/gpt-3.5-turbo-0613/test_task/
path=results/mind2web/exp_plannarrate2/gpt-3.5-turbo-0613/test_task/

path=results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_baseline2/gpt-3.5-turbo-0613/test_task/

#path=results/mind2web/exp_react2/gpt-3.5-turbo-0613/test_task/

# set 3
path=results/mind2web/exp_base_3/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_base_3/gpt-3.5-turbo-0613/test_website/
#path=results/mind2web/exp_base_3/gpt-3.5-turbo-0613/test_domain/

#path=results/mind2web/exp_react_3/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_react_3/gpt-3.5-turbo-0613/test_website/

# compositional annotation
path=results/mind2web/exp_react_comp_1/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_react_comp_1/gpt-3.5-turbo-0613/test_website/

path=results/mind2web/exp_react_comp_2/gpt-3.5-turbo-0613/test_task/
path=results/mind2web/exp_react_comp_2/gpt-3.5-turbo-0613/test_website/
path=results/mind2web/exp_react_comp_v2_1/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_react_comp_v2_1/gpt-3.5-turbo-0613/test_website/

path=results/mind2web/exp_react_comp_v2_3_oracle/gpt-3.5-turbo-0613/test_task/
path=results/mind2web/exp_react_comp_v2_3_oracle/gpt-3.5-turbo-0613/test_website/


# rerun baseline for parity
#path=results/mind2web/exp_parity_baseline_1/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_parity_baseline_2/gpt-3.5-turbo-0613/test_task/
#path=results/mind2web/exp_parity_baseline_2/gpt-3.5-turbo-0613/test_website/

# baseline
# step sr 0.21
path=results/mind2web/exp_parity_baseline_3/gpt-3.5-turbo-0613/test_domain/
# 0.2487/0.3632/0.2155
path=results/mind2web/exp_base_3_synapsenew/gpt-35-turbo-16k-mnskim/test_domain/
# 0.1946/0.5716/0.1688
path=results/mind2web/exp_base_3_synapsenew_topk20/gpt-35-turbo-16k-mnskim/test_domain/
# 0.1797/0.5452/0.1610
path=results/mind2web/exp_base_3_synapsenew_topk20_prevk3/gpt-35-turbo-16k-mnskim/test_domain/
# 0.2020/0.4473/0.1782
path=results/mind2web/exp_base_3_synapsenew_topk10_prevk3/gpt-35-turbo-16k-mnskim/test_domain/

# step sr 0.21
#path=results/mind2web/exp_ours_comp_v6_oracle_2/gpt-35-turbo-mnskim/test_domain/
# step sr 0.21
#path=results/mind2web/exp_ours_comp_v6_oracle_3/gpt-35-turbo-mnskim/test_domain/
# 0.2801/0.5651/0.2445
#path=results/mind2web/exp_ours_comp_v6_oracle_5_topk20/gpt-35-turbo-16k-mnskim/test_domain/


# 0.2956/0.5679/0.2590
#path=results/mind2web/exp_ours_comp_v6_oracle_6_topk20/gpt-35-turbo-16k-mnskim/test_domain/
# 0.3168/0.6216/0.2785 (previous_k = 3)
#path=results/mind2web/exp_ours_comp_v6_oracle_6_topk30/gpt-35-turbo-16k-mnskim/test_domain/
# 0.3103/0.6358/0.2749 (previous_k = 3, add completed subtask info in the input)
#path=results/mind2web/exp_ours_comp_v6_oracle_6_topk30_prevk3_addcompletedsubtask/gpt-35-turbo-16k-mnskim/test_domain/
# 0.3129/0.6377/0.2779 (previous_k = 3, add remaining subtask info in the input)
path=results/mind2web/exp_ours_comp_v6_oracle_6_topk30_prevk3_addremainingsubtask/gpt-35-turbo-16k-mnskim/test_domain/
# 0.3054/0.6141/0.2702 (previous_k = 5)
#path=results/mind2web/exp_ours_comp_v6_oracle_6_topk30_prevk5/gpt-35-turbo-16k-mnskim/test_domain/
# 0.2838/0.6723/0.2555 (previous_k = 3)
#path=results/mind2web/exp_ours_comp_v6_oracle_6_topk50/gpt-35-turbo-16k-mnskim/test_domain/


###  no dataset complexity ranking
# 0.2790/0.6308/0.2661 (previous_k = 3, add remaining subtask info in the input)
path=results/mind2web/exp_ours_comp_v6_oracle_6_topk30_prevk3_addremainingsubtask_nocomplexity/gpt-35-turbo-16k-mnskim/test_domain/
# 0.2828/0.6290/0.2681 (prevk 3, prevtopk 30, topk 30) add remaining subtask info in the input
path=results/mind2web/exp_ours_comp_v6_oracle_6_topk30_prevtopk30_prevk3_addremainingsubtask_nocomplexity_minorfix/gpt-35-turbo-16k-mnskim/test_domain/
# 0.2467/0.6069/0.2262 (previous_k = 3, add remaining subtask info in the input)
#path=results/mind2web/exp_ours_comp_v6_oracle_6_topk30_prevk3_addremainingsubtask_nocomplexity_minorfix/gpt-35-turbo-16k-mnskim/test_domain/
# 0.2420/0.4944/0.2213 baseline, no previous_k
#path=results/mind2web/exp_base_3_synapsenew_topk10_nocomplexity/gpt-35-turbo-16k-mnskim/test_domain/


# v7 
# 0.2914/0.6272/0.2767
path=results/mind2web/exp_ours_comp_v6_oracle_7_topk30_prevtopk30_prevk3_completiontrajcheck/gpt-35-turbo-16k-mnskim/test_domain/
# 0.2797/0.6376/0.2656/0.0200 , 100 samples
path=results/mind2web/exp_ours_comp_v6_oracle_7_100samples_topk30_prevtopk30_prevk3_completiontrajcheck/gpt-35-turbo-16k-mnskim/test_domain/
# 0.3005/0.6421/0.2697, 100 samples, compl rank
path=results/mind2web/exp_ours_comp_v6_oracle_7_100samples_topk30_prevtopk30_prevk3_completiontrajcheck_complexity/gpt-35-turbo-16k-mnskim/test_domain/
# 0.3054/0.6307/0.2768, 100 samples, compl rank prevk5
dir1=results/mind2web/exp_ours_comp_v6_oracle_7_100samples_topk30_prevtopk30_prevk5_completiontrajcheck_complexity/gpt-35-turbo-16k-mnskim/test_domain/
# 0.3028/0.5921/0.2722/0.000, 100 samples, compl rank prevk5
path=results/mind2web/exp_ours_comp_v6_oracle_7_100samples_topk20_prevtopk20_prevk5_completiontrajcheck_complexity/gpt-35-turbo-16k-mnskim/test_domain/


# v7 baseline
# 0.2880/0.3976/0.2665/0.0000
#path=results/mind2web/exp_base_3_synapsenew_100samples_topk5_nocomplexity/gpt-35-turbo-16k-mnskim/test_domain/
# 0.2473/0.3778/0.2134, 100 samples, compl rank
dir2=results/mind2web/exp_base_3_synapsenew_100samples_topk5_complexity/gpt-35-turbo-16k-mnskim/test_domain/

path=results/mind2web/gpt4exp/gpt-4-0613/test_domain/


# synapse baseline
path=/data/mnskim/data/mind2web/synapse/baseline/results/mind2web/gpt-3.5-turbo-16k-0613/test_domain
path=/data/mnskim/data/mind2web/synapse/baseline/results/mind2web/gpt-3.5-turbo-16k-0613/test_website
path=/data/mnskim/data/mind2web/synapse/baseline/results/mind2web/gpt-3.5-turbo-16k-0613/test_task
#path=/data/mnskim/data/mind2web/synapse/baseline/results/mind2web/codellama/test_domain/3shot/top5
#path=/data/mnskim/data/mind2web/synapse/baseline/results/mind2web/codellama/test_task/3shot/top5
#path=/data/mnskim/data/mind2web/synapse/baseline/results/mind2web/codellama/test_website/3shot/top5

savedir=results/mind2web/trigger1/

python run_eval_with_trigger.py --dir1 $dir1 --dir2 $dir2 --trigger_map trigger_map.json --save_dir $savedir

