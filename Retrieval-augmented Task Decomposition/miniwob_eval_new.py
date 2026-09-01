import ipdb
import os
from pathlib import Path
import argparse
import json 

from synapse.utils.compwob import get_subtasks, get_subtasks_from_env_name

def run(args):
    context_limit_failure_count = 0
    context_reduce_count = 0
    total_items = 0

    res_dict = {}
    # get all 1-level subdirs
    dir_paths = list(Path(args.path).glob('./*'))
    #ipdb.set_trace()
    # for each subdir, get the task name
    for dir_path in dir_paths:
        task_name = dir_path.name
        subtasks = get_subtasks_from_env_name(task_name)
        #ipdb.set_trace()
        
        #ipdb.set_trace()

        # get all files in the dir
        files = list(dir_path.rglob('*'))
        if len(files) == 0:
            #ipdb.set_trace()
            print(f"### Skipping {task_name}...")
            continue
        # for each file, if it ends with success.json, its a success file
        # if it ends with failure.json, its a failure file
        # tally the score in the dir
        success_count = 0
        failure_count = 0
        for file in files:
            if file.stem.endswith('success'):
                success_count += 1
            elif file.stem.endswith('fail'):
                failure_count += 1
        
        if success_count + failure_count == 0:
            ipdb.set_trace()

        # load the file and look for "FAILED DUE TO THE CONTEXT LIMIT:" in the output
        for file in files:
            total_items += 1
            if file.stem.endswith('fail'):
                dat = json.load(open(file, 'r'))
                for item in dat:
                    if 'reduce' in item['output']:
                        context_reduce_count += 1
                        
                    if "FAILED DUE TO THE CONTEXT LIMIT" in item['output']:
                        context_limit_failure_count += 1
                        break
               
        score = success_count / (success_count + failure_count)
        res_dict[task_name] = {'score': score, 'subtasks': subtasks}
        #ipdb.set_trace()

    

    # print the macro average score
    #print(f"Macro average: {sum(res_dict.values()) / len(dir_paths)} over {len(dir_paths)} tasks")
    #ipdb.set_trace()

    # print all scores < 1.0   
    for task_name, item in res_dict.items():
        if item['score'] < 1.0:
            print(f"{task_name}: {item['score']}")
            print(f"Subtasks: {item['subtasks']}")

    # print the score for each task
    print("\n\nScores for each task:")
    for task_name, item in res_dict.items():
        print(f"{task_name}: {item['score']}")
        print(f"Subtasks: {item['subtasks']}")
        print()

    # categorize tasks by number of subtasks and print the scores
    print("\n\nScores for each task by number of subtasks:")
    subtask_dict = {}
    for task_name, item in res_dict.items():
        n_subtasks = len(item['subtasks'])
        if n_subtasks not in subtask_dict:
            subtask_dict[n_subtasks] = []
        subtask_dict[n_subtasks].append(item)
    
    # sort by number of subtasks
    subtask_dict = dict(sorted(subtask_dict.items(), key=lambda item: item[0]))
    for n_subtasks, items in subtask_dict.items():
        print(f"## N={n_subtasks} subtasks: Average score: {sum([item['score'] for item in items]) / len(items)}")
        # sort the items by task name
        items = sorted(items, key=lambda item: item['subtasks'])
        for item in items:
            print(f"{item['score']}: {item['subtasks']}")
        print()

    # print the number of tasks with valid scores, print the skipped tasks
    print(f"Valid scores: {len(res_dict)}")
    print(f"Skipped tasks: {len(dir_paths) - len(res_dict)}")
    # print the overall average score
    print(f"Overall average: {sum([item['score'] for item in res_dict.values()]) / len(dir_paths)} over {len(dir_paths)} tasks")

    print(f"Context limit failure count: {context_limit_failure_count}")
    print(f"Context reduce count: {context_reduce_count}")
    print(f"Total items: {total_items}")

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str)

    args = parser.parse_args()
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_miniwob_baseline_1/gpt-3.5-turbo-0301"   
    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_miniwob_baseline_2/gpt-3.5-turbo-0301"   

    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_miniwob_baseline_2_compwob_compositional/gpt-3.5-turbo-0301"   

    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_baseline_compwob_compositional_first/gpt-3.5-turbo-0301"
    
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_baseline_2_compwob_compositional_first/gpt-35-turbo-16k-mnskim"
    
    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_baseline_5_compwob_compositional_first/gpt-3.5-turbo-0301"

    model="gpt-3.5-turbo-0301"
    exp_name="debug_synapse_comp_plan_naive_update_6"
    
    #exp_name="debug_synapse_update_5"

    model="gpt-3.5-turbo-16k-0613"
    
    args.path = f"/Users/minsookim/Workspace/web/m2w2/results/miniwob/{exp_name}_compwob_compositional_first/{model}"


    #subtasks
    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_subtask_2_compwob_compositional_first/gpt-3.5-turbo-16k-0613"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_subtask_2_compwob_compositional_first/gpt-3.5-turbo-16k-0613"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_naive_update_6b_compwob_compositional_first/gpt-3.5-turbo-0613"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_naive_update_6b_compwob_compositional_first/gpt-3.5-turbo-0613"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_subtask_2_planner_compwob_compositional_first/gpt-35-turbo-mnskim"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_subtask_2_planner_reduceiffail_compwob_compositional_first/gpt-35-turbo-mnskim"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_1_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    # improve and bugfix
    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    # reverse
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    # reverse and reorder
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"


    # new 
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    # reverse and reorder
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reorder_reverse_compwob_compositional_first_planning/gpt-35-turbo-mnskim"


    args.path ="/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_reverse_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reorder_reverse_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v2b_planner_2_reorder_8_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    
    # 40% success
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    # 51.76	/ 36
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v2b_planner_2_reorder_8_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    # 
    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v6_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v6_2_orig_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v6_2_orig_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    run(args)
