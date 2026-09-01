import ipdb
import os
from pathlib import Path
import argparse
import json 

from synapse.utils.compwob import get_subtasks, get_subtasks_from_env_name

def run(args):
    
    res1 = get_res(args.path1, args)
    res2 = get_res(args.path2, args)
    #ipdb.set_trace()

    # get shared keys between res1 and res2
    shared_keys = set(res1.keys()).intersection(set(res2.keys()))
    # get the scores for each key
    # sort by number of subtasks and print the scores
    print("\n\nScores for each task by number of subtasks:")
    subtask_dict = {}
    for task_name in shared_keys:
        n_subtasks = len(res1[task_name]['subtasks'])
        if n_subtasks not in subtask_dict:
            subtask_dict[n_subtasks] = []
        subtask_dict[n_subtasks].append(task_name)

    # print paths
    print(f"Path1: {args.path1}")
    print(f"Path2: {args.path2}")

    # sort by number of subtasks
    subtask_dict = dict(sorted(subtask_dict.items(), key=lambda item: item[0]))
    for n_subtasks, items in subtask_dict.items():
        # print average score for each subtask, for res1 and res2
        print(f"## N={n_subtasks} subtasks: res1 average score: {sum([res1[item]['score'] for item in items]) / len(items)}")
        print(f"## N={n_subtasks} subtasks: res2 average score: {sum([res2[item]['score'] for item in items]) / len(items)}"
)
        # sort the items by task name
        items = sorted(items, key=lambda item: res1[item]['subtasks'])
        
        # now print res1 and res2 scores
        for item in items:
            print(f"Res1: {res1[item]['score']}: {res1[item]['subtasks']}")
            print(f"Res2: {res2[item]['score']}: {res2[item]['subtasks']}")
            print()

    # print the overall average score for res1 and res2, on the same set of tasks
    print(f"Overall average res1: {sum([res1[item]['score'] for item in shared_keys]) / len(shared_keys)} over {len(shared_keys)} tasks")
    print(f"Overall average res2: {sum([res2[item]['score'] for item in shared_keys]) / len(shared_keys)} over {len(shared_keys)} tasks")

    # export results to csv using following format
    # task_name, subtasks, res1_score, res2_score
    with open("comparison_results.csv", 'w') as f:
        f.write("task_name, subtasks, n_subtasks, res1_score, res2_score\n")
        # sort by number of subtasks
        subtask_dict = dict(sorted(subtask_dict.items(), key=lambda item: item[0]))
        # write the results
        for n_subtasks, items in subtask_dict.items():
            # sort the items by task name
            items = sorted(items, key=lambda item: res1[item]['subtasks'])
            # write the task name, subtasks, n_subtasks, res1_score, res2_score
            for item in items:
                f.write(f"{item}, {' | '.join(res1[item]['subtasks'])}, {len(res1[item]['subtasks'])}, {res1[item]['score']}, {res2[item]['score']}\n")
                #f.write(f"{task_name}, {'|'.join(res1[task_name]['subtasks'])}, {len(res1[task_name]['subtasks'])}, {res1[task_name]['score']}, {res2[task_name]['score']}\n")

            #f.write(f"{task_name}, {'|'.join(res1[task_name]['subtasks'])}, {len(res1[task_name]['subtasks'])}, {res1[task_name]['score']}, {res2[task_name]['score']}\n")
    print(f"Exported results to comparison_results.csv")

def get_res(path, args):
    context_limit_failure_count = 0
    total_items = 0

    res_dict = {}
    # get all 1-level subdirs
    dir_paths = list(Path(path).glob('./*'))
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
    print(f"Total items: {total_items}")

    return res_dict

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str)

    args = parser.parse_args()
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_miniwob_baseline_1/gpt-3.5-turbo-0301"   
    args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_miniwob_baseline_2/gpt-3.5-turbo-0301"   

    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_miniwob_baseline_2_compwob_compositional/gpt-3.5-turbo-0301"   

    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_baseline_compwob_compositional_first/gpt-3.5-turbo-0301"
    
    #args.path = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_baseline_2_compwob_compositional_first/gpt-35-turbo-16k-mnskim"
    
    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/synapse_baseline_5_compwob_compositional_first/gpt-3.5-turbo-0301"
    #args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_update_5_compwob_compositional_first/gpt-3.5-turbo-16k-0613"
    
    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_naive_update_6_compwob_compositional_first/gpt-3.5-turbo-16k-0613"


    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_subtask_2_planner_reduceiffail_compwob_compositional_first/gpt-35-turbo-mnskim"

    model="gpt-3.5-turbo-0301"
    exp_name="debug_synapse_comp_plan_naive_update_5"
    exp_name="debug_synapse_update_5"
    
    model="gpt-3.5-turbo-16k-0613"
    
    args.path1 = f"/Users/minsookim/Workspace/web/m2w2/results/miniwob/{exp_name}_compwob_compositional_first/{model}"


    # subtask
    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_subtask_2_compwob_compositional_first/gpt-3.5-turbo-16k-0613"
    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_naive_update_6b_compwob_compositional_first/gpt-3.5-turbo-0613"
    
    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_synapse_comp_plan_subtask_2_planner_compwob_compositional_first/gpt-35-turbo-mnskim"

    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_1_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    # improve above: subtask planner making errors
    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"


    # reverse res
    #args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    #args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    # with reorder
    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    # new
    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    #args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reorder_reverse_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_reverse_compwob_compositional_first_planning/gpt-35-turbo-mnskim"


    #args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    #args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    #args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    #args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_reverse_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"



    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_reverse_7_compwob_compositional_first_planning/gpt-35-turbo-mnskim"


    # 
    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_7_compwob_compositional_first_planning/gpt-35-turbo-mnskim"


    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2b_planner_2_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v2b_planner_2_reorder_8_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/ours_v2_planner_2_reverse_reorder_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v2b_planner_2_reorder_reverse_8_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v2b_planner_2_reorder_reverse_8_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v6_2_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    args.path1 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v2b_planner_2_reorder_8_compwob_compositional_first_planning/gpt-35-turbo-mnskim"
    args.path2 = "/Users/minsookim/Workspace/web/m2w2/results/miniwob/debug_ours_v6_2_orig_compwob_compositional_first_planning/gpt-35-turbo-mnskim"

    run(args)
