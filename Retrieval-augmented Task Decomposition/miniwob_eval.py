import ipdb
import os
from pathlib import Path
import argparse

def run(args):

    res_dict = {}
    # get all 1-level subdirs
    dir_paths = list(Path(args.path).glob('./*'))
    #ipdb.set_trace()
    # for each subdir, get the task name
    for dir_path in dir_paths:
        task_name = dir_path.name
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

        score = success_count / (success_count + failure_count)
        res_dict[task_name] = score
        #ipdb.set_trace()

    # print the score for each task
    for task_name, score in res_dict.items():
        print(f"{task_name}: {score}")

    # print the number of tasks with valid scores, print the skipped tasks
    print(f"Valid scores: {len(res_dict)}")
    print(f"Skipped tasks: {len(dir_paths) - len(res_dict)}")
    # print the macro average score
    print(f"Macro average: {sum(res_dict.values()) / len(dir_paths)} over {len(dir_paths)} tasks")
    #ipdb.set_trace()

    # print all scores < 1.0   
    for task_name, score in res_dict.items():
        if score < 1.0:
            print(f"{task_name}: {score}")

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
    exp_name="synapse_comp_plan_naive_1"
    #args.path = f"/Users/minsookim/Workspace/web/m2w2/results/miniwob/{exp_name}_compwob_compositional_first/{model}"

    run(args)
