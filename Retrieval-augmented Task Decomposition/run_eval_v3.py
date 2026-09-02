
import ipdb
import os
import json
import argparse

import numpy as np

from pathlib import Path

def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def analyze_prediction(data):
    for item in data:
        if 'pred_act' in item and 'target_act' in item:
            #ipdb.set_trace()
            print(item)
    
    ipdb.set_trace()

def dir_files(dir1):
    files1 = os.listdir(dir1)
    

    # Get the list of files that are in both directories
    files = set(files1)        
    # sort 0.json, 1.json, 2.json, ...
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    print(f"## Directory: {dir1}")
    print(f"## Number of files in dir: {len(files1)}\n")

    #ipdb.set_trace()
    return files

def task_str_map(dir1):
    """
    get mapping of id to task string
    """
    # Get the list of files in the directory
    dir1 = Path(dir1)    
    mapping = {}
    
    n_error = 0

    files = dir_files(dir1)

    # Compare the json files
    for file in files:
        path1 = dir1 / file                

        data1 = load_json(path1)
        #ipdb.set_trace()
        
        try:
            task_str = data1[0]['input'][1]['content'].split('\n')[0]    
        except Exception as e:
            n_error += 1
            #ipdb.set_trace()
            #break
            continue

        if task_str not in mapping:
            mapping[task_str] = file

    return mapping
    

def main(args):

    #excludes = [13, 18]
    excludes = []

    results = {'element_acc': [], 'action_f1': [], 'step_success': [], 'success': []}

    

    rada_logs = Path("/home/mnskim/workspace/web/Synapse/results/mind2web/exp_ours_comp_v6_oracle_7_100samples_topk30_prevtopk30_prevk5_completiontrajcheck_complexity/gpt-35-turbo-16k-mnskim/test_domain/")
    rada_files = dir_files(rada_logs)

    #ipdb.set_trace()


    metrics = {}
    n_found = 0
    n_error = 0
    

    map1 = task_str_map(args.dir1)

    map2 = task_str_map(rada_logs)

    ipdb.set_trace()
        
   

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str)
    #parser.add_argument("--dir2", type=str, default="results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/")
    args = parser.parse_args()

    main(args)
