
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

def main(args):

    #excludes = [13, 18]
    excludes = []

    results = {'element_acc': [], 'action_f1': [], 'step_success': [], 'success': []}

    # Get the list of files in the directory
    dir1 = Path(args.dir1)    
    files1 = os.listdir(dir1)
    

    # Get the list of files that are in both directories
    files = set(files1)        
    # sort 0.json, 1.json, 2.json, ...
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    print(f"## Directory: {dir1}")
    print(f"## Number of files in dir: {len(files1)}\n")

    #ipdb.set_trace()
    if len(excludes) > 0:
        print(f"## Excluding files: {excludes}")
        files = [file for file in files if int(file.split(".")[0]) not in excludes]
        print(f"## Number of files: {len(files)}\n")

    metrics = {}
    
    unique_tasks = set()

    # Compare the json files
    for file in files:
        path1 = dir1 / file                

        data1 = load_json(path1)
        #ipdb.set_trace()
        
        
        if len(data1) == 1:
            # skipped sample, auto fail
            ipdb.set_trace()
        else:
            #ipdb.set_trace()
            try:
                task = data1[0]['input'][-1]['content']
                # check if the task is unique
                if task in unique_tasks:
                    print(f"## {task} is not unique")
                    ipdb.set_trace()
                else:
                    unique_tasks.add(task)

            except Exception as e:
                #ipdb.set_trace()
                pass
            

        data1_results = data1[-1]
        # {'element_acc': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'action_f1': [1.0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0], 'step_success': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'success': [0]}
        #ipdb.set_trace()

        for key in data1_results:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(data1_results[key])
    
    # get macro average of each metric
    for key in metrics:
        _macro = []
        for item in metrics[key]:
            _macro.append(np.mean(item))
        results[key] = np.mean(_macro)

    # Print the macro average:
    for key in results:
        print(f"## {key}: {results[key]}")

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str)
    #parser.add_argument("--dir2", type=str, default="results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/")
    args = parser.parse_args()

    main(args)
