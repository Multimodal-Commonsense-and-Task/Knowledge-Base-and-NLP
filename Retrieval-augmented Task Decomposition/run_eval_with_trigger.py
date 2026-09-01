
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

    # load trigger map
    trigger_map = {}
    if args.trigger_map:
        with open(args.trigger_map, "r") as f:
            trigger_map = json.load(f)
    
    #ipdb.set_trace()

    #excludes = [13, 18]
    excludes = []

    results = {'element_acc': [], 'action_f1': [], 'step_success': [], 'success': []}

    # Get the list of files in the directory
    dir1 = Path(args.dir1)    
    files1 = os.listdir(dir1)
    
    dir2 = Path(args.dir2)
    files2 = os.listdir(dir2)

    if not Path(args.save_dir).exists():
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

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
        # replace to dir2
        path2 = dir2 / file
        data2 = load_json(path2)
        
        if not len(data1[-1]['element_acc']) == len(data2[-1]['element_acc']):
            print(f"## {file} has different length of actions")
            #ipdb.set_trace()
            ipdb.set_trace()

        if trigger_map[file.split('.json')[0]] == False:
            data1 = data2
            1
            #try:
            #    assert data1[0]['input'][1]['content'] == data2[0]['input'][1]['content']
            #except Exception as e:
            #    print(f"Not able to match {file}")
            
            #data1 = data2
            #ipdb.set_trace()
        
        if not args.save_dir is None:
            #ipdb.set_trace()
            # save the data1 to the save_dir
            save_path = Path(args.save_dir) / file
            #ipdb.set_trace()
            with open(save_path, "w") as f:
                json.dump(data1, f)
        
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

        #analyze_prediction(data1)

        metrics[file] = data1_results
        
    
    #ipdb.set_trace()

    # Print the macro average: 
    for file in files:
        #print(f"## {file}")
        for metric_type in ['element_acc', 'action_f1', 'step_success', 'success']:
            #ipdb.set_trace()
            #print(f"## {metric_type}")
            #print(f"## {metrics[file][metric_type]}")

            avg = sum(metrics[file][metric_type])/len(metrics[file][metric_type])

            results[metric_type].append(avg)
            #ipdb.set_trace()
    


    # print the average over all files
    print(f"## Directory: {dir1}")
    print(f"Average metrics of {len(results['element_acc'])} files")
    for metric_type in ['element_acc', 'action_f1', 'step_success', 'success']:
        average = sum(results[metric_type]) / len(results[metric_type])
        print(f"  {metric_type.capitalize()}: {average:.4f}")
        #ipdb.set_trace()

    #ipdb.set_trace()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str)
    parser.add_argument("--dir2", type=str, default=None)
    parser.add_argument("--trigger_map", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    #parser.add_argument("--dir2", type=str, default="results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/")
    args = parser.parse_args()

    main(args)
