
import ipdb
import os
import json
import argparse


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

    
    results = {'element_acc': [], 'action_f1': [], 'step_success': [], 'success': []}

    # Get the list of files in the directory
    dir1 = Path(args.dir1)    
    files1 = os.listdir(dir1)
    

    # Get the list of files that are in both directories
    files = set(files1)        
    # sort 0.json, 1.json, 2.json, ...
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    print(f"## Number of files in dir1: {len(files1)}")        

    metrics = {}
    

    # Compare the json files
    for file in files:
        path1 = dir1 / file                

        data1 = load_json(path1)

        data1_results = data1[-1]
        # {'element_acc': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'action_f1': [1.0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0], 'step_success': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'success': [0]}
        

        #analyze_prediction(data1)

        metrics[file] = data1_results
        
    
    #ipdb.set_trace()

    # Print the macro average: 
    for file in files:
        #print(f"## {file}")
        for metric_type in ['element_acc', 'action_f1', 'step_success', 'success']:
            #print(f"## {metric_type}")
            #print(f"## {metrics[file][metric_type]}")

            avg = sum(metrics[file][metric_type])/len(metrics[file][metric_type])

            results[metric_type].append(avg)
            #ipdb.set_trace()
    
    # print the average over all files
    for metric_type in ['element_acc', 'action_f1', 'step_success', 'success']:
        average = sum(results[metric_type]) / len(results[metric_type])
        print(f"## {metric_type.capitalize()} Average: {average:.2f}")
    
    ipdb.set_trace()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str)
    #parser.add_argument("--dir2", type=str, default="results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/")
    args = parser.parse_args()

    main(args)
