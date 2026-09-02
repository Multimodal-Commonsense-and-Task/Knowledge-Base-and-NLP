
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

    # Get the list of files in the directory
    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Get the list of files that are in both directories
    files1 = set(files1)
    files2 = set(files2)
    files = files1.intersection(files2)
    # sort 0.json, 1.json, 2.json, ...
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    print(f"## Number of files in dir1: {len(files1)}")
    print(f"## Number of files in dir2: {len(files2)}")
    print(f"## Number of files in both dirs: {len(files)}")

    # Compare the json files
    for file in files:
        path1 = dir1 / file
        path2 = dir2 / file

        print(f"## Comparing {path1} and {path2}")

        data1 = load_json(path1)

        data1_results = data1[-1]
        # {'element_acc': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'action_f1': [1.0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0], 'step_success': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'success': [0]}

        analyze_prediction(data1)

        data2 = load_json(path2)
    
        ipdb.set_trace()










if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, default="./results/mind2web/debugging_nar2_lookahead4/gpt-3.5-turbo-0613/test_task/")
    parser.add_argument("--dir2", type=str, default="results/mind2web/exp_base2/gpt-3.5-turbo-0613/test_task/")
    args = parser.parse_args()

    main(args)
