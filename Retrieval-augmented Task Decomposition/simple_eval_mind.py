import pickle
import logging
import argparse
import os
import sys
import json
from tqdm import tqdm
#import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
#from peft import PeftModel

#from synapse.envs.mind2web.env_utils import load_json
#from synapse.agents.mind2web import eval_sample_llama

import ipdb

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        "--benchmark", type=str, choices=["test_task", "test_website", "test_domain"]
    )
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=3)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--no_memory", action="store_true", default=False)
    parser.add_argument("--no_trajectory", action="store_true", default=False)
    parser.add_argument("--multi_choice", action="store_true", default=False)

    return parser

def load_json(data_dir, folder_name):
    folder_path = os.path.join(data_dir, folder_name)
    print(f"Data path: {folder_path}")
    data_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".json")
    ]
    data_paths = sorted(data_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Construct trajectory dataset
    samples = []
    for data_path in data_paths:
        with open(data_path, "r") as f:
            samples.extend(json.load(f))
    print("# of samples:", len(samples))

    return samples

def main():
    #parser = create_parser()
    #args = parser.parse_args()
    #current_path = os.getcwd()
    #args.memory_path = os.path.join(current_path, "synapse/memory/mind2web")
    #args.log_dir = os.path.join(current_path, "results/mind2web")

    #ipdb.set_trace()

    # Evaluate test set
    #assert args.benchmark in ["test_task", "test_website", "test_domain"]
    benchmark = "test_domain"
    data_dir = "/home/mnskim/workspace/web/Synapse/data"

    data_dir = "results/mind2web/exp_ours_comp_v6_oracle_7_100samples_topk30_prevtopk30_prevk5_completiontrajcheck_complexity/gpt-35-turbo-16k-mnskim/"
    benchmark = "test_domain"

    samples = load_json(data_dir, benchmark)

    # add prediction scores and ranks to candidates
    with open(os.path.join(data_dir, "scores_all_data.pkl"), "rb") as f:
        candidate_results = pickle.load(f)

    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]
    for sample in samples:
        for s in sample["actions"]:
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]

    ipdb.set_trace()
    

if __name__ == "__main__":
    main()
