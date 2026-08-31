import argparse
import json
import os
from utils import *
import pdb
import re

parser = argparse.ArgumentParser(description="Time-Sensitive QA Evaluation")
parser.add_argument("--prompt-type", type=str, default="sp", choices=["sp", "icl", "stepback", "refine", "batch"], help="Prompt type")
parser.add_argument("--api_call", action="store_true", help="Use API call")
parser.add_argument("--data-path", type=str, default="dataset/test.hard.json", help="Data path")
parser.add_argument("--output-path", type=str, default="output/output.json", help="Output path")
args = parser.parse_args()

def parse_generation(pred):
    original_pred = pred
    '''
    Parse the generated answer based on rules
    '''
    pred = pred.lower()
    if args.prompt_type == "stepback":
        if '\n\nstep 2:' in pred:
            pred = pred.split("\n\nstep 2:")[1]
        if 'final answer:' in pred:
            pred = pred.split('final answer:')[1]
        if '\n' in pred:
            pred = pred.split("\n")[0]
        # elif '\n\n2. ' in pred:
        #     pred = pred.split("\n\n2. ")[0].strip()
        # if 'so,' in pred:
        #     pred = pred.split('so,')[1]
        #     if 'correct answer is' in pred:
        #         pred = pred.split('correct answer is')[1]
        #     elif 'the answer is' in pred:
        #         pred = pred.split('the answer is')[1]
        # elif 'therefore' in pred:
        #     pred = pred.split('therefore')[1]
        # if '\n\n' in pred:
        #     pred = pred.split("\n\n")[0]

        # if "." in pred:
        #     pred = pred.split(".")[0]
    
    elif args.prompt_type == "icl":
        pred = pred.lstrip("answer:")
        if '\n\n' in pred:
            pred = pred.split("\n\n")[0]
        pred = pred.strip()
    
    elif args.prompt_type == "refine":
        if "therefore, the refined answers are:\n" in pred:
            pred = pred.split("therefore, the refined answers are:\n")[1]
            # split by 1. 2. 3. ..
            pred = pred.split("\n")
        # elif "\n" in pred:
        #     pred = pred.split("\n")
        #     pred = [p for p in pred if 'answer' in p]
        # elif "answers" in pred:
        #     pred = pred.split(".")
        #     pred = [p for p in pred if 'answer' in p]
        # elif "\nprediction" in pred:
        #     pred = pred.split("prediction")[1]
        #     pred = pred.split(".")
        #     pred = [p for p in pred if 'answer' in p]
        # if not isinstance(pred, list) or len(pred) == 0:
    elif args.prompt_type == "batch":
        if "step 2" in pred:
            pred = pred.split("step 2")[1]
        elif "step2" in pred:
            pred = pred.split("step2")[1]
        # delete "(e.g. A[1]:, A[2]:,...)" from pred
        if "(e.g. A[1]:, A[2]:,...)" in pred:
            index_eg = pred.index("(e.g. A[1]:, A[2]:,...)")
            pred = pred[:index_eg] + pred[index_eg + len("(e.g. A[1]:, A[2]:,...)"):]
        # answers: A[1]:~, A[2]:~, A[3]:~, ...
        # if such pattern doesn't exist, 1. 2. 3. ...
        pred = pred.split("\n")
        pred1 = pred.copy()
        regex = r"A\[\d+\]:"
        pred = [p for p in pred if re.search(regex, p)]
        if not pred:
            pred = [p for p in pred1 if re.search(r"\d+\.", p)]


    return pred

inputs = []
with open(args.data_path, "r") as f:
    inputs = [json.loads(line) for line in f]
# key: targets, idx
outputs = []
with open(args.output_path, "r") as f:
    outputs = [json.loads(line) for line in f]
# key: idx, answer
# assert idx is same
include_scores = []
exact_scores = []
f1_scores = []

if args.prompt_type in ["refine", "batch"]:
    count = 0
    total_idx_groups = []
    for i in range(len(outputs)):
        predictions = outputs[i]["answer"]
        predictions = parse_generation(predictions)
        idx = outputs[i]["idx"]
        input_idx_groups = [g for g in inputs if g["idx"].split("#")[0] == idx.split("#")[0]]
        if len(predictions) == len(input_idx_groups):
            count += len(predictions)
            total_idx_groups.extend([g["idx"] for g in input_idx_groups])
            targetss = [g["targets"] for g in input_idx_groups]
            for j in range(len(predictions)):
                prediction = predictions[j]
                targets = targetss[j]
                include_score = [int(normalize_answer(a) in normalize_answer(prediction)) for a in targets]
                include_scores.append(max(include_score))
                exact_score = [compute_exact(a, prediction) for a in targets]
                exact_scores.append(max(exact_score))
                f1_score = [compute_f1(a, prediction) for a in targets]
                f1_scores.append(max(f1_score))
    print("count: ", count)
    print("Include: ", sum(include_scores) / len(include_scores) * 100)
    print("Exact Match: ", sum(exact_scores) / len(exact_scores) * 100)
    print("F1: ", sum(f1_scores) / len(f1_scores) * 100)
    include_scores = []
    exact_scores = []
    f1_scores = []
    if "stepback" in args.output_path:
        args.prompt_type = "stepback"
    elif "icl" in args.output_path:
        args.prompt_type = "icl"
    elif "refine" in args.output_path:
        args.prompt_type = "refine"
    elif "batch" in args.output_path:
        args.prompt_type = "icl"
        base_output_path = "output/2024-06-30/output_prompt_icl_Meta-Llama-3-8B-Instruct.json"
    
    if "refine" in args.output_path:
        base_output_path = args.output_path.replace("_refined.json", ".json")
    # args.prompt_type = "icl"
    # base_output_path = "output/2024-06-30/output_prompt_icl_Meta-Llama-3-8B-Instruct.json"
    with open(base_output_path, "r") as f:
        outputs = [json.loads(line) for line in f]
    for i in range(len(outputs)):
        if outputs[i]["idx"] in total_idx_groups:
            prediction = outputs[i]["answer"]
            targets = inputs[i]["targets"]
            prediction = parse_generation(prediction)
            # targets = " ".join(targets)
            # targets = get_tokens(targets)
            # EM score (just check whether the answer is in the targets)
            include_score = [int(normalize_answer(a) in normalize_answer(prediction)) for a in targets]
            include_scores.append(max(include_score))
            # Exact match score
            exact_score = [compute_exact(a, prediction) for a in targets]
            exact_scores.append(max(exact_score))
            # F1 score
            f1_score = [compute_f1(a, prediction) for a in targets]
            f1_scores.append(max(f1_score))
    # evaluate the 
    print("Include: ", sum(include_scores) / len(include_scores) * 100)
    print("Exact Match: ", sum(exact_scores) / len(exact_scores) * 100)
    print("F1: ", sum(f1_scores) / len(f1_scores) * 100)

else:
    for i in range(len(outputs)):
        assert inputs[i]["idx"] == outputs[i]["idx"]
        prediction = outputs[i]["answer"]
        targets = inputs[i]["targets"] # list of answers
        # parse the prediction based on rules
        prediction = parse_generation(prediction)
        # targets = " ".join(targets)
        # targets = get_tokens(targets)
        # EM score (just check whether the answer is in the targets)
        include_score = [int(normalize_answer(a) in normalize_answer(prediction)) for a in targets]
        include_scores.append(max(include_score))
        # Exact match score
        exact_score = [compute_exact(a, prediction) for a in targets]
        exact_scores.append(max(exact_score))
        # F1 score
        f1_score = [compute_f1(a, prediction) for a in targets]
        f1_scores.append(max(f1_score))

    print("Include: ", sum(include_scores) / len(include_scores) * 100)
    print("Exact Match: ", sum(exact_scores) / len(exact_scores) * 100)
    print("F1: ", sum(f1_scores) / len(f1_scores) * 100)
# print("Exact Match: ", sum(include_scores) / len(include_scores) * 100)



