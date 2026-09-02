import os
from pathlib import Path
import numpy as np
import argparse
import torch

def handle_mlm_path(mlm_path):
    if os.path.isdir(mlm_path):
        mlm_dict = {}
        for sub_path in Path(mlm_path).glob("logit*.pt"):
            sub_dict = torch.load(sub_path, weights_only=False)
            mlm_dict.update(sub_dict)
        return mlm_dict
    else:
        raise ValueError(f"mlm_path {mlm_path} is neither a directory nor a file")

def detect_local_minima(values, position=-2):
    """
    Detect if the value at the given position is a local minima.
    
    Args:
        values: List of values
        position: Position to check (default -2 for second-to-last)
    
    Returns:
        bool: True if local minima is detected, False otherwise
    """
    if len(values) < 3:
        return False
    
    # Convert negative index to positive
    if position < 0:
        position = len(values) + position
    
    # Check if position is valid for local minima (not at boundaries)
    if position <= 0 or position >= len(values) - 1:
        return False
    
    # Check if value at position is less than both neighbors
    return values[position] < values[position - 1] and values[position] < values[position + 1]

def main(dataset, model, case_pattern, alpha=0.4):
    # load MLM score dicts
    mlm_dicts = []
    for mlm_path in sorted(Path(f"intermediates/{dataset}/{model}").glob(f"{case_pattern}*")):
        mlm_dict = handle_mlm_path(mlm_path)
        mlm_dicts.append(mlm_dict)
    
    # calculate the average MLM score for each iteration and apply smoothing (EMA)
    avg_mlm_scores = [np.mean([e for e in mlm_dict.values() if not np.isnan(e)]) for mlm_dict in mlm_dicts]
    smoothed = []
    for u in avg_mlm_scores:
        if not smoothed:
            smoothed.append(u)
        else:
            smoothed.append(alpha * u + (1 - alpha) * smoothed[-1])
    avg_mlm_scores = smoothed

    # Find if the local minima is detected at the second-to-last position
    is_early_stop = detect_local_minima(avg_mlm_scores, position=-2)
    
    print(f"Dataset: {dataset}")
    print(f"Case pattern: {case_pattern}")
    print(f"Number of iterations: {len(avg_mlm_scores)}")
    print(f"Average MLM scores (smoothed): {avg_mlm_scores}")
    print(f"Early stop detected (local minima at second-to-last): {is_early_stop}")
    
    if is_early_stop:
        print(f"Local minima found at iteration {len(avg_mlm_scores) - 2}")
        print(f"Value: {avg_mlm_scores[-2]:.4f}")
        print(f"Previous value: {avg_mlm_scores[-3]:.4f}")
        print(f"Next value: {avg_mlm_scores[-1]:.4f}")
    
    return is_early_stop, avg_mlm_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect early stopping based on local minima in MLM scores")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., fiqa, trec-covid)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., dpr, cocondenser)"
    )
    
    parser.add_argument(
        "--case_pattern",
        type=str,
        required=True,
        help="Pattern to match case directories (e.g., case_1, experiment_*)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Smoothing factor for exponential moving average (default: 0.4)"
    )
    
    args = parser.parse_args()
    
    main(
        dataset=args.dataset,
        case_pattern=args.case_pattern,
        alpha=args.alpha
    )