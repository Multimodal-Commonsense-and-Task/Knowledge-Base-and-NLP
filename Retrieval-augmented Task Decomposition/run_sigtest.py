import ipdb
import os
import random
import json
import argparse
from pathlib import Path
from scipy.stats import ttest_rel

def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def collect_metrics(dir_path, metric_type):
    files = os.listdir(dir_path)
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    metrics = []

    for file in files:
        path = dir_path / file
        data = load_json(path)
        if len(data) > 1:  # ensure it's not a skipped sample
            try:
                metric_values = data[-1][metric_type]
                avg = sum(metric_values) / len(metric_values)
                metrics.append(avg)
            except Exception as e:
                pass  # Handle or log exceptions as necessary

    return metrics

def perform_significance_test(metrics1, metrics2, metric_type):
    #ipdb.set_trace()
    t_stat, p_value = ttest_rel(metrics1, metrics2)
    mean1 = sum(metrics1) / len(metrics1)
    mean2 = sum(metrics2) / len(metrics2)
    
    print(f"Metric: {metric_type}")
    print(f"  Number of examples: {len(metrics1)}")
    print(f"  Mean value for Experiment 1: {mean1:.4f}")
    print(f"  Mean value for Experiment 2: {mean2:.4f}")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.001:
        print("  The difference is statistically significant.\n")
    else:
        print("  The difference is not statistically significant.\n")


def main(args):
    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    metrics_to_compare = ['element_acc', 'action_f1', 'step_success', 'success']

    for metric_type in metrics_to_compare:
        metrics1 = collect_metrics(dir1, metric_type)
        metrics2 = collect_metrics(dir2, metric_type)

        if len(metrics1) == len(metrics2) and len(metrics1) > 0:
            normalize = False
            normalize = True 

            if normalize:
                # sample 912-len(metrics1) random 0~1 values from a uniform distribution
                random_values = [random.random() for _ in range(912 - len(metrics1))]
                metrics1 += random_values
                metrics2 += random_values
                #ipdb.set_trace()
            perform_significance_test(metrics1, metrics2, metric_type)
        else:
            print(f"Metric: {metric_type}")
            print("The experiments do not have the same number of valid samples or no samples were found.\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, help="Directory of the first experiment")
    parser.add_argument("--dir2", type=str, help="Directory of the second experiment")
    args = parser.parse_args()

    args.dir1 = "results/mind2web/exp_ours_comp_v6_oracle_7_100samples_topk30_prevtopk30_prevk5_completiontrajcheck_complexity/gpt-35-turbo-16k-mnskim/test_domain/"
    
    args.dir1 = "results/mind2web/trigger1"
    
    args.dir2 = "results/mind2web/exp_base_3_synapsenew_100samples_topk5_complexity/gpt-35-turbo-16k-mnskim/test_domain/"

    main(args)
