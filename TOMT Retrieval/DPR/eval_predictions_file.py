import os
import json
from collections import defaultdict

from tomt.benchmarks.gt import GTData, get_documents

import argparse
import numpy as np
import pytrec_eval

from tomt.data import utils

metrics_to_compute = {'recip_rank', "recall_1000", 'recall_10', "recall_1", "ndcg_cut_10"}
parser = argparse.ArgumentParser("create_data_dpr")
parser.add_argument("--root", required=True)
parser.add_argument("--dataset", required=True, choices=("Movies", "Books"))
parser.add_argument("--predictions", required=True)
parser.add_argument("--qas_file", required=False)
parser.add_argument("--plot_in_query_file", default="", type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    data = GTData(os.path.join(args.root, args.dataset, "splits", "test"))
    queries = data.get_queries()
    qids = [q["id"] for q in queries]
    qrels = data.get_qrels(True)

    with open(args.predictions, "r") as reader:
        predictions = json.load(reader)

    assert all([qid in predictions for qid in qids])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)

    acc_metrics = defaultdict(list)
    for qid, res in evaluator.evaluate(predictions).items():
        for met in metrics_to_compute:
            acc_metrics[met].append((qid, res[met]))

    mean_metrics = {}

    for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
        vals = [val for (qid, val) in all_vals]
        mean_metrics[met] = {
            "mean": np.mean(vals),
            "std": np.std(vals)
        }
        print(f"{met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})")

    # version 0
    # if args.plot_in_query_file:
    #     plot_in_query = json.load(open(args.plot_in_query_file, "r"))
    #     predictions_w_plot = {k:v for k,v in predictions.items() if plot_in_query[k] == 1} 
    #     # 1. evaluate queries with plot
    #     evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    #     print("# queries: w plot: ", len(predictions_w_plot))
    #     acc_metrics = defaultdict(list)
    #     for qid, res in evaluator.evaluate(predictions_w_plot).items():
    #         for met in metrics_to_compute:
    #             acc_metrics[met].append((qid, res[met]))

    #     mean_metrics = {}

    #     for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
    #         vals = [val for (qid, val) in all_vals]
    #         mean_metrics[met] = {
    #             "mean": np.mean(vals),
    #             "std": np.std(vals)
    #         }
    #         print(f"{met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})")

    #     # 2. evaluate queries without plot
    #     predictions_wo_plot = {k:v for k,v in predictions.items() if plot_in_query[k] == 0}
    #     print("# queries: without plot: ", len(predictions_wo_plot))
    #     evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    #     acc_metrics = defaultdict(list)
    #     for qid, res in evaluator.evaluate(predictions_wo_plot).items():
    #         for met in metrics_to_compute:
    #             acc_metrics[met].append((qid, res[met]))

    #     mean_metrics = {}

    #     for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
    #         vals = [val for (qid, val) in all_vals]
    #         mean_metrics[met] = {
    #             "mean": np.mean(vals),
    #             "std": np.std(vals)
    #         }
    #         print(f"{met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})")

    # # version 1: evaluate queries according to the number of sentences with plot
    # if args.plot_in_query_file:
    #     plot_in_query = json.load(open(args.plot_in_query_file, "r"))
    #     max_num_plots = max(plot_in_query.values())
    #     for num_plots in range(max_num_plots+1):
    #         predictions_w_plot = {k:v for k,v in predictions.items() if plot_in_query[k] == num_plots} 
    #         if len(predictions_w_plot) == 0:
    #             continue
    #         # 1. evaluate queries with plot
    #         evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    #         #print("# queries: ", len(predictions_w_plot), "num_plots: ", num_plots, end=" ")
    #         print(len(predictions_w_plot), num_plots, end=" ")
    #         acc_metrics = defaultdict(list)
    #         for qid, res in evaluator.evaluate(predictions_w_plot).items():
    #             for met in metrics_to_compute:
    #                 acc_metrics[met].append((qid, res[met]))

    #         mean_metrics = {}

    #         for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
    #             vals = [val for (qid, val) in all_vals]
    #             mean_metrics[met] = {
    #                 "mean": np.mean(vals),
    #                 "std": np.std(vals)
    #             }
    #             # print(f"{met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})", end=" ")
    #             print(f"{round(np.mean(vals), 4)}", end=" ")
    #         print()

    # version 2: evaluate queries according to the ratio of sentences with plot
    # if args.plot_in_query_file:
    #     plot_in_query = json.load(open(args.plot_in_query_file, "r"))
    #     num_plots = 0
    #     while num_plots < 1 :
    #         predictions_w_plot = {k:v for k,v in predictions.items() if (num_plots <= plot_in_query[k] and num_plots + 0.2 > plot_in_query[k]) } 
    #         if len(predictions_w_plot) == 0:
    #             continue
    #         # 1. evaluate queries with plot
    #         evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    #         #print("# queries: ", len(predictions_w_plot), "num_plots: ", num_plots, end=" ")
    #         print(str(num_plots)+"~"+str(num_plots+0.2) + "("+ str(len(predictions_w_plot)) + ")", end=" ")
    #         acc_metrics = defaultdict(list)
    #         for qid, res in evaluator.evaluate(predictions_w_plot).items():
    #             for met in metrics_to_compute:
    #                 acc_metrics[met].append((qid, res[met]))

    #         mean_metrics = {}

    #         for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
    #             vals = [val for (qid, val) in all_vals]
    #             mean_metrics[met] = {
    #                 "mean": np.mean(vals),
    #                 "std": np.std(vals)
    #             }
    #             # print(f"{met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})", end=" ")
    #             print(f"{round(np.mean(vals), 4)}", end=" ")
    #         num_plots += 0.2
    #         print()

    """ 
    version 3
    plot_in_query_file: ratio of queries that are plot
    certain_in_query_file: among plots, ratio of queries that are certain

    """
    # if args.plot_in_query_file:
    #     plot_in_query = json.load(open(args.plot_in_query_file, "r"))
    #     certain_in_query = json.load(open("/data1/jongho/anomia/tomt-data/DPR/dataset/Movies/query_classification/ratio_certainamongplot_in_query.json", "r"))
    #     # 1. certain queries
    #     num_plots = 0
    #     print("certain queries")
    #     while num_plots < 1 :
    #         predictions_w_plot = {k:v for k,v in predictions.items() if (num_plots <= plot_in_query[k] and num_plots + 0.2 > plot_in_query[k] and certain_in_query[k] > 0.5) }
    #         if len(predictions_w_plot) == 0:
    #             continue
    #         # 1. evaluate queries with plot
    #         evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    #         #print("# queries: ", len(predictions_w_plot), "num_plots: ", num_plots, end=" ")
    #         print(str(num_plots)+"~"+str(num_plots+0.2) + "("+ str(len(predictions_w_plot)) + ")", end=" ")
    #         acc_metrics = defaultdict(list)
    #         for qid, res in evaluator.evaluate(predictions_w_plot).items():
    #             for met in metrics_to_compute:
    #                 acc_metrics[met].append((qid, res[met]))

    #         mean_metrics = {}

    #         for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
    #             vals = [val for (qid, val) in all_vals]
    #             mean_metrics[met] = {
    #                 "mean": np.mean(vals),
    #                 "std": np.std(vals)
    #             }
    #             # print(f"{met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})", end=" ")
    #             print(f"{round(np.mean(vals), 4)}", end=" ")
    #         num_plots += 0.2
    #         print()
    #     # 2. uncertain queries
    #     num_plots = 0
    #     print("uncertain queries")
    #     while num_plots < 1 :
    #         predictions_w_plot = {k:v for k,v in predictions.items() if (num_plots <= plot_in_query[k] and num_plots + 0.2 > plot_in_query[k] and certain_in_query[k] < 0.5) }
    #         if len(predictions_w_plot) == 0:
    #             continue
    #         # 1. evaluate queries with plot
    #         evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    #         #print("# queries: ", len(predictions_w_plot), "num_plots: ", num_plots, end=" ")
    #         print(str(num_plots)+"~"+str(num_plots+0.2) + "("+ str(len(predictions_w_plot)) + ")", end=" ")
    #         acc_metrics = defaultdict(list)
    #         for qid, res in evaluator.evaluate(predictions_w_plot).items():
    #             for met in metrics_to_compute:
    #                 acc_metrics[met].append((qid, res[met]))

    #         mean_metrics = {}

    #         for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
    #             vals = [val for (qid, val) in all_vals]
    #             mean_metrics[met] = {
    #                 "mean": np.mean(vals),
    #                 "std": np.std(vals)
    #             }
    #             # print(f"{met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})", end=" ")
    #             print(f"{round(np.mean(vals), 4)}", end=" ")
    #         num_plots += 0.2
    #         print()

    """ 
    version 4
    plot_in_query_file: ratio of queries that are plot
    certain_in_query_file: among plots, ratio of queries that are certain

    """