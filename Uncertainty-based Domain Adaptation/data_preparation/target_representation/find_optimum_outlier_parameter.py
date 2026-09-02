from sklearn.cluster import DBSCAN, HDBSCAN
import numpy as np
import json
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
import faiss
from tqdm import tqdm
import argparse
from numpy import dot
import math
import random
from numpy.linalg import norm
from collections import OrderedDict, defaultdict
from pyserini.search.lucene import LuceneSearcher
from sklearn.model_selection import ParameterGrid

special_datasets_list = ["signal1m", "hotpotqa", "quora", "climate-fever"]

# SEED_LIST = [1953, 883, 2, 9312, 56]
SEED_LIST = [683, 771, 51, 352, 19]
# SEED_LIST = [35, 745, 10, 6534, 2]
N_DOCTEXT_FILTER = 300
N_ITER = 100
T = 1.0
VERBOSE = True

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1'}:
        return True
    elif value.lower() in {'false', '0'}:
        return False
    else:
        raise ValueError(f"{value} is not a valid boolean value")

cosine_fn = lambda A, B: dot(A, B) / (norm(A)*norm(B))


def argmax(keys, f):
    return max(keys, key=f)


def get_seeditem_item_simscore_dict(seed_docid, docids, dataset_doc_emb_data_dict):
    seeditem_item_rel_dict = defaultdict()
    for docid in docids:
        seeditem_item_rel_dict[docid] = float( cosine_fn(dataset_doc_emb_data_dict[docid].numpy(), dataset_doc_emb_data_dict[seed_docid].numpy()) )
    return seeditem_item_rel_dict


def get_item_item_simscore_dict(docids, dataset_doc_emb_data_dict):
    item_item_cos_dict = defaultdict(dict)
    for i in docids:
        for j in docids:
            cos_sim = float( cosine_fn(dataset_doc_emb_data_dict[i].numpy(), dataset_doc_emb_data_dict[j].numpy()) )
            item_item_cos_dict[i][j] = cos_sim
    return item_item_cos_dict


def mmr_sorted(docs, lambda_, similarity1, similarity2):
    selected = OrderedDict()
    doc_score_dict = defaultdict()
    while set(selected) != docs:
        remaining = docs - set(selected)

        mmr_score = lambda x: lambda_*similarity1[x] - (1-lambda_)*max([similarity2[x][y] for y in set(selected)-{x}] or [0])
        mmr_score_values = np.array([mmr_score(doc) for doc in remaining])

        next_selected = argmax(remaining, mmr_score)
        next_selected_score = np.max(mmr_score_values)

        doc_score_dict[next_selected] = next_selected_score

        selected[next_selected] = len(selected)

    return selected, dict(doc_score_dict)


def passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                      n_selection, seed_docid, docids, lambda_val=0.5):
    seeddoc_doc_rel_dict = get_seeditem_item_simscore_dict(seed_docid, docids, dataset_doc_emb_data_dict)

    doc_doc_rel_dict = get_item_item_simscore_dict(docids, dataset_doc_emb_data_dict)

    _, output_scores_dict = mmr_sorted(set(docids), lambda_val, seeddoc_doc_rel_dict, doc_doc_rel_dict)

    sort_orders = sorted(output_scores_dict.items(), key=lambda x: x[1], reverse=True)
    return sort_orders[:n_selection]


def get_doc_text(docid, searcher):
    try:
        doctext = searcher.doc(docid).raw()

        doctext_str = doctext.split('"text" : "')[-1].split('"metadata')[0].strip()
        if doctext_str[-2:] == '",':
            doctext_str = doctext_str.replace('",', '').strip()

        doctitle_str = doctext.split('"title" : "')[-1].split('"text')[0].strip()
        if doctitle_str[-2:] == '",':
            doctitle_str = doctitle_str.replace('",', '').strip()

        return doctitle_str + ' ' + doctext_str
    except AttributeError:
        return None


def dbscan_outlier(data, eps=0.5, min_samples=5):
    """
    DBSCAN outlier detection
    :param data: input data
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: outlier label
    """
    # DBSCAN outlier detection
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=8)
    outlier_label = dbscan.fit_predict(data)
    return outlier_label

def hdbscan_outlier(data, min_cluster_size=5, min_samples=5):
    """
    HDBSCAN outlier detection
    :param data: input data
    :param min_cluster_size: The minimum size of clusters.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: outlier label
    """
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    outlier_label = hdbscan.fit_predict(data)
    return outlier_label

def vae_outlier_detection(vae, data_loader, device):
    pass

def iforest_outlier_detection(data, n_estimators=500, contamination=0.1):
    """
    Isolation Forest outlier detection
    :param data: input data
    :param n_estimators: The number of base estimators in the ensemble.
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    :return: outlier label
    """
    # Isolation Forest outlier detection
    iforest = IsolationForest(n_estimators=n_estimators,
                              max_samples="auto",
                              contamination=contamination,
                              max_features=1.0,
                              random_state=42,
                              n_jobs=8)
    outlier_label = iforest.fit_predict(data)
    return outlier_label


def main_run(dataset_name, collection_text_filepath, collection_embedding_filepath, remove_outlier=None, normalize=False, target_ratios=[0.99,0.95]):
    lexical_index_name = f'beir-v1.0.0-{dataset_name}.multifield' # pyserini multi-field index name
    ########################################################################
    #                   Load collection embedding
    ########################################################################
    # 1. Load collection documents and filter out noisy documents
    filtered_docid_doctext_dict = {}
    for line in open(collection_text_filepath):
        data = json.loads(line)
        docid = data['docid']
        doctext = data['doctext']
        filtered_docid_doctext_dict[docid] = doctext

    # 2. Load document embedding
    dataset_doc_emb_data_dict = {k: v for k, v in torch.load(collection_embedding_filepath, weights_only=True).items() if filtered_docid_doctext_dict.get(k) }

    target_docids = list(dataset_doc_emb_data_dict.keys())
    target_docids_embs = torch.stack( [dataset_doc_emb_data_dict[docid] for docid in target_docids] )

    # Normalize the embeddings
    if normalize:
        normalized_target_docids_embs = F.normalize(target_docids_embs, p=2, dim=1)
    else:
        normalized_target_docids_embs = target_docids_embs
    print('... completed collection loading')
    n_corpus = len(target_docids)
    print(f'Number of documents in the collection: {n_corpus}')

    # outlier_label = dbscan_outlier(target_docids_embs, eps=1, min_samples=3)

    # if remove_outlier == 'dbscan':
    #     # Get Outliers from normalized embeddings
    #     outlier_label = dbscan_outlier(normalized_target_docids_embs, eps=epsilon, min_samples=min_samples)
    #     # get index of outliers(where label is -1)
    #     # outlier_idx = np.where(outlier_label == -1)[0]
    #     nonoutlier_idx = np.where(outlier_label != -1)[0]

    #     nonoutlier_target_docids_embs = target_docids_embs[nonoutlier_idx]
    # elif remove_outlier == 'iforest':
    #     # Get Outliers from normalized embeddings
    #     outlier_label = iforest_outlier_detection(normalized_target_docids_embs, n_estimators=n_estimators, contamination=contamination)
    #     # get index of outliers(where label is -1)
    #     # outlier_idx = np.where(outlier_label == -1)[0]
    #     nonoutlier_idx = np.where(outlier_label != -1)[0]

    #     nonoutlier_target_docids_embs = target_docids_embs[nonoutlier_idx]
    # else:
    #     nonoutlier_target_docids_embs = target_docids_embs
    
    if remove_outlier == 'dbscan':
        # parameter_grid = {
        #     'eps': [0.92, 0.93],
        #     # 'min_samples': [3, 5, 10]
        #     'min_samples': [5]
        # }

        # grid = ParameterGrid(parameter_grid)
        # best_params = None
        # best_score = None

        temp_results = {}
        print(f"Target Ratios: {target_ratios}")
        for target_ratio in target_ratios:
            eps = 7
            stride = 0.1

            if eps not in temp_results:
                outlier_label = dbscan_outlier(normalized_target_docids_embs, eps=eps, min_samples=5)
                nonoutlier_idx = np.where(outlier_label != -1)[0]
                ratio = len(nonoutlier_idx) / n_corpus
                temp_results[eps] = ratio
            else:
                ratio = temp_results[eps]
            previous_diff = target_ratio - ratio
            print(f"Initial Non-Outlier Ratio: {ratio}: {previous_diff}")

            while abs(previous_diff) > 0.01:
                print(f"bigger than 0.01 : {abs(previous_diff)}")
                eps += (stride if previous_diff > 0 else -stride)
                if eps not in temp_results:
                    outlier_label = dbscan_outlier(normalized_target_docids_embs, eps=eps, min_samples=5)
                    nonoutlier_idx = np.where(outlier_label != -1)[0]
                    ratio = len(nonoutlier_idx) / n_corpus
                    temp_results[eps] = ratio
                else:
                    ratio = temp_results[eps]
                current_diff = target_ratio - ratio
                print(f"eps: {eps:.2f}")
                print(f"Non-Outlier Ratio: {ratio}: {current_diff}")

                if np.sign(current_diff) != np.sign(previous_diff):
                    stride = 0.01
                    eps += stride if previous_diff > 0 else -stride
                
                previous_diff = current_diff
            print(f"Target Ratio: {target_ratio}")
            print(f"Parameters: {eps:.2f}")
            print(f"Non-Outlier Ratio: {ratio}")
                # if best_score is None or len(nonoutlier_target_docids_embs) > best_score:
                #     best_score = len(nonoutlier_target_docids_embs)
                #     best_params = params
        
        # print(f"Best Parameters: {best_params}")
    elif remove_outlier == 'hdbscan':
        parameter_grid = {
            'min_cluster_size': [5],
            'min_samples': [5]
        }

        grid = ParameterGrid(parameter_grid)
        best_params = None

        for params in grid:
            outlier_label = hdbscan_outlier(normalized_target_docids_embs, min_cluster_size=params['min_cluster_size'], min_samples=params['min_samples'])
            nonoutlier_idx = np.where(outlier_label != -1)[0]
            ratio = len(nonoutlier_idx) / n_corpus

            print(f"Parameters: {params}")
            print(f"Non-Outlier Ratio: {ratio}")
        
    elif remove_outlier == 'iforest':
        parameter_grid = {
            'n_estimators': [200, 500, 1000, 1500],
            'contamination': [0.1, 0.15, 0.2, 0.25]
        }

        grid = ParameterGrid(parameter_grid)
        best_params = None
        best_score = None

        for params in grid:
            iso_forest = IsolationForest(n_estimators=params['n_estimators'],
                                        contamination=params['contamination'], random_state=42)
            iso_forest.fit(normalized_target_docids_embs)
            anomaly_score = iso_forest.decision_function(normalized_target_docids_embs)
            score_variance = np.var(anomaly_score)
            if best_score is None or score_variance < best_score:
                best_score = score_variance
                best_params = params
        
        print(f"Best Parameters: {best_params}")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Document Sampling')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name to be specific')
    parser.add_argument('--collection_text_filepath', required=True, type=str,
                        help='file path to document collection text')
    parser.add_argument('--collection_embedding_filepath', required=True, type=str,
                        help='file path to document collection embedding')
    parser.add_argument('--remove_outlier', type=str, required=False,
                    help='removing outlier with method: dbscan or iforest')
    parser.add_argument('--normalize', type=str_to_bool, required=False,
                        help='normalize the embeddings')
    parser.add_argument('--target_ratios', type=float, nargs='+', required=False,
                        help='target ratio of non-outlier documents')
    args = parser.parse_args()

    main_run(args.dataset_name, args.collection_text_filepath, args.collection_embedding_filepath, args.remove_outlier, args.normalize, args.target_ratios)