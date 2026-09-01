from sklearn.cluster import DBSCAN
import os
import numpy as np
import json
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from hdbscan import HDBSCAN
import faiss
from tqdm import tqdm
import argparse
from numpy import dot
import math
import random
from numpy.linalg import norm
from collections import OrderedDict, defaultdict
from pyserini.search.lucene import LuceneSearcher
import time
from kmedoids import KMedoids
from sklearn_extra.cluster import CLARA

special_datasets_list = ["signal1m", "hotpotqa", "quora", "climate-fever"]

SEED_LIST = [683, 771, 51, 352, 19]
# SEED_LIST = [1953, 883, 2, 9312, 56]
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
    # Measure time taken for DBSCAN
    start_time = time.time()
    dbscan.fit_predict(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    return dbscan


# def kmedoid_outlier(data, n_clusters=)


def hdbscan_outlier(data, min_cluster_size=5, min_samples=5):
    """
    HDBSCAN outlier detection
    :param data: input data
    :param min_cluster_size: The minimum size of clusters.
    :param min_samples: The number of samples in a neighbourhood for a point to be considered as a core point.
    :return: outlier label
    """
    # HDBSCAN outlier detection
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    # Measure time taken for HDBSCAN
    start_time = time.time()
    outlier_label = hdbscan.fit_predict(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    return outlier_label

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
    # Measure time taken for Isolation Forest
    start_time = time.time()
    outlier_label = iforest.fit_predict(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    return outlier_label


def clustering_sample(target_docids_embs, target_docids, n_clusters, n_train, n_corpus, 
                      dataset_doc_emb_data_dict, lexical_index_name, lambda_val=1.0):
        ########################################################################
    #                   Apply clustering on collection
    ########################################################################
    # 3. Train and index clustering algorithm
    d = target_docids_embs.shape[1]
    target_docids_embs = target_docids_embs.cpu().numpy()
    kmeans = faiss.Kmeans(d, n_clusters, niter=N_ITER, verbose=VERBOSE, gpu=True, seed=420)
    kmeans.train(target_docids_embs)

    index = faiss.IndexFlatL2 (d)
    index.add ( target_docids_embs )
    D, I = index.search (kmeans.centroids, 1)

    # 4. Find docids closes to each cluster centroid
    centroid_nearest_docids_list = []
    for centroid_nearest_idxs, _ in zip(I, D):
        centroid_nearest_docid = target_docids[centroid_nearest_idxs[0]]
        centroid_nearest_docids_list.append( centroid_nearest_docid )

    D_inv, I_inv = kmeans.index.search(target_docids_embs, 1)

    # 5. Find cluster size (number of documents in each cluster)
    docid_clusteridx_dict = {}
    clusteridx_docids_dict = defaultdict(list)
    cluster_size_dict = defaultdict(int)
    for docid, cluster_idx in zip(target_docids, I_inv):
        docid_clusteridx_dict[docid] = cluster_idx[0]
        clusteridx_docids_dict[cluster_idx[0]].append( docid )
        cluster_size_dict[ cluster_idx[0] ] += 1

    cluster_size_dict_sorted = sorted(cluster_size_dict.items(), key=lambda x: x[1], reverse=False)
    print('... completed collection clustering')

    ########################################################################
    #                   Determine sample size for each cluster
    ########################################################################
    # 6. Find initial round of sample size for each cluster
    clusteridx_samplesize_dict = defaultdict(int)
    sample_cnt = 0
    for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted):
        sample_size = 1 + int(math.floor( (n_train - n_clusters) * (size / n_corpus) ))
        clusteridx_samplesize_dict[cluster_idx] = sample_size
        sample_cnt += sample_size

    # 7. Sample more from highly populated clusters
    n_remaining = n_train - sample_cnt
    if n_train != n_clusters:
        for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted[-n_remaining::]):
            clusteridx_samplesize_dict[cluster_idx] += 1
            sample_cnt += 1

    # 8. Derive cosine similarity (~distance) for each documents
    clusterids_docids_dist_cosdict = defaultdict(dict)
    for docid, cluster_idx in zip(target_docids, I_inv):
        cluster_centroid = kmeans.centroids[cluster_idx[0]]
        docemb = dataset_doc_emb_data_dict[docid].numpy()

        cos_distance = cosine_fn(docemb, cluster_centroid)
        clusterids_docids_dist_cosdict[cluster_idx[0]][docid] = float(cos_distance)

    cluster_idx_list = list(clusteridx_samplesize_dict.keys())
    random.shuffle(cluster_idx_list)

    ########################################################################
    #                   Sample documents from each cluster
    ########################################################################
    # 9. Probabilistic sampling based on document distance
    total_sample_size = 0
    cluster_idx_set = set()
    docids_set = set()
    all_sampled_docids_set = set()
    for cluster_idx in tqdm(cluster_idx_list, total=len(clusteridx_samplesize_dict)):
        sample_size = clusteridx_samplesize_dict[cluster_idx]
        total_sample_size += sample_size
        cluster_idx_set.add(cluster_idx)

        # (1) find docids belong to cluster
        curr_docids = clusteridx_docids_dict[cluster_idx]
        for docid in curr_docids:
            docids_set.add(docid)

        # (2) get cosine-similarity for each docid
        curr_docid_distances = [clusterids_docids_dist_cosdict[cluster_idx][docid] for docid in curr_docids]

        # (3) define probabilities based on distance
        prob_values = [e/T for e in curr_docid_distances]
        curr_docid_probs = np.exp(prob_values) / np.sum(np.exp(prob_values), axis=0)
        assert np.sum(curr_docid_probs) >= 0.99

        # (4) random sample with probabilities
        curr_selected_docids_pool = set()
        for seed in SEED_LIST:
            np.random.seed(seed)
            sampled_docids = np.random.choice(curr_docids, p=curr_docid_probs, size=sample_size, replace=False)
            for docid in sampled_docids:
                curr_selected_docids_pool.add(str(docid))

        # (5) apply MMR and pick top-sample_size documents
        closest_to_centroid_docid = centroid_nearest_docids_list[cluster_idx]
        final_sampled_docids_info = passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                                                      n_selection=sample_size, seed_docid=closest_to_centroid_docid, docids=curr_selected_docids_pool,
                                                      lambda_val=lambda_val)
        final_sampled_docids = [e[0] for e in final_sampled_docids_info]

        all_sampled_docids_set = all_sampled_docids_set.union(set(final_sampled_docids))
    print('... completed document sampling')


    ########################################################################
    #                   Find document-text for the sampled documents
    ########################################################################
    # 10. Look-up Pyserini index to fetch the document-text
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    target_docid_doctext_list = []
    for docid in all_sampled_docids_set:
        doctext = get_doc_text( docid, searcher=searcher)
        if not doctext:
            continue
        target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
    print('... completed document text lookup')
    return target_docid_doctext_list

def kmedoid_clustering_sample(target_docids_embs, target_docids, n_clusters, n_train, n_corpus, 
                      dataset_doc_emb_data_dict, lexical_index_name, lambda_val=1.0):
        ########################################################################
    #                   Apply clustering on collection
    ########################################################################
    # 3. Train and index clustering algorithm
    if isinstance(target_docids_embs, torch.Tensor):
        target_docids_embs = target_docids_embs.cpu().numpy()
    
    kmedoids = KMedoids(n_clusters=n_clusters, max_iter=N_ITER, random_state=420, metric='euclidean')
    kmedoids.fit(target_docids_embs)

    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_

    centroid_nearest_docids_list = [target_docids[idx] for idx in medoid_indices]

    # 5. Find cluster size (number of documents in each cluster)
    docid_clusteridx_dict = {}
    clusteridx_docids_dict = defaultdict(list)
    cluster_size_dict = defaultdict(int)
    for docid, cluster_idx in zip(target_docids, labels):
        docid_clusteridx_dict[docid] = cluster_idx
        clusteridx_docids_dict[cluster_idx].append(docid)
        cluster_size_dict[cluster_idx] += 1

    cluster_size_dict_sorted = sorted(cluster_size_dict.items(), key=lambda x: x[1], reverse=False)
    print('... completed collection clustering')

    ########################################################################
    #                   Determine sample size for each cluster
    ########################################################################
    # 6. Find initial round of sample size for each cluster
    clusteridx_samplesize_dict = defaultdict(int)
    sample_cnt = 0
    for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted):
        sample_size = 1 + int(math.floor( (n_train - n_clusters) * (size / n_corpus) ))
        clusteridx_samplesize_dict[cluster_idx] = sample_size
        sample_cnt += sample_size

    # 7. Sample more from highly populated clusters
    n_remaining = n_train - sample_cnt
    if n_train != n_clusters:
        for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted[-n_remaining::]):
            clusteridx_samplesize_dict[cluster_idx] += 1
            sample_cnt += 1

    # 8. Derive cosine similarity (~distance) for each documents
    clusterids_docids_dist_cosdict = defaultdict(dict)
    for docid, cluster_idx in zip(target_docids, labels):
        medoid_idx = medoid_indices[cluster_idx]
        cluster_medoid_emb = target_docids_embs[medoid_idx]
        docemb = dataset_doc_emb_data_dict[docid].numpy()

        cos_distance = cosine_fn(docemb, cluster_medoid_emb)
        clusterids_docids_dist_cosdict[cluster_idx][docid] = float(cos_distance)

    cluster_idx_list = list(clusteridx_samplesize_dict.keys())
    random.shuffle(cluster_idx_list)

    ########################################################################
    #                   Sample documents from each cluster
    ########################################################################
    # 9. Probabilistic sampling based on document distance
    total_sample_size = 0
    cluster_idx_set = set()
    docids_set = set()
    all_sampled_docids_set = set()
    for cluster_idx in tqdm(cluster_idx_list, total=len(clusteridx_samplesize_dict)):
        sample_size = clusteridx_samplesize_dict[cluster_idx]
        total_sample_size += sample_size
        cluster_idx_set.add(cluster_idx)

        # (1) find docids belong to cluster
        curr_docids = clusteridx_docids_dict[cluster_idx]
        for docid in curr_docids:
            docids_set.add(docid)

        # (2) get cosine-similarity for each docid
        curr_docid_distances = [clusterids_docids_dist_cosdict[cluster_idx][docid] for docid in curr_docids]

        # (3) define probabilities based on distance
        prob_values = [e/T for e in curr_docid_distances]
        curr_docid_probs = np.exp(prob_values) / np.sum(np.exp(prob_values), axis=0)
        assert np.sum(curr_docid_probs) >= 0.99

        # (4) random sample with probabilities
        curr_selected_docids_pool = set()
        for seed in SEED_LIST:
            np.random.seed(seed)
            sampled_docids = np.random.choice(curr_docids, p=curr_docid_probs, size=sample_size, replace=False)
            for docid in sampled_docids:
                curr_selected_docids_pool.add(str(docid))

        # (5) apply MMR and pick top-sample_size documents
        closest_to_centroid_docid = centroid_nearest_docids_list[cluster_idx]
        final_sampled_docids_info = passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                                                      n_selection=sample_size, seed_docid=closest_to_centroid_docid, docids=curr_selected_docids_pool,
                                                      lambda_val=lambda_val)
        final_sampled_docids = [e[0] for e in final_sampled_docids_info]

        all_sampled_docids_set = all_sampled_docids_set.union(set(final_sampled_docids))
    print('... completed document sampling')


    ########################################################################
    #                   Find document-text for the sampled documents
    ########################################################################
    # 10. Look-up Pyserini index to fetch the document-text
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    target_docid_doctext_list = []
    for docid in all_sampled_docids_set:
        doctext = get_doc_text( docid, searcher=searcher)
        if not doctext:
            continue
        target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
    print('... completed document text lookup')
    return target_docid_doctext_list

def kmedoid_clustering_sample(target_docids_embs, target_docids, n_clusters, n_train, n_corpus, 
                      dataset_doc_emb_data_dict, lexical_index_name, lambda_val=1.0):
        ########################################################################
    #                   Apply clustering on collection
    ########################################################################
    # 3. Train and index clustering algorithm
    if isinstance(target_docids_embs, torch.Tensor):
        target_docids_embs = target_docids_embs.cpu().numpy()
    
    start_time = time.time()
    kmedoids = KMedoids(n_clusters=n_clusters, max_iter=N_ITER, random_state=420, metric='euclidean')
    kmedoids.fit(target_docids_embs)
    print(f'... completed KMedoids clustering in {time.time() - start_time} seconds')

    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_

    centroid_nearest_docids_list = [target_docids[idx] for idx in medoid_indices]

    # 5. Find cluster size (number of documents in each cluster)
    docid_clusteridx_dict = {}
    clusteridx_docids_dict = defaultdict(list)
    cluster_size_dict = defaultdict(int)
    for docid, cluster_idx in zip(target_docids, labels):
        docid_clusteridx_dict[docid] = cluster_idx
        clusteridx_docids_dict[cluster_idx].append(docid)
        cluster_size_dict[cluster_idx] += 1

    cluster_size_dict_sorted = sorted(cluster_size_dict.items(), key=lambda x: x[1], reverse=False)
    print('... completed collection clustering')

    ########################################################################
    #                   Determine sample size for each cluster
    ########################################################################
    # 6. Find initial round of sample size for each cluster
    clusteridx_samplesize_dict = defaultdict(int)
    sample_cnt = 0
    for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted):
        sample_size = 1 + int(math.floor( (n_train - n_clusters) * (size / n_corpus) ))
        clusteridx_samplesize_dict[cluster_idx] = sample_size
        sample_cnt += sample_size

    # 7. Sample more from highly populated clusters
    n_remaining = n_train - sample_cnt
    if n_train != n_clusters:
        for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted[-n_remaining::]):
            clusteridx_samplesize_dict[cluster_idx] += 1
            sample_cnt += 1

    # 8. Derive cosine similarity (~distance) for each documents
    clusterids_docids_dist_cosdict = defaultdict(dict)
    for docid, cluster_idx in zip(target_docids, labels):
        medoid_idx = medoid_indices[cluster_idx]
        cluster_medoid_emb = target_docids_embs[medoid_idx]
        docemb = dataset_doc_emb_data_dict[docid].numpy()

        cos_distance = cosine_fn(docemb, cluster_medoid_emb)
        clusterids_docids_dist_cosdict[cluster_idx][docid] = float(cos_distance)

    cluster_idx_list = list(clusteridx_samplesize_dict.keys())
    random.shuffle(cluster_idx_list)

    ########################################################################
    #                   Sample documents from each cluster
    ########################################################################
    # 9. Probabilistic sampling based on document distance
    total_sample_size = 0
    cluster_idx_set = set()
    docids_set = set()
    all_sampled_docids_set = set()
    for cluster_idx in tqdm(cluster_idx_list, total=len(clusteridx_samplesize_dict)):
        sample_size = clusteridx_samplesize_dict[cluster_idx]
        total_sample_size += sample_size
        cluster_idx_set.add(cluster_idx)

        # (1) find docids belong to cluster
        curr_docids = clusteridx_docids_dict[cluster_idx]
        for docid in curr_docids:
            docids_set.add(docid)

        # (2) get cosine-similarity for each docid
        curr_docid_distances = [clusterids_docids_dist_cosdict[cluster_idx][docid] for docid in curr_docids]

        # (3) define probabilities based on distance
        prob_values = [e/T for e in curr_docid_distances]
        curr_docid_probs = np.exp(prob_values) / np.sum(np.exp(prob_values), axis=0)
        assert np.sum(curr_docid_probs) >= 0.99

        # (4) random sample with probabilities
        curr_selected_docids_pool = set()
        for seed in SEED_LIST:
            np.random.seed(seed)
            sampled_docids = np.random.choice(curr_docids, p=curr_docid_probs, size=sample_size, replace=False)
            for docid in sampled_docids:
                curr_selected_docids_pool.add(str(docid))

        # (5) apply MMR and pick top-sample_size documents
        closest_to_centroid_docid = centroid_nearest_docids_list[cluster_idx]
        final_sampled_docids_info = passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                                                      n_selection=sample_size, seed_docid=closest_to_centroid_docid, docids=curr_selected_docids_pool,
                                                      lambda_val=lambda_val)
        final_sampled_docids = [e[0] for e in final_sampled_docids_info]

        all_sampled_docids_set = all_sampled_docids_set.union(set(final_sampled_docids))
    print('... completed document sampling')


    ########################################################################
    #                   Find document-text for the sampled documents
    ########################################################################
    # 10. Look-up Pyserini index to fetch the document-text
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    target_docid_doctext_list = []
    for docid in all_sampled_docids_set:
        doctext = get_doc_text( docid, searcher=searcher)
        if not doctext:
            continue
        target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
    print('... completed document text lookup')
    return target_docid_doctext_list


def clara_clustering_sample(target_docids_embs, target_docids, n_clusters, n_train, n_corpus, 
                      dataset_doc_emb_data_dict, lexical_index_name, lambda_val=1.0):
    ########################################################################
    #                   Apply CLARA clustering on collection
    ########################################################################
    # 3. Train and index clustering algorithm
    if isinstance(target_docids_embs, torch.Tensor):
        target_docids_embs = target_docids_embs.cpu().numpy()
    
    print('... start CLARA clustering')
    start_time = time.time()
    clara = CLARA(n_clusters=n_clusters, max_iter=N_ITER, random_state=420, metric='euclidean')
    clara.fit(target_docids_embs)
    print(f'... completed CLARA clustering in {time.time() - start_time} seconds')

    labels = clara.labels_
    medoid_indices = clara.medoid_indices_

    centroid_nearest_docids_list = [target_docids[idx] for idx in medoid_indices]

    # 5. Find cluster size (number of documents in each cluster)
    docid_clusteridx_dict = {}
    clusteridx_docids_dict = defaultdict(list)
    cluster_size_dict = defaultdict(int)
    for docid, cluster_idx in zip(target_docids, labels):
        docid_clusteridx_dict[docid] = cluster_idx
        clusteridx_docids_dict[cluster_idx].append(docid)
        cluster_size_dict[cluster_idx] += 1

    cluster_size_dict_sorted = sorted(cluster_size_dict.items(), key=lambda x: x[1], reverse=False)
    print('... completed collection clustering')

    ########################################################################
    #                   Determine sample size for each cluster
    ########################################################################
    # 6. Find initial round of sample size for each cluster
    clusteridx_samplesize_dict = defaultdict(int)
    sample_cnt = 0
    for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted):
        sample_size = 1 + int(math.floor( (n_train - n_clusters) * (size / n_corpus) ))
        clusteridx_samplesize_dict[cluster_idx] = sample_size
        sample_cnt += sample_size

    # 7. Sample more from highly populated clusters
    n_remaining = n_train - sample_cnt
    if n_train != n_clusters:
        for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted[-n_remaining::]):
            clusteridx_samplesize_dict[cluster_idx] += 1
            sample_cnt += 1

    # 8. Derive cosine similarity (~distance) for each documents
    clusterids_docids_dist_cosdict = defaultdict(dict)
    for docid, cluster_idx in zip(target_docids, labels):
        medoid_idx = medoid_indices[cluster_idx]
        cluster_medoid_emb = target_docids_embs[medoid_idx]
        docemb = dataset_doc_emb_data_dict[docid].numpy()

        cos_distance = cosine_fn(docemb, cluster_medoid_emb)
        clusterids_docids_dist_cosdict[cluster_idx][docid] = float(cos_distance)

    cluster_idx_list = list(clusteridx_samplesize_dict.keys())
    random.shuffle(cluster_idx_list)

    ########################################################################
    #                   Sample documents from each cluster
    ########################################################################
    # 9. Probabilistic sampling based on document distance
    total_sample_size = 0
    cluster_idx_set = set()
    docids_set = set()
    all_sampled_docids_set = set()
    for cluster_idx in tqdm(cluster_idx_list, total=len(clusteridx_samplesize_dict)):
        sample_size = clusteridx_samplesize_dict[cluster_idx]
        total_sample_size += sample_size
        cluster_idx_set.add(cluster_idx)

        # (1) find docids belong to cluster
        curr_docids = clusteridx_docids_dict[cluster_idx]
        for docid in curr_docids:
            docids_set.add(docid)

        # (2) get cosine-similarity for each docid
        curr_docid_distances = [clusterids_docids_dist_cosdict[cluster_idx][docid] for docid in curr_docids]

        # (3) define probabilities based on distance
        prob_values = [e/T for e in curr_docid_distances]
        curr_docid_probs = np.exp(prob_values) / np.sum(np.exp(prob_values), axis=0)
        assert np.sum(curr_docid_probs) >= 0.99

        # (4) random sample with probabilities
        curr_selected_docids_pool = set()
        for seed in SEED_LIST:
            np.random.seed(seed)
            sampled_docids = np.random.choice(curr_docids, p=curr_docid_probs, size=sample_size, replace=False)
            for docid in sampled_docids:
                curr_selected_docids_pool.add(str(docid))

        # (5) apply MMR and pick top-sample_size documents
        closest_to_centroid_docid = centroid_nearest_docids_list[cluster_idx]
        final_sampled_docids_info = passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                                                      n_selection=sample_size, seed_docid=closest_to_centroid_docid, docids=curr_selected_docids_pool,
                                                      lambda_val=lambda_val)
        final_sampled_docids = [e[0] for e in final_sampled_docids_info]

        all_sampled_docids_set = all_sampled_docids_set.union(set(final_sampled_docids))
    print('... completed document sampling')


    ########################################################################
    #                   Find document-text for the sampled documents
    ########################################################################
    # 10. Look-up Pyserini index to fetch the document-text
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    target_docid_doctext_list = []
    for docid in all_sampled_docids_set:
        doctext = get_doc_text( docid, searcher=searcher)
        if not doctext:
            continue
        target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
    print('... completed document text lookup')
    return target_docid_doctext_list


# def dbscan_clustering_sample(dbscan, target_docids_embs, target_docids, n_train, n_corpus,
#                       dataset_doc_emb_data_dict, lexical_index_name, lambda_val=1.0):
#         ########################################################################
#     #                   Apply clustering on collection
#     ########################################################################
#     labels = dbscan.labels_

#     # 5. Find cluster size (number of documents in each cluster)
#     docid_clusteridx_dict = {}
#     clusteridx_docids_dict = defaultdict(list)
#     cluster_size_dict = defaultdict(int)
#     for docid, cluster_idx in zip(target_docids, labels):
#         docid_clusteridx_dict[docid] = cluster_idx
#         clusteridx_docids_dict[cluster_idx].append(docid)
#         cluster_size_dict[cluster_idx] += 1

#     cluster_size_dict_sorted = sorted(cluster_size_dict.items(), key=lambda x: x[1], reverse=False)
#     print('... completed collection clustering')

#     ########################################################################
#     #                   Determine sample size for each cluster
#     ########################################################################
#     # 6. Find initial round of sample size for each cluster
#     clusteridx_samplesize_dict = defaultdict(int)
#     sample_cnt = 0
#     for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted):
#         sample_size = 1 + int(math.floor( (n_train) * (size / n_corpus) ))
#         clusteridx_samplesize_dict[cluster_idx] = sample_size
#         sample_cnt += sample_size
    
#     if sample_cnt < n_train:
#     # 7. Sample more from highly populated clusters
#         n_remaining = n_train - sample_cnt
#         for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted[-n_remaining::]):
#             clusteridx_samplesize_dict[cluster_idx] += 1
#             sample_cnt += 1
#     elif sample_cnt > n_train:
#         over = sample_cnt - n_train
#         for cluster_idx, _ in cluster_size_dict_sorted[:over]:
#             clusteridx_samplesize_dict[cluster_idx] = max(1, clusteridx_samplesize_dict[cluster_idx] - 1)
#             sample_cnt -= 1
#             if sample_cnt == n_train:
#                 break

#     cluster_medoid_docids = {}
#     for cluster_idx, docids in clusteridx_docids_dict.items():
#         medoid_docid = random.choice(docids)
#         cluster_medoid_docids[]
#     # 8. Derive cosine similarity (~distance) for each documents
#     clusterids_docids_dist_cosdict = defaultdict(dict)
#     for docid, cluster_idx in zip(target_docids, labels):
#         medoid_idx = medoid_indices[cluster_idx]
#         cluster_medoid_emb = target_docids_embs[medoid_idx]
#         docemb = dataset_doc_emb_data_dict[docid].numpy()

#         cos_distance = cosine_fn(docemb, cluster_medoid_emb)
#         clusterids_docids_dist_cosdict[cluster_idx][docid] = float(cos_distance)

#     cluster_idx_list = list(clusteridx_samplesize_dict.keys())
#     random.shuffle(cluster_idx_list)

#     ########################################################################
#     #                   Sample documents from each cluster
#     ########################################################################
#     # 9. Probabilistic sampling based on document distance
#     total_sample_size = 0
#     cluster_idx_set = set()
#     docids_set = set()
#     all_sampled_docids_set = set()
#     for cluster_idx in tqdm(cluster_idx_list, total=len(clusteridx_samplesize_dict)):
#         sample_size = clusteridx_samplesize_dict[cluster_idx]
#         total_sample_size += sample_size
#         cluster_idx_set.add(cluster_idx)

#         # (1) find docids belong to cluster
#         curr_docids = clusteridx_docids_dict[cluster_idx]
#         for docid in curr_docids:
#             docids_set.add(docid)

#         # (2) get cosine-similarity for each docid
#         curr_docid_distances = [clusterids_docids_dist_cosdict[cluster_idx][docid] for docid in curr_docids]

#         # (3) define probabilities based on distance
#         prob_values = [e/T for e in curr_docid_distances]
#         curr_docid_probs = np.exp(prob_values) / np.sum(np.exp(prob_values), axis=0)
#         assert np.sum(curr_docid_probs) >= 0.99

#         # (4) random sample with probabilities
#         curr_selected_docids_pool = set()
#         for seed in SEED_LIST:
#             np.random.seed(seed)
#             sampled_docids = np.random.choice(curr_docids, p=curr_docid_probs, size=sample_size, replace=False)
#             for docid in sampled_docids:
#                 curr_selected_docids_pool.add(str(docid))

#         # (5) apply MMR and pick top-sample_size documents
#         closest_to_centroid_docid = centroid_nearest_docids_list[cluster_idx]
#         final_sampled_docids_info = passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
#                                                       n_selection=sample_size, seed_docid=closest_to_centroid_docid, docids=curr_selected_docids_pool,
#                                                       lambda_val=lambda_val)
#         final_sampled_docids = [e[0] for e in final_sampled_docids_info]

#         all_sampled_docids_set = all_sampled_docids_set.union(set(final_sampled_docids))
#     print('... completed document sampling')


#     ########################################################################
#     #                   Find document-text for the sampled documents
#     ########################################################################
#     # 10. Look-up Pyserini index to fetch the document-text
#     searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

#     target_docid_doctext_list = []
#     for docid in all_sampled_docids_set:
#         doctext = get_doc_text( docid, searcher=searcher)
#         if not doctext:
#             continue
#         target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
#     print('... completed document text lookup')
#     return target_docid_doctext_list


def orth_sample(target_doc_embs, target_docids, n_samples, lexical_index_name):
    representations = target_doc_embs.to('cuda')
    norms = representations.norm(dim=1, keepdim=True)
    print(f"... completed latent norm computation: {norms.shape}")
    normalized_vectors = representations / norms  # Shape: (N, dim)
    print(f"... completed latent vector normalization: {normalized_vectors.shape}")

    # Initialize selected indices
    selected_indices = []

    # Initialize candidate indices as a tensor on GPU
    candidate_indices = torch.arange(normalized_vectors.size(0), device='cuda')  # Shape: (N,)

    # Select the first vector (e.g., the one with the largest norm)
    first_index = torch.argmax(norms).item()
    selected_indices.append(first_index)
    candidate_indices = candidate_indices[candidate_indices != first_index]

    # Store the selected normalized vectors
    selected_vectors = normalized_vectors[first_index].unsqueeze(0)  # Shape: (1, D)

    # Iteratively select vectors based on orthogonality
    for _ in tqdm(range(1, n_samples), desc="Selecting documents", total=n_samples-1):
        if candidate_indices.numel() == 0:
            break  # No more candidates to select from

        # Retrieve candidate vectors
        candidate_vectors = normalized_vectors[candidate_indices]  # Shape: (num_candidates, D)

        # Compute inner products between candidate vectors and selected vectors
        # Shape of inner_products: (num_candidates, num_selected_vectors)
        inner_products = torch.mm(candidate_vectors, selected_vectors.t())

        # Compute squared inner products and sum over selected vectors
        squared_inner_products = inner_products ** 2  # Element-wise square
        s_i = squared_inner_products.sum(dim=1)  # Shape: (num_candidates,)

        # Find the candidate with minimal s_i
        min_s_i, min_idx = torch.min(s_i, dim=0)
        next_index = candidate_indices[min_idx].item()

        # Update selected indices and vectors
        selected_indices.append(next_index)
        selected_vectors = torch.cat([selected_vectors, normalized_vectors[next_index].unsqueeze(0)], dim=0)

        # Remove the selected index from candidate indices
        candidate_indices = candidate_indices[candidate_indices != next_index]

    # Retrieve the document IDs of the selected documents
    selected_docids = [target_docids[idx] for idx in selected_indices]

    ########################################################################
    #                   Find document-text for the sampled documents
    ########################################################################
    # 10. Look-up Pyserini index to fetch the document-text
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    target_docid_doctext_list = []
    for docid in selected_docids:
        doctext = get_doc_text( docid, searcher=searcher)
        if not doctext:
            continue
        target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
    print('... completed document text lookup')
    return target_docid_doctext_list


def main_run(dataset_name, collection_text_filepath, collection_embedding_filepath , save_sampled_documents_filepath, n_clusters, n_train, lambda_value, remove_outlier=None, epsilon=1.0, min_cluster_size=5, min_samples=5, n_estimators=500, contamination=0.1, duqgen_filter=False, normalize=False, sampling_method='clustering'):
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
        if duqgen_filter:
            if dataset_name == 'signal1m' and len(doctext) < 3:
                continue
            elif dataset_name == 'hotpotqa' and len(doctext) < 150:
                continue
            elif dataset_name == 'quora' and len(doctext) < 15:
                continue
            elif dataset_name == 'climate-fever' and len(doctext) < 3:
                continue
            elif dataset_name not in special_datasets_list and len(doctext) < N_DOCTEXT_FILTER:
                continue
        filtered_docid_doctext_dict[docid] = doctext

    # 2. Load document embedding
    dataset_doc_emb_data_dict = {k: v for k, v in torch.load(collection_embedding_filepath, weights_only=True).items() if filtered_docid_doctext_dict.get(k) }

    target_docids = list(dataset_doc_emb_data_dict.keys())
    target_docids_embs = torch.stack( [dataset_doc_emb_data_dict[docid] for docid in target_docids] )
    print(f"... loaded {len(target_docids)} documents")

    # Normalize the embeddings
    if normalize:
        normalized_target_docids_embs = F.normalize(target_docids_embs, p=2, dim=1)
    else:
        normalized_target_docids_embs = target_docids_embs
    print('... completed collection loading')

    # outlier_label = dbscan_outlier(target_docids_embs, eps=1, min_samples=3)

    if remove_outlier == 'dbscan':
        # Get Outliers from normalized embeddings
        if os.path.exists(f'{save_sampled_documents_filepath[:-6]}_intermediate.npy'):
            nonoutlier_idx = np.load(f'{save_sampled_documents_filepath[:-6]}_intermediate.npy')
            print(f'... loaded intermediate nonoutlier index')
        else:
            start_time = time.time()
            dbscan = dbscan_outlier(normalized_target_docids_embs, eps=epsilon, min_samples=min_samples)
            end_time = time.time()
            print(f'... completed DBSCAN outlier detection in {end_time - start_time} seconds')
            outlier_label = dbscan.labels_
            # get index of outliers(where label is -1)
            # outlier_idx = np.where(outlier_label == -1)[0]
            nonoutlier_idx = np.where(outlier_label != -1)[0]

        nonoutlier_target_docids_embs = target_docids_embs[nonoutlier_idx]
    elif remove_outlier == 'hdbscan':
        # Get Outliers from normalized embeddings
        outlier_label = hdbscan_outlier(normalized_target_docids_embs, min_cluster_size=min_cluster_size, min_samples=min_samples)
        # get index of outliers(where label is -1)
        # outlier_idx = np.where(outlier_label == -1)[0]
        nonoutlier_idx = np.where(outlier_label != -1)[0]

        nonoutlier_target_docids_embs = target_docids_embs[nonoutlier_idx]
    elif remove_outlier == 'iforest':
        # Get Outliers from normalized embeddings
        outlier_label = iforest_outlier_detection(normalized_target_docids_embs, n_estimators=n_estimators, contamination=contamination)
        # get index of outliers(where label is -1)
        # outlier_idx = np.where(outlier_label == -1)[0]
        nonoutlier_idx = np.where(outlier_label != -1)[0]

        nonoutlier_target_docids_embs = target_docids_embs[nonoutlier_idx]
    else:
        nonoutlier_target_docids_embs = target_docids_embs
    n_corpus = len(nonoutlier_target_docids_embs)
    print(f"... completed outlier detection: {n_corpus} documents remaining")

    if sampling_method == 'clustering':
        print('... starting clustering sampling')
        target_docid_doctext_list = clustering_sample(nonoutlier_target_docids_embs, target_docids, n_clusters=n_clusters, n_train=n_train, 
                                                    n_corpus=n_corpus, dataset_doc_emb_data_dict=dataset_doc_emb_data_dict,
                                                    lexical_index_name=lexical_index_name, lambda_val=lambda_value)
    elif sampling_method == 'kmedoid':
        print('... starting kmedoid sampling')
        target_docid_doctext_list = kmedoid_clustering_sample(nonoutlier_target_docids_embs, target_docids, n_clusters=n_clusters, n_train=n_train, 
                                                    n_corpus=n_corpus, dataset_doc_emb_data_dict=dataset_doc_emb_data_dict,
                                                    lexical_index_name=lexical_index_name, lambda_val=lambda_value)
    elif sampling_method == 'orth':
        print('... starting orthogonal sampling')
        target_docid_doctext_list = orth_sample(nonoutlier_target_docids_embs, target_docids, n_samples=n_train, lexical_index_name=lexical_index_name)
    elif sampling_method == 'clara':
        print('... starting clara sampling')
        target_docid_doctext_list = clara_clustering_sample(nonoutlier_target_docids_embs, target_docids, n_clusters=n_clusters, n_train=n_train, 
                                                    n_corpus=n_corpus, dataset_doc_emb_data_dict=dataset_doc_emb_data_dict,
                                                    lexical_index_name=lexical_index_name, lambda_val=lambda_value)
    # elif sampling_method == 'dbscan' and remove_outlier == 'dbscan':
    #     target_docid_doctext_list = dbscan_clustering_sample(dbscan, nonoutlier_target_docids_embs, target_docids, n_train=n_train, 
    #                                             n_corpus=n_corpus, dataset_doc_emb_data_dict=dataset_doc_emb_data_dict,
    #                                             lexical_index_name=lexical_index_name, lambda_val=lambda_value)
    ########################################################################
    #                   Save sampled documents with text
    ########################################################################
    # 11. Save the documents sampled from collection
    with open(save_sampled_documents_filepath, 'w') as f:
        for _, line in enumerate(target_docid_doctext_list):
            json.dump(line, f)
            f.write('\n')
    print('... completed saving sampled documents')


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Document Sampling')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name to be specific')
    parser.add_argument('--collection_text_filepath', required=True, type=str,
                        help='file path to document collection text')
    parser.add_argument('--collection_embedding_filepath', required=True, type=str,
                        help='file path to document collection embedding')
    parser.add_argument('--save_sampled_documents_filepath', required=True, type=str,
                        help='file path to save output of the script: sampled documents')
    parser.add_argument('--n_clusters', type=int, default=1000, required=False,
                        help='number of clusters')
    parser.add_argument('--n_train', type=int, default=1000, required=False,
                        help='number of training examples')
    parser.add_argument('--lambda_val', type=float, default=1.0, required=False,
                        help='lambda value used in MMR diversify measure')
    parser.add_argument('--remove_outlier', type=str, required=False,
                    help='removing outlier with method: dbscan or iforest')
    parser.add_argument('--epsilon', type=float, default=1.0, required=False,
                        help='epsilon parameter for DBSCAN')
    parser.add_argument('--min_cluster_size', type=int, default=5, required=False,
                        help='min_cluster_size parameter for HDBSCAN')
    parser.add_argument('--min_samples', type=int, default=5, required=False,
                        help='min_samples parameter for DBSCAN')
    parser.add_argument('--n_estimators', type=int, default=500, required=False,
                        help='n_estimators parameter for Isolation Forest')
    parser.add_argument('--contamination', type=float, default=0.05, required=False,
                        help='contamination parameter for Isolation Forest')
    parser.add_argument('--duqgen_filter', type=str_to_bool, required=False,
                        help='filter out noisy documents based on dataset')
    parser.add_argument('--normalize', type=str_to_bool, required=False,
                        help='normalize the embeddings')
    parser.add_argument('--sampling_method', type=str, default='clustering', required=False,
                        help='sampling method: clustering or kmedoid')
    args = parser.parse_args()


    main_run(args.dataset_name, args.collection_text_filepath, args.collection_embedding_filepath , args.save_sampled_documents_filepath,
             args.n_clusters, args.n_train, args.lambda_val, args.remove_outlier, epsilon=args.epsilon, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, n_estimators=args.n_estimators, contamination=args.contamination, duqgen_filter=args.duqgen_filter, normalize=args.normalize, sampling_method=args.sampling_method)
