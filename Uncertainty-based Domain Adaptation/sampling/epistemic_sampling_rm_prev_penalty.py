# from sklearn.cluster import DBSCAN
import numpy as np
import json
import torch
import faiss
from tqdm import tqdm
import argparse
from numpy import dot
import math
import random
from numpy.linalg import norm
from collections import OrderedDict, defaultdict
from pyserini.search.lucene import LuceneSearcher
from load import load_embeddings
from pathlib import Path

SEED_LIST = [35, 745, 10, 6534, 2]
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
        seeditem_item_rel_dict[docid] = float( cosine_fn(dataset_doc_emb_data_dict[docid].cpu().numpy(), dataset_doc_emb_data_dict[seed_docid].cpu().numpy()) )
    return seeditem_item_rel_dict


def get_item_item_simscore_dict(docids, dataset_doc_emb_data_dict):
    item_item_cos_dict = defaultdict(dict)
    for i in docids:
        for j in docids:
            cos_sim = float( cosine_fn(dataset_doc_emb_data_dict[i].cpu().numpy(), dataset_doc_emb_data_dict[j].cpu().numpy()) )
            item_item_cos_dict[i][j] = cos_sim
    return item_item_cos_dict

def safe_softmax(this_value, total_values):
    max_value = max(total_values)
    total_values = [value - max_value for value in total_values]
    this_value -= max_value
    return np.exp(this_value) / np.sum(np.exp(total_values))

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
                      n_selection, seed_docid, docids, lambda_val=0.5, mlm_scores=None, dropout_scores=None, alpha=0.5, beta=0.5):
    seeddoc_doc_rel_dict = get_seeditem_item_simscore_dict(seed_docid, docids, dataset_doc_emb_data_dict)

    doc_doc_rel_dict = get_item_item_simscore_dict(docids, dataset_doc_emb_data_dict)

    _, output_scores_dict = mmr_sorted(set(docids), lambda_val, seeddoc_doc_rel_dict, doc_doc_rel_dict)
    if mlm_scores:
        mlm_scores_dict = {}
        docids_list = list(docids)
        mlm_scores_list = [mlm_scores[docid] for docid in docids_list]
        mean_mlm_score = np.mean(mlm_scores_list)
        std_mlm_score = np.std(mlm_scores_list)
        for docid, doc_mlm_score in zip(docids_list, mlm_scores_list):
            # (3) get the z-score for each score and take the sum
            zscore_mlm = (doc_mlm_score - mean_mlm_score) / (std_mlm_score + 1e-6)
            mlm_scores_dict[docid] = zscore_mlm
        mmr_scores = list(output_scores_dict.values())
        mean_mmr_score = np.mean(mmr_scores)
        std_mmr_score = np.std(mmr_scores) # fixed 2025-0811-1348

        if dropout_scores:
            dropout_scores_dict = {}
            dropout_scores_list = [dropout_scores[docid] for docid in docids_list]
            mean_dropout_score = np.mean(dropout_scores_list)
            std_dropout_score = np.std(dropout_scores_list)
            for docid, doc_dropout_score in zip(docids_list, dropout_scores_list):
                # (3) get the z-score for each score and take the sum
                zscore_dropout = (doc_dropout_score - mean_dropout_score) / (std_dropout_score + 1e-6)
                dropout_scores_dict[docid] = zscore_dropout
            
            new_output_scores_dict = {}
            for docid, mmr_score in output_scores_dict.items():
                # (1) get the z-score for each score and take the sum
                zscore_mmr = (mmr_score - mean_mmr_score) / (std_mmr_score + 1e-6)
                new_output_scores_dict[docid] = (alpha * zscore_mmr) + (beta * mlm_scores_dict[docid]) + ((1 - alpha - beta) * dropout_scores_dict[docid])
            output_scores_dict = new_output_scores_dict
        else:
            new_output_scores_dict = {}
            for docid, mmr_score in output_scores_dict.items():
                # (1) get the z-score for each score and take the sum
                zscore_mmr = (mmr_score - mean_mmr_score) / (std_mmr_score + 1e-6)
                new_output_scores_dict[docid] = (alpha * zscore_mmr) + ((1 - alpha) * mlm_scores_dict[docid])
            output_scores_dict = new_output_scores_dict

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

def clustering_sample_original(target_docids_embs, target_docids, n_clusters, n_train, n_corpus,
                               dataset_doc_emb_data_dict, lexical_index_name, prev_ids_embs, lambda_val=1.0, mlm_scores=None, corpus=None, dropout_scores=None, alpha=0.5, beta=0.5):
        ########################################################################
    #                       Apply clustering on collection
    ########################################################################
    # 10. Look-up Pyserini index to fetch the document-text
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    # 3. Train and index clustering algorithm
    d = target_docids_embs.shape[1]
    target_docids_embs_np = target_docids_embs.cpu().numpy()
    
    kmeans = faiss.Kmeans(d, n_clusters, niter=N_ITER, verbose=VERBOSE, gpu=True, seed=420)
    kmeans.train(target_docids_embs_np)

    index = faiss.IndexFlatL2 (d)
    index.add ( target_docids_embs_np )
    D, I = index.search (kmeans.centroids, 1)

    # 4. Find docids closes to each cluster centroid
    centroid_nearest_docids_list = []
    for centroid_nearest_idxs, _ in zip(I, D):
        centroid_nearest_docid = target_docids[centroid_nearest_idxs[0]]
        centroid_nearest_docids_list.append( centroid_nearest_docid )

    D_inv, I_inv = kmeans.index.search(target_docids_embs_np, 1)

    # 5. Find cluster size (number of documents in each cluster)
    docid_clusteridx_dict = {}
    clusteridx_docids_dict = defaultdict(list)
    cluster_size_dict = defaultdict(int)
    for docid, cluster_idx in zip(target_docids, I_inv):
        docid_clusteridx_dict[docid] = cluster_idx[0]
        clusteridx_docids_dict[cluster_idx[0]].append( docid )
        cluster_size_dict[ cluster_idx[0] ] += 1

    print('... completed collection clustering')

    ########################################################################
    #               Determine sample size for each cluster
    ########################################################################
    
    print('... calculating penalty for previously sampled clusters')
    gamma = 1.0

    cluster_penalty_count = defaultdict(int)
    if prev_ids_embs is not None and len(prev_ids_embs) > 0:
        prev_ids_embs_np = prev_ids_embs.cpu().numpy()
        D_prev, I_prev = kmeans.index.search(prev_ids_embs_np, 1)
        for cluster_idx in I_prev:
            cluster_penalty_count[cluster_idx[0]] += 1
    
    cluster_weights = {}
    total_weight = 0.0
    for cluster_idx, size in cluster_size_dict.items():
        penalty = cluster_penalty_count[cluster_idx]
        weight = size / (1 + gamma * penalty)
        cluster_weights[cluster_idx] = weight
        total_weight += weight
    
    print('... allocating budget based on new weights')
    clusteridx_samplesize_dict = defaultdict(int)

    cluster_expected_samples = {}
    for cluster_idx, weight in cluster_weights.items():
        if total_weight > 0:
            expected_samples = n_train * (weight / total_weight)
            cluster_expected_samples[cluster_idx] = expected_samples

    sample_cnt = 0
    for cluster_idx, expected in cluster_expected_samples.items():
        num_samples = math.floor(expected)
        clusteridx_samplesize_dict[cluster_idx] = num_samples
        sample_cnt += num_samples

    n_remaining = n_train - sample_cnt
    if n_remaining > 0:
        sorted_by_remainder = sorted(
            cluster_expected_samples.keys(),
            key=lambda cid: cluster_expected_samples[cid] - math.floor(cluster_expected_samples[cid]),
            reverse=True
        )
        
        for i in range(n_remaining):
            cluster_to_add = sorted_by_remainder[i]
            clusteridx_samplesize_dict[cluster_to_add] += 1
    
    # 8. Derive cosine similarity (~distance) for each documents
    clusterids_docids_dist_cosdict = defaultdict(dict)
    for docid, cluster_idx in zip(target_docids, I_inv):
        cluster_centroid = kmeans.centroids[cluster_idx[0]]
        docemb = dataset_doc_emb_data_dict[docid].cpu().numpy()

        cos_distance = cosine_fn(docemb, cluster_centroid)
        clusterids_docids_dist_cosdict[cluster_idx[0]][docid] = float(cos_distance)

    cluster_idx_list = list(clusteridx_samplesize_dict.keys())

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
        if sample_size == 0:
            continue

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
            # [수정] sample_size가 curr_docids 개수보다 많을 경우를 대비하여 replace=False 유지하되 min으로 크기 조정
            num_to_sample = min(sample_size, len(curr_docids))
            if num_to_sample > 0:
                sampled_docids = np.random.choice(curr_docids, p=curr_docid_probs, size=num_to_sample, replace=False)
                for docid in sampled_docids:
                    curr_selected_docids_pool.add(str(docid))

        # (5) apply MMR and pick top-sample_size documents
        if len(curr_selected_docids_pool) > 0:
            closest_to_centroid_docid = centroid_nearest_docids_list[cluster_idx]
            
            final_n_selection = min(sample_size, len(curr_selected_docids_pool))
            
            final_sampled_docids_info = passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                                                          n_selection=final_n_selection, seed_docid=closest_to_centroid_docid, docids=curr_selected_docids_pool,
                                                          lambda_val=lambda_val, mlm_scores=mlm_scores, dropout_scores=dropout_scores, alpha=alpha, beta=beta)
            final_sampled_docids = [e[0] for e in final_sampled_docids_info]
            
            all_sampled_docids_set = all_sampled_docids_set.union(set(final_sampled_docids))
    print('... completed document sampling')


    ########################################################################
    #               Find document-text for the sampled documents
    ########################################################################

    target_docid_doctext_list = []
    for docid in all_sampled_docids_set:
        doctext = get_doc_text( docid, searcher=searcher)
        if not doctext:
            continue
        target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
    print('... completed document text lookup')
    return target_docid_doctext_list

def _robust_zscores(x: np.ndarray) -> np.ndarray:
    """
    Modified Z if MAD>0, else fallback to z using IQR/1.349 (normal equivalence).
    If data are constant (IQR==0), returns zeros.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 0:
        return 0.6745 * (x - med) / mad
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    if iqr > 0:
        return (x - med) / (1.349 * iqr)
    return np.zeros_like(x)

def elbow_inlier_ids(scores_by_id: dict, z_threshold: int = 1.5):
    if not scores_by_id:
        return []
    ids, vals = zip(*scores_by_id.items())
    ids = np.array(ids, dtype=object)
    vals = np.asarray(vals, dtype=float)

    # Robust Z-scores
    z = _robust_zscores(vals)

    # Inliers are those with Z <= threshold
    inlier_mask = z <= z_threshold
    inlier_ids = ids[inlier_mask].tolist()
    return inlier_ids

def main_run(dataset_name, cleaned_document_path, document_embedding_path, entropy_dict_path, mlm_scores_path, save_sampled_documents_filepath, collection_path, model_name, n_clusters=1000, n_train=5000, remove_outlier="entropy", contamination=1.5, sampling_method='clustering', lambda_value=1.0, duqgen_filter=False, dropout_scores_path=None, alpha=0.5, beta=0.5):
    lexical_index_name = f'beir-v1.0.0-{dataset_name}.multifield' # pyserini multi-field index name
    ########################################################################
    #                       Load collection embedding
    ########################################################################

    # preload the previous sampled paths
    prefix = save_sampled_documents_filepath.replace(".jsonl", "").split("online")[0]
    prev_paths = Path("datasets").glob(f"{prefix}online*.jsonl")
    prev_ids = []
    for path in prev_paths:
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                prev_ids.append(doc["docid"])

    # Load cleaned sentences and documents.
    corpus = {}
    with open(cleaned_document_path, "r") as f:
        for line in tqdm(f, desc="Loading documents"):
            doc = json.loads(line)
            doctext = doc["cleaned"]
            corpus[doc["did"]] = doctext

    # Load mlm_idf_scores of documents.
    if mlm_scores_path is not None:
        if mlm_scores_path.endswith('.pt'):
            mlm_scores = torch.load(mlm_scores_path, weights_only=False)
        elif mlm_scores_path.endswith('.json'):
            with open(mlm_scores_path, 'r') as f:
                mlm_scores = json.load(f)
        elif Path(mlm_scores_path).is_dir():
            mlm_scores = {}
            for file in Path(mlm_scores_path).glob("logit*.pt"):
                print(f"Loading {file}")
                mlm_scores.update(torch.load(file, weights_only=False))
    else:
        mlm_scores = None
    print("len mlm_scores")
    print(len(mlm_scores) if mlm_scores is not None else 0)

    if dropout_scores_path is not None:
        with open(dropout_scores_path, 'r') as f:
            dropout_scores = json.load(f)
    else:
        dropout_scores = None

    # 2. Load document embedding
    embeddings = load_embeddings(document_embedding_path, collection_path, model_name)
    print("len embeddings")
    print(len(embeddings))

    prev_ids_embs = None
    if prev_ids:
        # prev_ids에 해당하는 임베딩이 있는지 확인 후 stack
        valid_prev_embs = [embeddings[pid] for pid in prev_ids if pid in embeddings]
        if valid_prev_embs:
            prev_ids_embs = torch.stack(valid_prev_embs)
            
    dataset_doc_emb_data_dict = {}
    for did in corpus:
        if did in embeddings and did not in prev_ids:
            dataset_doc_emb_data_dict[did] = embeddings[did]
    n_corpus = len(dataset_doc_emb_data_dict)

    with open(entropy_dict_path, 'r') as f:
        entropy_dict = json.load(f)
    target_docids = list(dataset_doc_emb_data_dict.keys())
    print(f"... loaded {len(target_docids)} documents")

    if remove_outlier == 'entropy':
        print(f"... removing outliers based on entropy")
        target_entropies = {did: entropy_dict.get(did, 0) for did in target_docids}
        inliers = elbow_inlier_ids(target_entropies, z_threshold=contamination)

        nonoutlier_target_docids_embs = torch.stack([dataset_doc_emb_data_dict[e] for e in inliers])
        target_docids = inliers
    elif remove_outlier == "none" or remove_outlier is None:
        print(f"... skipping outlier removal")
        nonoutlier_target_docids_embs = torch.stack([dataset_doc_emb_data_dict[e] for e in target_docids])
    else:
        raise ValueError(f"Unknown outlier removal method: {remove_outlier}")
    n_corpus = len(nonoutlier_target_docids_embs)
    print(f"... completed outlier detection: {n_corpus} documents remaining")

    if sampling_method == 'clustering':
        print('... starting clustering sampling')
        target_docid_doctext_list = clustering_sample_original(nonoutlier_target_docids_embs, target_docids, n_clusters=n_clusters, n_train=n_train,
                                                             n_corpus=n_corpus, dataset_doc_emb_data_dict=dataset_doc_emb_data_dict,
                                                             lexical_index_name=lexical_index_name, prev_ids_embs=prev_ids_embs, lambda_val=lambda_value, mlm_scores=mlm_scores, dropout_scores=dropout_scores, alpha=alpha, beta=beta)

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
    parser.add_argument('--cleaned_document_path', required=True, type=str,
                        help='file path to document collection text')
    parser.add_argument('--document_embedding_path', required=True, type=str,
                        help='file path to document collection embedding')
    parser.add_argument('--entropy_dict_path', required=True, type=str,
                        help='file path to document collection score vector')
    parser.add_argument('--mlm_scores_path', required=False, type=str,
                        help='file path to document collection embedding')
    parser.add_argument('--dropout_scores_path', required=False, type=str,
                        help='file path to document collection dropout scores')
    parser.add_argument('--collection_path', default="", type=str,
                        help='file path to collection')
    parser.add_argument('--model_name', required=True, type=str,
                        help='model name used to generate document embedding')
    parser.add_argument('--save_sampled_documents_filepath', required=True, type=str,
                        help='file path to save output of the script: sampled documents')
    parser.add_argument('--lambda_val', type=float, default=1.0, required=False,
                        help='lambda value used in MMR diversify measure')
    parser.add_argument('--n_clusters', type=int, default=1000, required=False,
                        help='number of clusters')
    parser.add_argument('--n_train', type=int, default=5000, required=False,
                        help='number of training examples')
    parser.add_argument('--remove_outlier', type=str, required=False,
                        help='removing outlier with method: dbscan or iforest')
    parser.add_argument('--contamination', type=float, default=1.5, required=False,
                        help='contamination parameter for Isolation Forest')
    parser.add_argument('--sampling_method', type=str, default='clustering', required=False,
                        help='sampling method: clustering or kmedoid')
    parser.add_argument('--duqgen_filter', type=str_to_bool, default=False, required=False,
                        help='filter documents based on duqgen dataset')
    parser.add_argument('--alpha', type=float, default=0.5, required=False,
                        help='weight for z-MMR score when combining with aleatoric uncertainty scores')
    parser.add_argument('--beta', type=float, default=0.5, required=False,
                        help='weight for MLM score when combining with aleatoric uncertainty scores')
    args = parser.parse_args()

    main_run(args.dataset_name, args.cleaned_document_path, args.document_embedding_path, args.entropy_dict_path, args.mlm_scores_path, args.save_sampled_documents_filepath, args.collection_path, args.model_name,
             n_clusters=args.n_clusters, n_train=args.n_train, remove_outlier=args.remove_outlier, contamination=args.contamination, sampling_method=args.sampling_method,
             lambda_value=args.lambda_val, duqgen_filter=args.duqgen_filter, dropout_scores_path=args.dropout_scores_path, alpha=args.alpha, beta=args.beta)