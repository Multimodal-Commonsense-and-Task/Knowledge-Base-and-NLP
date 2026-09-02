import torch
from pathlib import Path as CPath
from tqdm.auto import tqdm
import pickle
import numpy as np
import glob
from itertools import chain

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup

def load_embeddings(embedding_path, collection_path=None, model_name="colbert", splits_no=None, splits_len=None) -> dict[torch.Tensor]:
    if model_name == "colbert":
        from colbert.indexing.loaders import get_parts
        from colbert.indexing.index_manager import load_index_part
        _, parts_paths, _ = get_parts(embedding_path)

        embeddings = [load_index_part(filename) for filename in parts_paths if filename is not None]
        embeddings = torch.cat(embeddings)
        print(f"Loaded embeddings, shape:{embeddings.shape}")
        doc_ids = [line.strip().split('\t')[0] for line in open(collection_path)]
        assert embeddings.shape[0] == len(doc_ids), "Number of embeddings and doc_ids do not match"
        if splits_no is not None and splits_len is not None:
            one_split = len(doc_ids) // splits_len
            indexes = [i for i in range(splits_no * one_split, (splits_no + 1) * one_split)]
        else:
            indexes = range(len(doc_ids))

        embedding_dict = {}
        for idx in tqdm(indexes, desc="adding index"):
            embedding_dict[doc_ids[idx]] = embeddings[idx]
        print(f"Loaded {len(embedding_dict)} embeddings")
        return embedding_dict
    
    elif model_name in ["cocondenser", "cocodr", "qwen3"]:
        if not CPath(embedding_path).is_dir():
            raise ValueError("For cocondenser, the embedding path must be a directory containing multiple .pt files")
        if splits_no is not None and splits_len is not None:
            file_patterns = str(embedding_path) + f"/corpus*{splits_no}.pkl"
        else:
            file_patterns = str(embedding_path) + "/corpus*.pkl"
        index_files = glob.glob(file_patterns)
        shards = chain(map(pickle_load, index_files))
        if len(index_files) > 1:
            shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
        embeddings_dict = {}
        for p_reps, p_lookup in shards:
            for emb, docid in zip(p_reps, p_lookup):
                embeddings_dict[docid] = torch.tensor(emb)
        return embeddings_dict

    elif model_name in ["monot5", "dpr"]:
        if not str(embedding_path).endswith(".pt"):
            if splits_no is None:
                pathlists = list(CPath(embedding_path).glob("*.pt"))
                if len(pathlists) == 8:
                    embeddings = {}
                    for file in pathlists:
                        print(f"Loading embeddings from {file}...")
                        part_embeddings = torch.load(file, weights_only=False, map_location='cpu')
                        embeddings.update(part_embeddings)
                    doc_ids = list(embeddings.keys())
                else:
                    raise ValueError("If the embedding path is not a .pt file, you must provide a split number")
            else:
                embedding_path = f"{embedding_path}/{splits_no}.pt"
                embeddings = torch.load(embedding_path, weights_only=False)
                doc_ids = list(embeddings.keys())
        else:
            print(f"Loading embeddings from {embedding_path}...")
            embeddings = torch.load(embedding_path, weights_only=False)
            doc_ids = list(embeddings.keys())
            if splits_no is not None and splits_len is not None:
                one_split = len(doc_ids) // splits_len
                doc_ids = doc_ids[splits_no * one_split : (splits_no + 1) * one_split]
        embedding_dict = {}
        for k in tqdm(doc_ids, desc="adding index to dict"):
            embedding_dict[k] = embeddings[k]
        return embedding_dict