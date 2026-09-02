import argparse
import copy
import json
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from load import load_embeddings
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoModelWithLMHead, AutoTokenizer, BertForMaskedLM, DistilBertForMaskedLM

def print_tokens(ids, tokenizer):
    for token_id in ids:
        print(f"id[{token_id}: {tokenizer.convert_ids_to_tokens(token_id)}")

def get_top_k_from_representations(reps, mlm_model, k) -> tuple[list[float], list[int]]:
    logits = mlm_model(reps)
    topk = logits.topk(k=k)
    return topk.values.cpu().tolist(), topk.indices.cpu().tolist()

def get_top_k_from_representations_batch(reps: torch.Tensor, mlm_model: torch.nn.Module, k: int):
    """
    reps: [B, D]  on device
    returns topk scores and indices, both on CPU as python lists:
      values: List[List[float]]  shape [B, k]
      indices: List[List[int]]   shape [B, k]
    """
    logits = mlm_model(reps)                # [B, Vocab]
    topk = logits.topk(k=k, dim=-1)         # both [B, k]
    return topk.values.cpu().tolist(), topk.indices.cpu().tolist()

if __name__ == '__main__':
    """
    Usage:
      python document_score.py \
          --dataset scifact \
          --output /path/to/output.json

    This script computes token amnesia score by applying mlm head
    The results are aggregated into a single JSON file.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--idf_dict_path", type=Path, required=True)
    argparser.add_argument("--document_embedding_path", type=Path, required=True)
    argparser.add_argument("--collection_path", type=Path, default=None)
    argparser.add_argument("--mlm_model", type=str, default="bert-base-uncased")
    argparser.add_argument("--embedding_model", type=str, default="colbert")
    argparser.add_argument("--cleaned_document_path", type=Path, required=True)
    argparser.add_argument("--score_dir", type=str, required=True)
    argparser.add_argument("--output_file", type=str, required=True)
    argparser.add_argument("--device", type=str, default="cuda")
    argparser.add_argument("--batch_size", type=int, default=512)
    argparser.add_argument("--split", type=int, required=False, default=None,
                           help="split number if exists each shard would be generated first")
    argparser.add_argument("--total_splits", type=int, default=None,
                           help="split number if exists each shard would be generated first")
    args = argparser.parse_args()

    dataset = args.dataset
    idf_dict_path = args.idf_dict_path
    clean_document_path = args.cleaned_document_path
    document_embedding_path = args.document_embedding_path
    collection_path = args.collection_path
    model_name = args.embedding_model
    device = args.device

    output_path = Path(args.output_file)
    score_path = Path(args.score_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.mlm_model)
    if args.mlm_model.startswith("Qwen"):
        model = AutoModelForCausalLM.from_pretrained(args.mlm_model).eval().to(device)
        mlm_model = model.lm_head
    else:
        model = AutoModelWithLMHead.from_pretrained(args.mlm_model).eval().to(device)
        if isinstance(model, BertForMaskedLM):
            mlm_model = model.cls 
        elif hasattr(model, 'lm_head'):
            mlm_model = model.lm_head
        elif isinstance(model, DistilBertForMaskedLM):
            mlm_model = torch.nn.Sequential(            # DistilBERT
                model.vocab_transform,
                model.vocab_layer_norm,
                model.vocab_projector,
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    # Save arguments to output directory as args.json
    with (score_path / "amnesia_args.json").open("w") as f:
        save_obj = copy.deepcopy(vars(args))
        for k, v in save_obj.items():
            if not isinstance(v, str):
                save_obj[k] = str(v)
        json.dump(save_obj, f)

    # load idf dictionary
    with open(idf_dict_path, 'r') as f:
        idf_dict = json.load(f)
    
    if args.split is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{args.split}.pt"

    # ============= Check logit values
    print("loading...")
    print(f"Loading document embeddings from {document_embedding_path} of {model_name}...")
    if args.split is not None and args.total_splits is not None:
        embeddings = load_embeddings(document_embedding_path, collection_path, model_name, args.split, args.total_splits)
    else:
        embeddings = load_embeddings(document_embedding_path, collection_path, model_name)
    print("complete!")
    doc_ids = list(embeddings.keys())

    batch_size = args.batch_size
    batched_group_ids = [doc_ids[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]
    # ============= Calculate IDF Scores of each sentences ============= #
    mlm_scores = defaultdict(dict)
    logit_dicts = defaultdict(tuple)
    for batched_ids in tqdm(batched_group_ids, desc="Processing batches"):
        batched_embeddings = torch.stack([embeddings[did] for did in batched_ids]).to(device)

        # Get the top k tokens from the representation
        batch_scores, batch_ids = get_top_k_from_representations_batch(batched_embeddings, mlm_model, 1000)

        for did, ids, scores in zip(batched_ids, batch_ids, batch_scores):
            logit_dicts[did] = (ids, scores)
            token_scores = []
            for id, score in zip(ids, scores):
                idf_value = idf_dict.get(str(id), 1)
                token_score = np.log(idf_value) - score
                token_scores.append({
                    "token_id": id,
                    "token_score": token_score,
                    "logit_score": score,
                    "idf_score": idf_value
                })
            token_scores.sort(key=lambda x: x["logit_score"], reverse=True)

            save_obj = {}
            for k in [1000]:
                top_k_tokens = token_scores[:k]
                mlm_score = sum(token["token_score"] for token in top_k_tokens)
                save_obj[f"logit_{k}"] = mlm_score
            mlm_scores[did] = save_obj
    for k in [1000]:
        save_dict = {}
        for doc_id, score in mlm_scores.items():
            save_dict[doc_id] = score[f"logit_{k}"]
        score_save_path = str(score_path / f"logit_{k}.pt") if not args.split else str(score_path / f"logit_{k}_split_{args.split}.pt")
        torch.save(save_dict, score_save_path)
    torch.save(logit_dicts, str(output_path))
