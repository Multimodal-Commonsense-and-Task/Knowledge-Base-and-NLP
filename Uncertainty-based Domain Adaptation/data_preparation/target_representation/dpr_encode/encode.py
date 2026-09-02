import json
import os
import torch
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
torch.manual_seed(123)


def encode(base_modename_or_path, clean_document_path, output_path, cache_dir):
    # 1. Load model & tokenizer
    passage_encoder = SentenceTransformer(base_modename_or_path, cache_folder=cache_dir)

    # 2. Load corpus data
    corpus = {}
    with open(clean_document_path, "r") as f:
        for line in tqdm(f, desc="Loading corpus"):
            doc = json.loads(line)
            corpus[doc["did"]] = doc["cleaned"]


    # 3. embed
    # Aggregated dictionaries to hold embeddings.
    batch_size = 128
    dids = list(corpus.keys())
    batched_did_groups = [dids[i:i + batch_size] for i in range(0, len(dids), batch_size)]
    
    # Aggregated dictionaries to hold embeddings.
    aggregated_doc_embeddings = {}
    
    # Process documents in batches.
    with torch.no_grad():
        for batched_did_group in tqdm(batched_did_groups, desc="Embedding corpus"):
            batched_docs = [corpus[did] for did in batched_did_group]
            # Encode the documents in the batch.
            batched_doc_embeddings = passage_encoder.encode(batched_docs, convert_to_tensor=True, show_progress_bar=False)

            # Aggregate embeddings in dictionaries.
            for did, doc_emb in zip(batched_did_group, batched_doc_embeddings):
                aggregated_doc_embeddings[did] = doc_emb.cpu()
    
    # Save the aggregated embeddings dictionaries as single files.
    if len(dids) > 2000000:
        splits_len = 8
        output_path = output_path.replace(".pt", "")
        os.makedirs(output_path, exist_ok=True)
        one_split = len(dids) // splits_len
        for splits_no in range(splits_len):
            splits_dids = dids[splits_no * one_split : (splits_no + 1) * one_split]
            split_embeddings = {k: aggregated_doc_embeddings[k] for k in splits_dids}
            torch.save(split_embeddings, f"{output_path}/{splits_no}.pt")
    else:
        torch.save(aggregated_doc_embeddings, output_path)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Train MonoT5-3B')
    parser.add_argument('--base_modename_or_path', required=True, type=str,
                        help='base model name or path to be fine-tuned upon')
    parser.add_argument("--clean_document_path", type=str, required=True,
                        help="Path to cleaned document corpus file")
    parser.add_argument("--document_embedding_file", type=str, required=True,
                        help="Path to save aggregated document embeddings (single .pt file)")
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='cache dir to download model and tokenizer')
    args = parser.parse_args()


    encode(args.base_modename_or_path, args.clean_document_path, args.document_embedding_file, args.cache_dir)