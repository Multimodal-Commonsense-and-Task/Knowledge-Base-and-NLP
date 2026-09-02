import json
import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration, T5EncoderModel)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(123)


def encode(base_modename_or_path, clean_document_path, output_path, cache_dir):
    # 1. Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_modename_or_path, use_fast=False, cache_dir=cache_dir)
    # model = T5ForConditionalGeneration.from_pretrained(base_modename_or_path, cache_dir=cache_dir, device_map='auto').eval()
    # encoder = model.encoder.eval()
    encoder = T5EncoderModel.from_pretrained(base_modename_or_path, cache_dir=cache_dir, device_map='auto').eval()


    # 2. Load corpus data
    corpus = {}
    with open(clean_document_path, "r") as f:
        for line in tqdm(f, desc="Loading corpus"):
            doc = json.loads(line)
            corpus[doc["did"]] = doc["cleaned"]


    # 3. embed
    # Aggregated dictionaries to hold embeddings.
    
    batch_size = 16
    dids = list(corpus.keys())
    batched_did_groups = [dids[i:i + batch_size] for i in range(0, len(dids), batch_size)]
    
    # Aggregated dictionaries to hold embeddings.
    aggregated_doc_embeddings = {}
    
    # Process documents in batches.
    with torch.no_grad():
        for batched_did_group in tqdm(batched_did_groups, desc="Embedding corpus"):
            batched_docs = [corpus[did] for did in batched_did_group]
            inputs = tokenizer(batched_docs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(encoder.device)
            outputs = encoder(**inputs)
            hidden = outputs.last_hidden_state               # [B, L, d]
            mask   = inputs.attention_mask.unsqueeze(-1)      # [B, L, 1]
            summed = (hidden * mask).sum(dim=1)               # [B, d]
            length = mask.sum(dim=1)                            # [B, 1]
            batched_doc_embeddings = summed / length.clamp(min=1e-9)         # [B, d]
            # batched_doc_embeddings = outputs.mean(dim=1)

            # Aggregate embeddings in dictionaries.
            for did, doc_emb in zip(batched_did_group, batched_doc_embeddings):
                aggregated_doc_embeddings[did] = doc_emb
    
    # Save the aggregated embeddings dictionaries as single files.
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