#!/usr/bin/env python3
"""
token_entropy_uncertainty_avg.py - Epistemic uncertainty based on average token entropy.

Calculates document uncertainty by processing the hidden states of ALL tokens
through an MLM head and computing the simple average of their entropies.

Example Usage:
$ python token_entropy_uncertainty_avg.py \
    --retriever_model_name_or_path facebook/contriever-msmarco \
    --mlm_model_name_or_path bert-base-uncased \
    --corpus_path corpus.jsonl \
    --output_path doc_uncertainty_avg.json \
    --batch_size 32
"""

import argparse
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM, DistilBertForMaskedLM

# For reproducibility
torch.manual_seed(42)

def get_mlm_head(model_name: str, device: str) -> torch.nn.Module:
    """Loads a pretrained model and extracts its MLM head."""
    print(f"[*] Loading MLM head from: {model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name).eval().to(device)

    if isinstance(model, BertForMaskedLM):
        return model.cls
    elif isinstance(model, DistilBertForMaskedLM):
        # For DistilBERT, the head is a sequence of layers
        return torch.nn.Sequential(
            model.vocab_transform,
            model.vocab_layer_norm,
            model.vocab_projector,
        )
    elif hasattr(model, 'lm_head'):
        return model.lm_head
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")

def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates the entropy of a batch of logit distributions.
    Entropy H(p) = -Σ(p * log(p))
    Args:
        logits (torch.Tensor): A tensor of shape [N, VocabSize].
    Returns:
        torch.Tensor: A tensor of shape [N] containing the entropy for each distribution.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def main(args):
    # 1. ============== LOAD MODELS and DATA ==============
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load the retriever model (e.g., Contriever)
    print(f"[*] Loading retriever model: {args.retriever_model_name_or_path}")
    retriever = SentenceTransformer(args.retriever_model_name_or_path, device=device)
    retriever_tokenizer = retriever.tokenizer

    # Load the MLM head from a separate model (e.g., BERT)
    mlm_head = get_mlm_head(args.mlm_model_name_or_path, device)

    # Load the corpus
    print(f"[*] Loading corpus from: {args.corpus_path}")
    docs, dids = [], []
    with open(args.corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            dids.append(obj["did"])
            docs.append(obj["cleaned"]) # Assuming documents have a "cleaned" field

    # 2. ============== COMPUTE UNCERTAINTY IN BATCHES ==============
    uncertainty_dict = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(docs), args.batch_size), desc="Calculating Average Token Entropy"):
            batch_texts = docs[i:i+args.batch_size]
            batch_dids = dids[i:i+args.batch_size]

            # A) Get token-level hidden states from the retriever
            encoded = retriever.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            model_output = retriever[0].auto_model(**encoded)
            token_embeddings = model_output.last_hidden_state # [B, SeqLen, Dim]

            # B) Process each document in the batch
            for j in range(len(batch_dids)):
                doc_did = batch_dids[j]
                doc_token_embeddings = token_embeddings[j] # [SeqLen, Dim]
                doc_input_ids = encoded['input_ids'][j]    # [SeqLen]
                attention_mask = encoded['attention_mask'][j] # [SeqLen]

                # Filter out padding tokens
                actual_token_ids = doc_input_ids[attention_mask == 1]
                actual_embeddings = doc_token_embeddings[attention_mask == 1]
                
                # Ignore [CLS] and [SEP] tokens if they exist
                if retriever_tokenizer.cls_token_id is not None:
                    actual_embeddings = actual_embeddings[actual_token_ids != retriever_tokenizer.cls_token_id]
                    actual_token_ids = actual_token_ids[actual_token_ids != retriever_tokenizer.cls_token_id]
                if retriever_tokenizer.sep_token_id is not None:
                    actual_embeddings = actual_embeddings[actual_token_ids != retriever_tokenizer.sep_token_id]
                    actual_token_ids = actual_token_ids[actual_token_ids != retriever_tokenizer.sep_token_id]

                if len(actual_embeddings) == 0:
                    uncertainty_dict[doc_did] = 0.0
                    continue
                
                # C) Calculate entropy for ALL tokens
                logits = mlm_head(actual_embeddings)   # [NumTokens, VocabSize]
                entropies = calculate_entropy(logits)  # [NumTokens]

                # D) Aggregate by taking the simple average
                final_uncertainty = torch.mean(entropies).item()
                uncertainty_dict[doc_did] = final_uncertainty

    # 3. ============== SAVE RESULTS ==============
    print(f"[✓] Saving uncertainty dictionary to: {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(uncertainty_dict, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average token-level entropy uncertainty for a corpus.")
    parser.add_argument("--retriever_model_name_or_path", type=str, required=True, help="Path to the SentenceTransformer model.")
    parser.add_argument("--mlm_model_name_or_path", type=str, required=True, help="Path to the model for the MLM head (e.g., 'bert-base-uncased').")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the input corpus file (JSONL format).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final uncertainty dictionary.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    main(args)