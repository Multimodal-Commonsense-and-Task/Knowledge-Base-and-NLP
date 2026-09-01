#!/usr/bin/env python3
"""
token_entropy_uncertainty.py - Epistemic uncertainty based on token-level entropy.

Calculates document uncertainty by processing the hidden states of high-IDF tokens
through an MLM head and computing the weighted average of their entropies.

Example Usage:
$ python token_entropy_uncertainty.py \
    --retriever_model_name_or_path facebook/contriever-msmarco \
    --mlm_model_name_or_path bert-base-uncased \
    --corpus_path corpus.jsonl \
    --idf_dict_path idf_dictionary.json \
    --output_path doc_uncertainty.json \
    --top_k_idf 10 \
    --batch_size 32
"""

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM, DistilBertForMaskedLM, AutoModel

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
    # Use log_softmax for numerical stability
    # probs = p = softmax(logits)
    # log_probs = log(p) = log_softmax(logits)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def main(args):
    # 1. ============== LOAD MODELS and DATA ==============
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load the base model (e.g., Contriever)
    print(f"[*] Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModel.from_pretrained(args.base_model_name_or_path).to(device)
    # We need the underlying tokenizer for its vocabulary
    base_model_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    # Load the MLM head from a separate model (e.g., BERT)
    mlm_head = get_mlm_head(args.mlm_model, device)

    # Load the IDF dictionary
    print(f"[*] Loading IDF dictionary from: {args.idf_dict_path}")
    with open(args.idf_dict_path, 'r') as f:
        # JSON keys are strings, so we'll access them with stringified token IDs
        idf_dict = json.load(f)

    # Load the corpus
    print(f"[*] Loading corpus from: {args.cleaned_document_path}")
    docs, dids = [], []
    with open(args.cleaned_document_path, "r") as f:
        for line in tqdm(f, desc="Loading corpus"):
            doc = json.loads(line)
            dids.append(doc["did"])
            docs.append(doc["cleaned"])  # Assuming documents have a "cleaned" field

    # 2. ============== COMPUTE UNCERTAINTY IN BATCHES ==============
    uncertainty_dict = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(docs), args.batch_size), desc="Calculating Token-Level Uncertainty"):
            batch_texts = docs[i:i+args.batch_size]
            batch_dids = dids[i:i+args.batch_size]

            # A) Get token-level hidden states from the retriever
            # We use `model.encode` with `output_value` to get token embeddings directly
            # This is more aligned with the SBERT API.
            inputs = base_model_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            # Directly use the transformer model for outputting all hidden states
            model_output = base_model(**inputs)
            token_embeddings = model_output.last_hidden_state # [B, SeqLen, Dim]

            # B) Process each document in the batch
            for j in range(len(batch_dids)):
                doc_did = batch_dids[j]
                doc_token_embeddings = token_embeddings[j] # [SeqLen, Dim]
                doc_input_ids = inputs['input_ids'][j]    # [SeqLen]
                attention_mask = inputs['attention_mask'][j] # [SeqLen]

                # Filter out padding tokens
                actual_token_ids = doc_input_ids[attention_mask == 1]
                actual_embeddings = doc_token_embeddings[attention_mask == 1]
                
                # Ignore [CLS] and [SEP] tokens if they exist
                if base_model_tokenizer.cls_token_id is not None:
                    actual_embeddings = actual_embeddings[actual_token_ids != base_model_tokenizer.cls_token_id]
                    actual_token_ids = actual_token_ids[actual_token_ids != base_model_tokenizer.cls_token_id]
                if base_model_tokenizer.sep_token_id is not None:
                    actual_embeddings = actual_embeddings[actual_token_ids != base_model_tokenizer.sep_token_id]
                    actual_token_ids = actual_token_ids[actual_token_ids != base_model_tokenizer.sep_token_id]

                if len(actual_token_ids) == 0:
                    uncertainty_dict[doc_did] = 0.0
                    continue

                # C) Find top-k IDF tokens
                token_idf_pairs = []
                for token_id in actual_token_ids:
                    # Use a default IDF of 1.0 (log(1)=0) for unknown words
                    idf_score = idf_dict.get(str(token_id.item()), 1.0)
                    token_idf_pairs.append((token_id, idf_score))
                
                # Sort by IDF score in descending order and get top k
                token_idf_pairs.sort(key=lambda x: x[1], reverse=True)
                top_k_tokens = token_idf_pairs[:args.top_k_idf]
                
                if not top_k_tokens:
                    uncertainty_dict[doc_did] = 0.0
                    continue

                # D) Get hidden states and weights for these specific tokens
                indices_to_select = []
                idf_weights = []
                top_k_ids = [pair[0] for pair in top_k_tokens]
                
                for token_id in top_k_ids:
                    # Find all occurrences of this token_id and select their embeddings
                    # This handles cases where a high-IDF word appears multiple times
                    matches = (actual_token_ids == token_id).nonzero(as_tuple=True)[0]
                    indices_to_select.extend(matches.tolist())
                
                # Ensure unique indices
                unique_indices = sorted(list(set(indices_to_select)))
                selected_embeddings = actual_embeddings[unique_indices]
                
                # Get the corresponding IDF weights for the selected unique tokens
                selected_ids = actual_token_ids[unique_indices]
                idf_weights = torch.tensor([idf_dict.get(str(tid.item()), 1.0) for tid in selected_ids], device=device)

                # E) Calculate weighted entropy
                # Pass selected states through MLM head to get logits
                logits = mlm_head(selected_embeddings) # [NumSelected, VocabSize]
                entropies = calculate_entropy(logits)  # [NumSelected]

                # F) Aggregate into a single uncertainty score
                weighted_sum_of_entropies = torch.sum(entropies * idf_weights)
                sum_of_weights = torch.sum(idf_weights)
                
                if sum_of_weights > 0:
                    final_uncertainty = (weighted_sum_of_entropies / sum_of_weights).item()
                else:
                    final_uncertainty = 0.0

                uncertainty_dict[doc_did] = final_uncertainty

    # 3. ============== SAVE RESULTS ==============
    print(f"[✓] Saving uncertainty dictionary to: {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(uncertainty_dict, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate token-level entropy uncertainty for a corpus.")
    parser.add_argument("--base_model_name_or_path", type=str, required=True, help="Path to the SentenceTransformer model.")
    parser.add_argument("--mlm_model", type=str, required=True, help="Path to the model for the MLM head (e.g., 'bert-base-uncased').")
    parser.add_argument("--cleaned_document_path", type=str, required=True, help="Path to the input corpus file (JSONL format).")
    parser.add_argument("--idf_dict_path", type=str, required=True, help="Path to the IDF dictionary JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final uncertainty dictionary.")
    parser.add_argument("--top_k_idf", type=int, default=10, help="Number of top IDF tokens to consider per document.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    main(args)