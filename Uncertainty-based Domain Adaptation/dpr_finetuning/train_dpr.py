#!/usr/bin/env python

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util

##############################################################################
# 1) Dataset + DataLoader
##############################################################################

class DPRTripletDataset(Dataset):
    """
    Expects a TSV file where each line: query \t pos_passage \t neg_passage
    """
    def __init__(self, path):
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                q, p_pos, p_neg = line.strip().split('\t')
                self.samples.append((q, p_pos, p_neg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def dpr_collate_fn(batch):
    """
    Collate function that just returns lists of queries, positives, negatives.
    We'll tokenize inside the training loop using the sentence-transformers model.
    """
    queries = [item[0] for item in batch]
    positives = [item[1] for item in batch]
    negatives = [item[2] for item in batch]
    return queries, positives, negatives

##############################################################################
# 2) Training Loop
##############################################################################

def train_dpr_sbert(
    question_encoder, passage_encoder, train_loader, device, epochs=1, lr=1e-5
):
    """
    A simple training loop using two SentenceTransformer models:
      - question_encoder for the queries
      - passage_encoder for the passages
    Each iteration:
      - tokenizes queries & docs via .tokenize()
      - calls model(...) in train mode
      - dot-product scores
      - cross-entropy on [pos_score, neg_score]
    """

    # Collect all parameters from both encoders
    param_optimizer = list(question_encoder.parameters()) + list(passage_encoder.parameters())
    optimizer = optim.AdamW(param_optimizer, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    question_encoder.train()
    passage_encoder.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (queries, pos_docs, neg_docs) in enumerate(train_loader):
            # We'll move everything to device after tokenization
            queries = list(queries)
            pos_docs = list(pos_docs)
            neg_docs = list(neg_docs)

            # -------------------------------------------
            # 2.1: Forward pass for queries
            # -------------------------------------------
            # We create sentence_transformers "features" by calling model.tokenize(...)
            # Then do a forward() call (NOT encode()) so we can get gradients.

            query_features = question_encoder.tokenize(queries)  # returns dict with input_ids, etc.
            for k in query_features:
                query_features[k] = query_features[k].to(device)

            # forward() returns a dict with "sentence_embedding" by default
            query_output = question_encoder.forward(query_features)
            q_reps = query_output["sentence_embedding"]  # shape [batch, hidden_dim]

            # -------------------------------------------
            # 2.2: Forward pass for positive passages
            # -------------------------------------------
            pos_features = passage_encoder.tokenize(pos_docs)
            for k in pos_features:
                pos_features[k] = pos_features[k].to(device)
            pos_output = passage_encoder.forward(pos_features)
            pos_reps = pos_output["sentence_embedding"]

            # -------------------------------------------
            # 2.3: Forward pass for negative passages
            # -------------------------------------------
            neg_features = passage_encoder.tokenize(neg_docs)
            for k in neg_features:
                neg_features[k] = neg_features[k].to(device)
            neg_output = passage_encoder.forward(neg_features)
            neg_reps = neg_output["sentence_embedding"]

            # -------------------------------------------
            # 2.4: Compute cross-entropy over dot-product
            # -------------------------------------------
            # dot-score: q_reps dot pos_reps (batch, hidden_dim) -> (batch)
            pos_scores = torch.sum(q_reps * pos_reps, dim=1).unsqueeze(-1)  # (B,1)
            neg_scores = torch.sum(q_reps * neg_reps, dim=1).unsqueeze(-1)  # (B,1)

            scores = torch.cat([pos_scores, neg_scores], dim=1)  # shape (B,2)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=device)  # 0 => pos doc

            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss = {avg_loss:.4f}")


##############################################################################
# 3) Main Script
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Train DPR (two separate SBERT models) on Hard-Negative Triplets")
    parser.add_argument("--triplets_path", type=str, required=True,
                        help="TSV file with lines: query, positive_passage, negative_passage")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned question & ctx encoders")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1. Load dataset
    train_dataset = DPRTripletDataset(args.triplets_path)
    print(f"Loaded {len(train_dataset)} triplets from {args.triplets_path}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dpr_collate_fn
    )

    # 2. Initialize separate SBERT encoders
    print("Loading question encoder from 'facebook-dpr-question_encoder-multiset-base'...")
    question_encoder = SentenceTransformer("facebook-dpr-question_encoder-multiset-base", device=device)

    print("Loading passage encoder from 'facebook-dpr-ctx_encoder-multiset-base'...")
    passage_encoder = SentenceTransformer("facebook-dpr-ctx_encoder-multiset-base", device=device)

    question_encoder.to(device)
    passage_encoder.to(device)

    # 3. Train
    train_dpr_sbert(
        question_encoder, passage_encoder,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr
    )

    # 4. Save
    os.makedirs(args.output_dir, exist_ok=True)
    question_encoder.save(os.path.join(args.output_dir, "question_encoder"))
    passage_encoder.save(os.path.join(args.output_dir, "ctx_encoder"))
    print(f"Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
