#!/usr/bin/env python3
"""
Preprocesses HotpotQA dataset for EXIT model training.
Creates balanced dataset with positive and negative examples.
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any
from datasets import Dataset
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess HotpotQA for EXIT training")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to hotpot_train_v1.1.json"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save processed datasets"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of examples to use for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--positive_ratio",
        type=float,
        default=0.5,
        help="Ratio of positive to negative examples"
    )
    return parser.parse_args()

def load_hotpotqa(file_path: str) -> List[Dict]:
    """Load HotpotQA dataset."""
    print(f"Loading dataset from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_example(
    example: Dict[str, Any],
    include_titles: bool = True
) -> Dict[str, List]:
    """Process a single HotpotQA example."""
    question = example["question"]
    supporting_facts = {(fact[0], fact[1]) for fact in example["supporting_facts"]}
    supporting_titles = {fact[0] for fact in example["supporting_facts"]}
    
    queries = []
    sentences = []
    contexts = []
    labels = []  # 0: negative, 1: hard negative, 2: positive
    
    for context in example["context"]:
        title = context[0]
        # Create full document text with title
        doc_sentences = context[1]
        context_text = f"{title}\n" + " ".join(doc_sentences) if include_titles else " ".join(doc_sentences)
        
        for i, sentence in enumerate(doc_sentences):
            if not sentence.strip():  # Skip empty sentences
                continue
                
            queries.append(question)
            sentences.append(sentence)
            contexts.append(context_text)
            
            # Determine label
            if (title, i) in supporting_facts:
                labels.append(2)  # Positive
            elif title in supporting_titles:
                labels.append(1)  # Hard negative
            else:
                labels.append(0)  # Negative
    
    return {
        "queries": queries,
        "sentences": sentences,
        "contexts": contexts,
        "labels": labels
    }

def generate_prompt(query: str, context: str, sentence: str) -> str:
    """Generate EXIT model prompt."""
    return (
        f"<start_of_turn>user\n"
        f"Query:\n{query}\n"
        f"Full context:\n{context}\n"
        f"Sentence:\n{sentence}\n"
        f'Is this sentence useful in answering the query? Answer only "Yes" or "No".'
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def create_balanced_dataset(
    examples: List[Dict[str, List]],
    positive_ratio: float = 0.5
) -> List[Dict[str, Any]]:
    """Create balanced dataset from processed examples."""
    positive_samples = []
    hard_negative_samples = []
    negative_samples = []
    
    # Collect samples by type
    for example in examples:
        for query, sentence, ctx, label in zip(
            example["queries"],
            example["sentences"],
            example["contexts"],
            example["labels"]
        ):
            sample = {
                "query": query,
                "full_passage": ctx,
                "sentence_text": sentence.strip(),
                "label": "Yes" if label == 2 else "No",
                "prompt": generate_prompt(query, ctx, sentence.strip()) + 
                         ("Yes" if label == 2 else "No")
            }
            
            if label == 2:
                positive_samples.append(sample)
            elif label == 1:
                hard_negative_samples.append(sample)
            else:
                negative_samples.append(sample)
    
    # Balance dataset
    num_positives = len(positive_samples)
    num_each_negative = int((1 - positive_ratio) * num_positives / 2)
    
    print(f"\nDataset statistics:")
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Hard negative samples: {len(hard_negative_samples)} -> {num_each_negative}")
    print(f"Random negative samples: {len(negative_samples)} -> {num_each_negative}")
    
    balanced_samples = (
        positive_samples +
        random.sample(hard_negative_samples, num_each_negative) +
        random.sample(negative_samples, num_each_negative)
    )
    random.shuffle(balanced_samples)
    
    return balanced_samples

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and split data
    data = load_hotpotqa(args.dataset_path)
    train_data = data[:-args.test_size]
    test_data = data[-args.test_size:]
    
    # Process examples
    print("Processing training examples...")
    train_examples = [
        process_example(example) 
        for example in tqdm(train_data)
    ]
    
    print("\nProcessing test examples...")
    test_examples = [
        process_example(example) 
        for example in tqdm(test_data)
    ]
    
    # Create balanced datasets
    print("\nCreating balanced training dataset...")
    train_samples = create_balanced_dataset(
        train_examples,
        positive_ratio=args.positive_ratio
    )
    
    print("\nCreating balanced test dataset...")
    test_samples = create_balanced_dataset(
        test_examples,
        positive_ratio=args.positive_ratio
    )
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_samples)
    test_dataset = Dataset.from_list(test_samples)
    
    # Save datasets
    print(f"\nSaving datasets to {args.save_dir}")
    train_dataset.save_to_disk(os.path.join(args.save_dir, "train_dataset"))
    test_dataset.save_to_disk(os.path.join(args.save_dir, "test_dataset"))
    
    print("\nDone! Final dataset statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

if __name__ == "__main__":
    main()