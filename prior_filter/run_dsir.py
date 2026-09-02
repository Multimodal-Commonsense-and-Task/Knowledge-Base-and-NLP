import json
import torch
import numpy as np
from pathlib import Path
import argparse
import pyarrow as pa
import pyarrow.ipc as ipc

from dsir.data_selection import HashedNgramDSIR

def parse_args():
    parser = argparse.ArgumentParser(description="Run DSIR filtering pipeline.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., openwebtext)')
    parser.add_argument('--num_shards', type=int, required=True, help='Number of shards (e.g., 129)')
    parser.add_argument('--raw_path_template', type=str, required=True, help='Path template to raw .arrow files (use {subset} for shard index)')
    parser.add_argument('--target_path', type=str, required=True, help='Path to target dataset (.arrow file)')
    parser.add_argument('--cache_dir', type=str, required=True, help='Directory to store cache')
    parser.add_argument('--mid_output_dir', type=str, required=True, help='Directory to store intermediate resampled output')
    parser.add_argument('--final_output_path', type=str, required=True, help='Path to save final filtering index (.pt)')
    parser.add_argument('--num_to_sample', type=int, required=True, help='Number of examples to sample')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes for multiprocessing')
    parser.add_argument('--min_len', type=int, default=100, help='Minimum token length per example')
    return parser.parse_args()

# Count total number of examples (sum of Arrow file rows)
def count_examples(raw_paths):
    total = 0
    for file_path in raw_paths:
        try:
            with pa.memory_map(file_path, 'r') as source:
                try:
                    reader = ipc.open_file(source)
                except pa.ArrowInvalid:
                    reader = ipc.open_stream(source)
                table = reader.read_all()
                total += table.num_rows
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    return total

# Parsing function: returns token ID list as is
def parse_example(ex):
    # The input example is a 'dict' type, and the "input_ids" field contains the token ID list
    if "input_ids" in ex:
        return ex["input_ids"]
    else:
        raise ValueError("Missing 'input_ids' field in example.")

def main():
    args = parse_args()

    # Load datasets
    subsets = [str(i).zfill(5) for i in range(args.num_shards)]
    raw_datasets = [args.raw_path_template.format(subset=s) for s in subsets]
    target_datasets = [args.target_path]

    # Count dataset size
    total_examples = count_examples(raw_datasets)
    print(f"Total number of examples: {total_examples}")
    num_to_sample = args.num_to_sample

    # Initialize DSIR (use token IDs directly)
    dsir = HashedNgramDSIR(
        raw_datasets=raw_datasets,
        target_datasets=target_datasets,
        cache_dir=args.cache_dir,
        raw_parse_example_fn=parse_example,
        target_parse_example_fn=parse_example,
        num_proc=args.num_proc,               
        ngrams=2,                             # Unigram + Bigram (adjust if needed)
        num_buckets=10000,                    # Number of hash buckets (adjust if needed)
        tokenizer="gpt2",                     # Assume input is already tokenized as ID list (adjust if needed)
        min_example_length=args.min_len,      
        target_laplace_smoothing=0.0,
        separate_targets=False                # Treat as single target set
    )

    # Run DSIR experiment
    dsir.fit_importance_estimator(num_tokens_to_fit='auto')
    print("Importance estimator training complete.")

    dsir.compute_importance_weights()
    print("Importance weight computation complete.")

    dsir.resample(out_dir=args.mid_output_dir, num_to_sample=num_to_sample, cache_dir=None, top_k=False)
    print(f"Resampling complete. Output saved to: {args.mid_output_dir}")

    # Generate filtering index
    # Load selected IDs
    selected_files = sorted(Path(args.mid_output_dir).glob("*.jsonl"))
    selected_ids = set()
    for file in selected_files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    selected_ids.add(ex["id"])
                except Exception as e:
                    print(f"Error in {file}: {e}")

    print(f"Selected {len(selected_ids)} examples.")

    # Create and save index
    filtering_index = np.zeros(total_examples, dtype=bool)
    for idx in selected_ids:
        if idx < total_examples:
            filtering_index[idx] = True

    torch.save(torch.tensor(filtering_index), args.final_output_path)
    print(f"Filtering index saved to {args.final_output_path}")

if __name__ == "__main__":
    main()