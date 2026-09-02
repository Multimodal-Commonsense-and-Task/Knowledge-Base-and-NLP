"""Run the reader LLM over a file and record, per example, whether it produced the
correct answer.

This single script covers every LLM call of the labeling stage; only the input file
and ``--n_context`` change:

  closed-book    --eval_data {split}_per_sentence.json  --n_context 0
  strong mining  --eval_data {split}_flat.json          --n_context 1
  weak mining    --eval_data {split}_perturb.json       --n_context 5

The output ``result.json`` holds an ``exactmatch`` list aligned with the input order,
which the label-building steps consume. This is by far the most expensive step, so the
work can be split across GPUs with ``--shard_id`` / ``--num_shards``.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import src.data
import src.evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='google/flan-ul2')
    parser.add_argument('--n_context', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--load_in_8bit', action='store_true', default=True)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    result_path = os.path.join(args.output_path, f'result_{args.shard_id}.json')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    eval_examples = src.data.load_data(args.eval_data)

    # contiguous shards, so concatenating result_0..result_{n-1} restores input order
    total = len(eval_examples)
    per_shard = (total + args.num_shards - 1) // args.num_shards
    start = args.shard_id * per_shard
    end = min(start + per_shard, total)
    eval_examples = eval_examples[start:end]
    print(f'shard {args.shard_id}/{args.num_shards}: examples [{start}, {end}) of {total}')

    eval_dataset = src.data.Dataset(eval_examples, tokenizer, args.n_context, args.max_length)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        device_map='auto',
        load_in_8bit=args.load_in_8bit,
    )
    model.eval()

    exactmatch, pred_list = [], []
    result = {'shard_id': args.shard_id, 'num_shards': args.num_shards,
              'start': start, 'end': end}

    for i in tqdm(range(len(eval_dataset))):
        example = eval_dataset[i]
        with torch.no_grad():
            output = model.generate(
                example['input_ids'].cuda(),
                attention_mask=example['attention_mask'].cuda(),
                max_new_tokens=100,
            )
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        gold = eval_dataset.get_example(i)['answers']

        exactmatch.append(src.evaluation.ems(pred, gold))
        pred_list.append(pred)

        if (i + 1) % args.save_every == 0:
            result.update(total_em=sum(exactmatch) / len(exactmatch),
                          exactmatch=exactmatch, pred=pred_list)
            with open(result_path, 'w') as f:
                json.dump(result, f)

    result.update(total_em=sum(exactmatch) / max(len(exactmatch), 1),
                  exactmatch=exactmatch, pred=pred_list)
    with open(result_path, 'w') as f:
        json.dump(result, f)
    print(f"EM: {result['total_em']:.4f} -> {result_path}")


if __name__ == '__main__':
    main()
