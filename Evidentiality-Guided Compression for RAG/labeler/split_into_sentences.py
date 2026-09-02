"""Decompose retrieved documents into sentences.

ECoRAG compresses at sentence granularity, so the top-100 retrieved documents are
first split with NLTK and each sentence is re-checked for answer containment.

Two outputs are produced:
  * ``{split}_per_sentence.json`` - one record per question, ``ctxs`` are sentences.
    This is the input of the closed-book run (``--n_context 0``) and the reference
    file used when the labels are assembled.
  * ``{split}_flat.json`` - one record per sentence (``ctxs`` has length 1), which is
    what the per-sentence LLM runs and the compressor inference consume.

``--mode label`` keeps only answer-containing sentences (capped per question) and is
used for the training split; ``--mode compress`` keeps every sentence and is used to
build the compressor's inference input for the test split.
"""

import argparse
import json
import os

from nltk import sent_tokenize
from tqdm import tqdm

from src.evaluation import SimpleTokenizer, has_answer


def split_into_sentences(data, tokenizer):
    out = []
    for example in tqdm(data, desc='splitting documents into sentences'):
        ctxs = []
        for ctx in example['ctxs']:
            for sent in sent_tokenize(ctx['text']):
                ctxs.append({
                    'id': ctx['id'],
                    'title': ctx['title'],
                    'text': sent,
                    'score': ctx['score'],
                    'has_answer': has_answer(example['answers'], sent, tokenizer),
                })
        out.append({
            'question': example['question'],
            'answers': example['answers'],
            'ctxs': ctxs,
        })
    return out


def flatten(data, mode, max_sents_per_question):
    """One record per sentence, tagged with the position it came from.

    ``qid``/``ctx_idx`` let the label-building steps join the LLM predictions back
    onto the per-question file without relying on positional alignment.
    """
    flat = []
    for qid, example in enumerate(tqdm(data, desc='flattening')):
        kept = 0
        for ctx_idx, ctx in enumerate(example['ctxs']):
            if mode == 'label' and not ctx['has_answer']:
                continue
            flat.append({
                'qid': qid,
                'ctx_idx': ctx_idx,
                'question': example['question'],
                'answers': example['answers'],
                'ctxs': [ctx],
            })
            kept += 1
            if mode == 'label' and kept >= max_sents_per_question:
                break
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True,
                        help='DPR top-100 retrieval output')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True, help='train / dev / test')
    parser.add_argument('--mode', type=str, default='label', choices=['label', 'compress'],
                        help='label: keep answer-containing sentences only (training split). '
                             'compress: keep every sentence (inference split).')
    parser.add_argument('--max_sents_per_question', type=int, default=16,
                        help='cap on answer-containing sentences per question in label mode')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = SimpleTokenizer()

    with open(args.input_data) as f:
        data = json.load(f)

    per_sentence = split_into_sentences(data, tokenizer)
    per_sentence_path = os.path.join(args.output_dir, f'{args.split}_per_sentence.json')
    with open(per_sentence_path, 'w') as f:
        json.dump(per_sentence, f)
    print(f'{len(per_sentence)} questions -> {per_sentence_path}')

    flat = flatten(per_sentence, args.mode, args.max_sents_per_question)
    flat_path = os.path.join(args.output_dir, f'{args.split}_flat.json')
    with open(flat_path, 'w') as f:
        json.dump(flat, f)
    print(f'{len(flat)} sentences -> {flat_path}')


if __name__ == '__main__':
    main()
