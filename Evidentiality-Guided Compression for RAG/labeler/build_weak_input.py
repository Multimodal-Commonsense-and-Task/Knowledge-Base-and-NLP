"""Apply condition 1 (strong evidentiality) and build the input for condition 2.

Condition 1: *without* the sentence the LLM cannot produce the correct answer, but
*with* it the LLM can. Sentences that satisfy it are strong evidence; the rest of the
answer-containing sentences stay candidates for the weak/distractor decision.

Questions the LLM already answers closed-book carry no evidentiality signal (the
answer may come from parametric knowledge alone) and are dropped here.

For condition 2 each candidate is then paired with a few distractor sentences drawn
from the same question. Running the LLM on those bundles (step 4) reveals whether the
companions interfere with the evidence.
"""

import argparse
import json
import os

from tqdm import tqdm

from src.signals import load_exactmatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_sentence', type=str, required=True,
                        help='{split}_per_sentence.json from step 1')
    parser.add_argument('--flat', type=str, required=True,
                        help='{split}_flat.json from step 1')
    parser.add_argument('--closed_book_result', type=str, required=True,
                        help='output dir of the --n_context 0 run')
    parser.add_argument('--strong_result', type=str, required=True,
                        help='output dir of the --n_context 1 run')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--num_distractors', type=int, default=4,
                        help='distractor sentences bundled with each candidate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.per_sentence) as f:
        per_sentence = json.load(f)
    with open(args.flat) as f:
        flat = json.load(f)

    closed_book_em = load_exactmatch(args.closed_book_result)
    strong_em = load_exactmatch(args.strong_result)

    assert len(closed_book_em) == len(per_sentence), \
        f'closed-book run covers {len(closed_book_em)} of {len(per_sentence)} questions'
    assert len(strong_em) == len(flat), \
        f'per-sentence run covers {len(strong_em)} of {len(flat)} sentences'

    # group the per-sentence predictions by the question they came from
    by_question = {}
    for k, record in enumerate(flat):
        by_question.setdefault(record['qid'], []).append((record['ctxs'][0], strong_em[k]))

    candidates, perturb = [], []
    for qid, example in enumerate(tqdm(per_sentence, desc='applying condition 1')):
        if closed_book_em[qid] == 1:
            # answerable from parametric knowledge; no evidence can be isolated
            continue

        elem = {
            'qid': qid,
            'question': example['question'],
            'answers': example['answers'],
            'positive_ctxs_1': [],       # strong evidence (condition 1 satisfied)
            'hard_negative_ctxs_1': [],  # answer-bearing but not sufficient on its own
            'candidate_ctxs': [],        # everything sent on to condition 2
            'negative_ctxs': [c for c in example['ctxs'] if not c['has_answer']],
        }

        for ctx, em in by_question.get(qid, []):
            if em == 1:
                elem['positive_ctxs_1'].append(ctx)
            else:
                elem['hard_negative_ctxs_1'].append(ctx)
            elem['candidate_ctxs'].append(ctx)

        if not elem['candidate_ctxs']:
            continue
        candidates.append(elem)

        for ctx in elem['candidate_ctxs']:
            perturb.append({
                'qid': qid,
                'question': example['question'],
                'answers': example['answers'],
                'ctxs': [ctx] + elem['negative_ctxs'][:args.num_distractors],
            })

    candidates_path = os.path.join(args.output_dir, f'{args.split}_candidates.json')
    with open(candidates_path, 'w') as f:
        json.dump(candidates, f)
    print(f'{len(candidates)} questions -> {candidates_path}')

    perturb_path = os.path.join(args.output_dir, f'{args.split}_perturb.json')
    with open(perturb_path, 'w') as f:
        json.dump(perturb, f)
    print(f'{len(perturb)} candidate bundles -> {perturb_path}')


if __name__ == '__main__':
    main()
