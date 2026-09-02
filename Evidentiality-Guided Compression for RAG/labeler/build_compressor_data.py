"""Apply condition 2 and emit the compressor's training data.

Condition 2 asks whether a sentence interferes with the evidence. Each candidate was
bundled with a few distractor sentences in step 3; the result of that run decides:

  * bundle answered correctly - the companion sentences did not interfere, so they
    are weak evidence;
  * bundle answered wrongly   - the candidate's evidentiality did not survive next to
    other sentences, so the candidate itself is demoted to weak evidence.

The three levels then map onto the dual-encoder training fields:

  positive_ctxs      strong evidence  (d*)
  hard_negative_ctxs weak evidence    (d+)
  negative_ctxs      distractor       (d-)
"""

import argparse
import json
import os

from tqdm import tqdm

from src.signals import dedup, load_exactmatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=str, required=True,
                        help='{split}_candidates.json from step 3')
    parser.add_argument('--perturb', type=str, required=True,
                        help='{split}_perturb.json from step 3')
    parser.add_argument('--perturb_result', type=str, required=True,
                        help='output dir of the --n_context 5 run')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--min_negatives', type=int, default=17,
                        help='drop questions with too few distractors to sample from')
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.candidates) as f:
        candidates = json.load(f)
    with open(args.perturb) as f:
        perturb = json.load(f)

    perturb_em = load_exactmatch(args.perturb_result)
    assert len(perturb_em) == len(perturb), \
        f'perturbation run covers {len(perturb_em)} of {len(perturb)} bundles'

    # weak evidence collected per question, in perturb-file order
    weak_by_question = {}
    for bundle, em in zip(perturb, perturb_em):
        weak = weak_by_question.setdefault(bundle['qid'], [])
        if em == 1:
            weak.extend(bundle['ctxs'][1:])   # companions did not interfere
        else:
            weak.append(bundle['ctxs'][0])    # the candidate itself did not hold up

    labeled = []
    for elem in tqdm(candidates, desc='applying condition 2'):
        strong = dedup(elem['positive_ctxs_1'])
        weak = dedup(weak_by_question.get(elem['qid'], []))
        weak_keys = {frozenset(c.items()) for c in weak}
        distractors = [c for c in elem['negative_ctxs']
                       if frozenset(c.items()) not in weak_keys]

        if not strong or len(distractors) < args.min_negatives:
            continue

        labeled.append({
            'question': elem['question'],
            'answers': elem['answers'],
            'positive_ctxs': strong,
            'hard_negative_ctxs': weak,
            'negative_ctxs': distractors,
        })

    split_at = int(len(labeled) * (1 - args.dev_ratio))
    train, dev = labeled[:split_at], labeled[split_at:]

    for name, part in (('train', train), ('dev', dev)):
        path = os.path.join(args.output_dir, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(part, f)
        print(f'{len(part)} questions -> {path}')

    n_strong = sum(len(d['positive_ctxs']) for d in labeled)
    n_weak = sum(len(d['hard_negative_ctxs']) for d in labeled)
    n_dist = sum(len(d['negative_ctxs']) for d in labeled)
    print(f'labeled sentences: strong={n_strong} weak={n_weak} distractor={n_dist}')


if __name__ == '__main__':
    main()
