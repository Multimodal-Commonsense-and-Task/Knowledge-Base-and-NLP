"""Score the labeled training data with the trained compressor.

The evaluator is trained on the same evidentiality labels, but it must see the
sentences in the order the compressor actually ranks them: at inference time the
reflection loop always consumes the highest-scoring evidence first, and
``train_evaluator.py --hardneg`` takes the top of each list rather than sampling.

This adds ``r_score`` to ``positive_ctxs`` and ``negative_ctxs`` and sorts both lists
by it, producing ``data/evaluator/{task}/{train,dev}.json``.
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

KEYS = ['positive_ctxs', 'negative_ctxs']


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='facebook/contriever')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='checkpoint.pth produced by train_compressor.py')
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu')['model'], strict=False)
    model = model.to(device).eval()

    with open(args.input_data) as f:
        data = json.load(f)

    @torch.no_grad()
    def embed(texts):
        out = []
        for i in range(0, len(texts), args.batch_size):
            inputs = tokenizer(texts[i:i + args.batch_size], padding=True,
                               truncation=True, return_tensors='pt').to(device)
            emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
            out.append(emb.detach().cpu())
        return torch.cat(out, dim=0)

    for example in tqdm(data, desc='scoring'):
        ctxs = [ctx for key in KEYS for ctx in example.get(key, [])]
        if not ctxs:
            continue
        texts = [ctx['title'] + ' ' + ctx['text'] if ctx.get('title') else ctx['text']
                 for ctx in ctxs]
        q_emb = embed([example['question']])[0]
        scores = embed(texts) @ q_emb
        for ctx, score in zip(ctxs, scores.tolist()):
            ctx['r_score'] = score
        for key in KEYS:
            if key in example:
                example[key] = sorted(example[key], key=lambda c: c['r_score'], reverse=True)

    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'{len(data)} questions -> {args.output_path}')


if __name__ == '__main__':
    main()
