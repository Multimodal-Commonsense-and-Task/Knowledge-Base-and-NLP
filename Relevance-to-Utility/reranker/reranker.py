import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import torch
import pickle
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.cuda()
model.eval()

def pickle_load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pickle_dump(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def json_load(path):
    with open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def json_dump(path, data):
    with open(path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def chunk_text(text, max_tokens=512, stride=256, tokenizer=None):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
        if i + max_tokens >= len(tokens):
            break
    return chunks

def rerank_document_chunks(query, chunks, concat='mean'):
    pairs = [(query, chunk) for chunk in chunks]

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
        scores = model(**inputs, return_dict=True).logits.view(-1,).float()
        if concat == 'mean':
            score = scores.mean().item()
        elif concat == 'max':
            score = scores.max().item()
    return score

def compute_rerank_scores(pairs):
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
        scores = model(**inputs, return_dict=True).logits.view(-1,).float()
        scores = scores.clone().detach().cpu().tolist()
    return scores

def main():
    path = '../cache/hotpotqa'
    cache_path_list = [
        'search_cache_500.json', 
    ]


    for cache_path in cache_path_list:
        cache = json_load(os.path.join(path, cache_path))
        
        ranked_cache = {}
        for question, docs in tqdm(cache.items()):
            scores = []
            for i, doc in enumerate(docs):
                if len(doc.get('title', '')) > 0:
                    document = f"**Title:** {doc.get('title', '')}\n"
                    document += f"**Content:** {doc.get('contents', '')}\n\n"
                else:
                    document = doc['contents']

                chunks = chunk_text(document, tokenizer=tokenizer)
                assert len(chunks) > 0, question
                score = rerank_document_chunks(question, chunks, concat = 'mean')
                scores.append((score, i))

            scores = sorted(scores, reverse=True)
            indices = [idx for _, idx in scores]
            ranked_docs = [docs[idx] for idx in indices]
            ranked_cache[question] = ranked_docs
        
        output_path = os.path.join(path, cache_path.replace('.json', '_ranked.json'))
        json_dump(output_path, ranked_cache)
        print(f'Saving {output_path} done.')


if __name__ == "__main__":
    main()