import os
import json
import time
import torch
import faiss
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


import pyterrier as pt
if not pt.started():
    pt.init()
import pyterrier_alpha as pta
# from pyterrier_pisa import PisaIndex
# from pyterrier.datasets import Dataset
from pyterrier_adaptive import CorpusGraph

from utils.utils import pickle_load, pickle_dump, jsonl_load, json_load


class ContrieverRetriever(pt.Transformer):
    def __init__(self, dataset, qa_path, model_name='facebook/contriever-msmarco', output_dir='./data/bright/', batch_size=512, num_results=16):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_results = num_results
        self.device = torch.cuda.current_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.short_model_name = model_name.split("/")[-1]
        self.qa_path = qa_path

        self.qid2excluded_ids = {}
        qa_list = json_load(self.qa_path)
        for qa in qa_list:
            qid = qa['id']
            excluded_ids = qa['excluded_ids']
            self.qid2excluded_ids[qid] = set(excluded_ids)

        self.idx2docno = {idx:data['docno'] for idx, data in enumerate(dataset)}
        texts = [data['content'] for data in dataset]
        print(f"#> {len(texts)} docs will be indexed")

        embedding_path = f"{self.output_dir}/embeddings.{self.short_model_name}.{self.num_results-1}.pickle"
        if not os.path.exists(embedding_path):
            self.index_embeddings = self.encode_texts(texts, self.batch_size)
            self.save_embeddings(embedding_path, self.index_embeddings)
            print(f"#> Saving embeddings {self.index_embeddings.shape}")
        else:
            self.index_embeddings = pickle_load(embedding_path)
            print(f"#> Load embeddings {self.index_embeddings.shape}")
        self.faiss_index = self.build_faiss_index(self.index_embeddings)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode_texts(self, texts, batch_size=512):
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def save_embeddings(self, path, embeddings):
        pickle_dump(path, embeddings)
        print(f"#> Saving {path} done")

    def build_faiss_index(self, embeddings):
        s = time.time()
        
        embeddings = embeddings.astype('float32')
        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # nprobe = 16
        # nlist = 4096

        # quantizer = faiss.IndexFlatIP(dim)
        # index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # num_samples = 204800
        # sample_indices = np.random.choice(embeddings.shape[0], size=num_samples, replace=False)
        # sampled_embeddings = embeddings[sample_indices]
        # print(f"#> sampled_embeddings for training: {sampled_embeddings.shape}")
        # index.train(sampled_embeddings)
        # print(f"#> train done; {time.time() - s}s.")
        # index.add(embeddings)
        # index.nprobe = nprobe
        # print(f"#> add done; {time.time() - s}s.")

        return index

    def transform(self, topics_df):
        queries = topics_df["query"].tolist()
        query_embeddings = self.encode_texts(queries, self.batch_size)
        faiss.normalize_L2(query_embeddings)

        scores, indices = self.faiss_index.search(query_embeddings, self.num_results)

        results = []
        for qid, (idxs, s) in enumerate(zip(indices, scores)):
            qid = topics_df.iloc[qid]['qid']

            for rank, (idx, score) in enumerate(zip(idxs, s)):
                if idx == -1:
                    print(f"#> qid: {qid} has {idx} index")
                    results.append({
                        "qid": qid, 
                        "docno": "-1", 
                        "score": -1, 
                        "rank": rank, 
                    })
                else:
                    results.append({
                        "qid": qid, 
                        "docno": self.idx2docno[idx], 
                        "score": float(score), 
                        "rank": rank, 
                    })
        
        results = pd.DataFrame(results)
        return results

def build_graph(args):
    dataset = jsonl_load(args.corpus_path) # {"docno": str, "content": str, "query": str}, "query" should be the same to "content"
    print(f"#> Load {args.corpus_path}")

    retriever = ContrieverRetriever(
        dataset, 
        qa_path=args.qa_path, 
        output_dir=args.output_dir, 
        batch_size=args.batch_size, 
        num_results=args.k+1
    )
    graph = CorpusGraph.from_retriever(
        retriever,
        dataset,
        f"{args.output_dir}/{args.dataset}.contriever.{args.k}",
        k=args.k
    )
    print(f"#> Build done")

    print(f"#> Sanity Check")
    docno = dataset[0]['docno']
    graph = CorpusGraph.load(f"{args.output_dir}/{args.dataset}.contriever.{args.k}")
    target_neighbors = graph.neighbours(docno)
    print(f"query: {docno}")
    print(target_neighbors)


if __name__ == "__main__":
    build_graph()
