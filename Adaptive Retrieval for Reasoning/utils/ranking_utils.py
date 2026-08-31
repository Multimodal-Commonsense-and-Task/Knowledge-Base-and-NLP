import os
import re
import math
import torch
import random
import pickle
import numpy as np
from typing import Dict, List
from types import SimpleNamespace
from pyterrier_adaptive import CorpusGraph
from collections import Counter, defaultdict
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel, AutoModelForSequenceClassification
import pyterrier_alpha as pta
from utils.utils import jsonl_load


def standard_tokenize(text: str):
    """
    Lucene StandardAnalyzer 에 비슷하게 맞춘 아주 단순 버전.
    • lower-case
    • 숫자/알파벳 연속 토큰
    • Unicode letter를 ASCII fold 하지 않았음(필요시 unidecode 사용)
    """
    return re.findall(r"[0-9A-Za-z]+", text.lower())


class InMemoryDocumentRanker:
    def __init__(self, args, k1=0.9, b=0.4):
        """
        docs: dict {docid: str(text)}
        """
        self.args = args
        self.batch_size = self.args.batch_size
        self.k1 = k1
        self.b = b
        self.device = 'cuda'
        self.graph = self.load_graph(args.graph_path)
        self.did2doc = self.load_did2doc(args.corpus_path)

        if 'monot5' in self.args.reranker:
            print(f"Using monot5 as document ranker")
            self.tok = AutoTokenizer.from_pretrained(
                'castorini/monot5-base-msmarco-10k')
            self.model = T5ForConditionalGeneration.from_pretrained(
                'castorini/monot5-base-msmarco-10k').to('cuda').eval()
        elif 'tct' in self.args.reranker:
            print("Using TCT-ColBERT (castorini/tct_colbert-v2-hnp-msmarco)")
            self.tok = AutoTokenizer.from_pretrained(
                'castorini/tct_colbert-v2-hnp-msmarco')
            self.model = AutoModel.from_pretrained(
                'castorini/tct_colbert-v2-hnp-msmarco').to('cuda').eval()
        elif 'contriever' in self.args.reranker:
            self.tok = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
            self.model = AutoModel.from_pretrained('facebook/contriever-msmarco').to('cuda').eval()
        elif 'bge-large' in self.args.reranker:
            self.tok = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
            self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large').to('cuda').eval()
        elif 'bge-base' in self.args.reranker:
            self.tok = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
            self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base').to('cuda').eval()
        elif 'modernbert' in self.args.reranker:
            print("Using ModernBERT (lightonai/Reason-ModernColBERT)")
            from pylate import models
            from pylate import rank
            self.model = models.ColBERT(
                model_name_or_path="lightonai/Reason-ModernColBERT",
                query_length=512,
                document_length=512,
            ).to(self.device).eval()
            self.rank = rank
        if 'l2g' in self.args.reranker:
            self.doc2idx: Dict[str, int] = {}   # pid → row id
            self.idx2doc: List[str] = []        # row id → pid
            self.G = None                       # first‑order doc‑to‑doc matrix (np.float32)
            self.G_k = None                     # multi‑hop version (up to 3 hops)
            self.max_hop = 3                    # hard‑coded per spec, feel free to expose
            self.centrality = None

    def load_did2doc(self, path):
        if path is None:
            return None
        if 'dl19' in path or 'dl20' in path:
            import pyterrier as pt
            dataset = pt.get_dataset('irds:msmarco-passage')
            corpus = dataset.get_corpus_iter()
        else: 
            corpus = jsonl_load(path)
        did2doc = {}
        for doc in corpus:
            if 'content' not in doc:
                doc['content'] = doc.pop('text')
            doc['id'] = doc.pop('docno')
            did2doc[doc['id']] = doc
        print(f"#> Load did2doc: {len(did2doc)} from {path}")
        return did2doc

    def load_graph(self, path):
        if path is None:
            return None
        if 'dl19' in path or 'dl20' in path:
            graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128').to_limit_k(16)
        else:
            graph = CorpusGraph.load(path)
        # graph = graph.to_limit_k(16)
        print(f"#> Load graph from {path}")
        return graph

    @staticmethod
    def convert_content(topk_results):
        # process docs
        docs = []
        for x in topk_results:
            content = x['text']
            if "title" in x and x['title']:
                content = "Title: " + x['title'] + " " + "Content: " + content
            docs.append([x['pid'], content])
        return docs

    def _index_bm25(self, docs):
        self.N = len(docs)

        self.doc_len = {}
        self.avgdl = 0.0
        self.f = defaultdict(Counter)   # term -> {docid: tf}
        self.df = Counter()

        # -------- 색인 --------
        for x in docs:
            did, text = x
            tokens = standard_tokenize(text)
            self.doc_len[did] = len(tokens)
            self.avgdl += len(tokens)

            tf = Counter(tokens)
            for t, c in tf.items():
                self.f[t][did] = c
            for t in tf:
                self.df[t] += 1

        self.avgdl /= self.N

        # -------- IDF 사전 --------
        self.idf = {
            t: math.log1p((self.N - df + 0.5) / (df + 0.5))   # log(1 + …)
            for t, df in self.df.items()
        }

    # ----------------------------
    def _score(self, query_tokens, did):
        score = 0.0
        dl   = self.doc_len[did]
        for t in query_tokens:
            if did not in self.f.get(t, {}):  # 등장 X
                continue
            tf = self.f[t][did]
            idf = self.idf.get(t, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def _tct_rank(self, queries, docs):
        """
        docs : List[content]
        returns : List['id':pid, 'score':score)]
        """
        num_queries = len(queries)
        with torch.no_grad():
            q_vecs = self.model.encode(
                queries,
                batch_size=self.batch_size,
                is_query=True, # Encoding queries
                show_progress_bar=False,
            )

        pids = [doc['id'] for doc in docs]
        docs = [doc['content'] for doc in docs]
        with torch.no_grad():
            doc_vecs = self.model.encode(
                docs,
                batch_size=self.batch_size,
                is_query=False, # Encoding docs
                show_progress_bar=False,
            )
        
        doc_vecs = [doc_vecs.copy() for _ in range(num_queries)] # shape: (n_queries, n_docs, 128)
        pids = [list(pids).copy() for _ in range(num_queries)] # shape: (n_queries, n_docs)

        ranked_docs = self.rank.rerank(
            documents_ids=pids,
            queries_embeddings=q_vecs,
            documents_embeddings=doc_vecs,
        ) # shape: (n_queries, n_docs)
        return ranked_docs

    def _contriever_rank(self, queries, docs):
        # Mean pooling
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        q_vecs = []
        for s in range(0, len(queries), self.batch_size):
            e = s + self.batch_size
            batch_texts = queries[s:e]
            q_tok = self.tok(batch_texts, padding=True, truncation=True,
                             max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**q_tok)
            q_vec = mean_pooling(outputs[0], q_tok['attention_mask'])
            q_vecs.append(q_vec.cpu())
        q_vecs = torch.cat(q_vecs, dim=0) # (M,H)  M: queries, H: hidden size
        
        pids = [doc['id'] for doc in docs]
        docs = [doc['content'] for doc in docs]
        doc_vecs = []
        for s in range(0, len(docs), self.batch_size):
            e = s + self.batch_size
            batch_texts = [row for row in docs[s:e]]
            d_tok = self.tok(batch_texts, padding=True, truncation=True,
                             max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**d_tok)
            d_vec = mean_pooling(outputs[0], d_tok['attention_mask'])
            doc_vecs.append(d_vec.cpu())
        doc_vecs = torch.cat(doc_vecs, dim=0) # (N,H)

        scores = torch.mm(doc_vecs, q_vecs.t())  # (N,M)  N: docs, M: queries
        scores_np = scores.numpy()
        num_queries = q_vecs.shape[0]

        ranked_results = []
        for q_idx in range(num_queries):
            query_scores = scores_np[:, q_idx]  # scores for all docs w.r.t. this query
            topk_idx = query_scores.argsort()[::-1]  # indices in descending order
            ranked = [{'id': pids[i], 'score':float(query_scores[i])} for i in topk_idx]
            ranked_results.append(ranked)
        return ranked_results
    
    def _modernbert_rank(self, queries, docs):
        """
        docs : List[content]
        returns : List['id':pid, 'score':score)]
        """

        num_queries = len(queries)
        with torch.no_grad():
            q_vecs = self.model.encode(
                queries,
                batch_size=self.batch_size,
                is_query=True, # Encoding queries
                show_progress_bar=False,
            )

        pids = [doc['id'] for doc in docs]
        docs = [doc['content'] for doc in docs]
        with torch.no_grad():
            doc_vecs = self.model.encode(
                docs,
                batch_size=self.batch_size,
                is_query=False, # Encoding docs
                show_progress_bar=False,
            )
        
        doc_vecs = [doc_vecs.copy() for _ in range(num_queries)] # shape: (n_queries, n_docs, 128)
        pids = [list(pids).copy() for _ in range(num_queries)] # shape: (n_queries, n_docs)

        ranked_docs = self.rank.rerank(
            documents_ids=pids,
            queries_embeddings=q_vecs,
            documents_embeddings=doc_vecs,
        ) # shape: (n_queries, n_docs)

        return ranked_results

    def _bge_rank(self, queries, docs):
        ranked_results = []
        for query in queries:
            pairs = []
            pids = []
            for doc in docs:
                pids.append(doc['id'])
                content = doc['content']
                pairs.append([query, content])

            scores = []
            for s in range(0, len(pairs), self.batch_size):
                e = s + self.batch_size
                batch_pairs = [row for row in pairs[s:e]]
                with torch.no_grad():
                    inputs = self.tok(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    inputs = {k:v.to(self.device) for k,v in inputs.items()}
                    batch_scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
                    scores.extend(batch_scores.tolist())

            results = []
            assert len(pids) == len(scores)
            for pid, score in zip(pids, scores):
                results.append({'id':pid, 'score':score})
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            ranked_results.append(results)
        return ranked_results

    def compute_scores(self, queries, docs):
        queries = [query['content'] for query in queries]
        if 'tct' in self.args.reranker or "modernbert" in self.args.reranker:
            scores_list = self._tct_rank(queries, docs)
        elif 'contriever' in self.args.reranker:
            scores_list = self._contriever_rank(queries, docs)
        elif 'bge' in self.args.reranker:
            scores_list = self._bge_rank(queries, docs)
        elif 'modernbert' in self.args.reranker:
            scores_list = self._modernbert_rank(queries, docs)
        else:
            raise NotImplementedError

        results = []
        for ranked_docs in scores_list:
            ranked_docs = sorted(ranked_docs, key=lambda x: x['score'], reverse=True)
            results.append(ranked_docs)
        return results

    def get_neighbours(self, queries, docs, excluded_ids, k=16):
        # Compute neighbours from graph
        if self.graph is not None:
            did2score = defaultdict(float)
            for query in queries:
                docno = query['id']
                neighbours, scores = self.graph.neighbours(docno, weights=True)
                # for did, score in zip(neighbours, scores):
                #     did2score[did] += score
                neighbours_scores = [(did, score) for did, score in zip(neighbours, scores) if did not in excluded_ids]
                neighbours_scores = neighbours_scores[:k]
                for did, score in neighbours_scores:
                    did2score[did] += score

        # Compute neighbours within top-k
        else:
            results = self.compute_scores(queries=queries, docs=docs)
            did2score = defaultdict(float)
            for ranked_docs in results:
                for doc in ranked_docs:
                    idx = doc['id']
                    score = doc['score']
                    did2score[idx] += score

        cur_doc_dids = set([query['id'] for query in queries])
        ranked_dids = sorted(did2score.items(), key=lambda x: x[1], reverse=True)
        ranked_dids = [idx for idx, score in ranked_dids if idx not in cur_doc_dids]
        ranked_docs = [self.did2doc[did] for did in ranked_dids]
        return ranked_docs


if __name__ == "__main__": # example usage
    args = SimpleNamespace(bsize=100, reranker='bm25')
    doc_ranker = InMemoryDocumentRanker(args)