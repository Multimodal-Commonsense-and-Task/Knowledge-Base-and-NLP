import re
import os
import time
import csv
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
from ftfy import fix_text
from copy import deepcopy
from collections import defaultdict
from typing import Union, Tuple, Dict, List
from types import SimpleNamespace
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler

from scripts.llm import Base_LLM
from scripts.dataset import ListwiseRerankDataset, collate_fn

from utils.utils import json_load, json_dump, pickle_load, pickle_dump, calculate_retrieval_metrics
from utils.prompts import get_rerank_system_prompt, get_rerank_user_prompt
from utils.ranking_utils import InMemoryDocumentRanker


class Reranker:
    def __init__(self, model_name_or_path, max_qlen=1024, max_dlen=1024, 
                batch_size=8, window_size=20, prompt_mode='reasoning', d2d='',
                **kwargs):
        self.kwargs = kwargs
        self.max_qlen = max_qlen
        self.max_dlen = max_dlen
        self.batch_size = batch_size
        self.window_size = window_size
        self.prompt_mode = prompt_mode
        self.model_name_or_path = model_name_or_path
        self.d2d = d2d  # Document-to-Document reranking mode

        self.reuse = self.kwargs.get("reuse", False)
        self.return_trace = self.kwargs.get("return_trace", False)
        self.rank_end = self.kwargs.get("top_k_docs", 100)
        self.peft_model_name = self.kwargs.get("peft_model_name")
        self.output_path = self.kwargs.get("output_path", "search_cache_tmp.json")

        self.vllm = self.kwargs.get("vllm", False)
        self.snowflake = self.kwargs.get("snowflake", False)
        self.debug = self.kwargs.get("debug", False)

        self.qa_path = self.kwargs.get("qa_path", "")
        self.dataset_name = os.path.basename(self.qa_path).split(".")[0].lower()

        self.bt_model = {}
        # bt_model_path = f"../cache/bright/bm25/repeat2/search_cache_bm25_{self.dataset_name}_reasoning_and_ranking_qwen2.5-7b-instruct_512_v8_prefix_tevatron_72k_ndcg0.1_deepseek-r1_checkpoint-4500_contriever_td2d_freq2_btiter9_bt_model.pickle"
        # self.bt_model = pickle_load(bt_model_path)
        self.bt_iter = self.kwargs.get("bt_iter", 5)
        print(f"#> bt_model: {len(self.bt_model)}")
        self.alpha = self.kwargs.get("alpha", 1.0)
        self.beta = self.kwargs.get("beta", 1.0)
        

        # For saving tmp file
        self.ranked_cache = self.kwargs.get("ranked_cache", {})

        self.qa_path = self.kwargs.get("qa_path", "")
        self.dataset_name = os.path.basename(self.qa_path).split(".")[0].lower()
        
        self.cached_trace = self.kwargs.get("cached_trace", "")
        if self.cached_trace != "" and os.path.exists(self.cached_trace):
            print(f"#> Using cached trace from {self.cached_trace}")
            self.query2trace = json_load(self.cached_trace)
            print(f"#> Load query2trace: {len(self.cached_trace)} for {self.dataset_name}")
        else:
            self.query2trace = None

        self.query2excluded_ids = {}
        self.query2gold_ids = {}
        qa_list = json_load(self.qa_path)
        for qa in qa_list:
            if 'query' in qa:
                query = qa['query']
            else:
                query = qa['Question']
            excluded_ids = qa.get('excluded_ids', set())
            gold_ids = qa.get('gold_ids', set())
            self.query2excluded_ids[query] = set(excluded_ids)
            self.query2gold_ids[query] = set(gold_ids)
        print(f"#> Load query2excluded_ids: {len(self.query2excluded_ids)} for {self.dataset_name}")
        print(f"#> Load query2gold_ids: {len(self.query2gold_ids)} for {self.dataset_name}")

        self.frequency = 0
        if self.d2d:
            print(f"#> Using D2D mode: {self.d2d}")
            assert "freq" in self.d2d, "Frequency must be specified for GAR mode"

            # frequency: how often the GAR is executed
            self.d2d_model_name, self.d2d_strategy, self.frequency = self.d2d.split("_")
            self.frequency = int(self.frequency[4:])
            self.graph_path = self.kwargs.get("graph_path", None)
            self.corpus_path = self.kwargs.get("corpus_path", None)

            self.d2d_ranker = InMemoryDocumentRanker(
                args=SimpleNamespace(
                    reranker=self.d2d_model_name,
                    batch_size=64,
                    seed=42,
                    graph_path=self.graph_path, 
                    corpus_path=self.corpus_path, 
                )
            )

        self.analyze_dist = {}
        self.seen_doc_ids = {}

        self.model = Base_LLM(
            self.model_name_or_path, 
            **kwargs, 
        )
        if self.snowflake:
            if os.path.exists('/data/models'):
                tokenizer_path = '/data/models/Llama-3.3-70B-Instruct'
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True, use_fast=True)

    def save_tmp_results(self, queries, documents, results):
        N = len(results)
        for query, ranked_docs in zip(queries, results):
            if len(ranked_docs) == 0:
                continue
            self.ranked_cache[query] = ranked_docs

        output_path = self.output_path[:-5] + "_tmp.json"
        json_dump(output_path, self.ranked_cache)
        print(f"#> Saving tmp {output_path} done")

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])

    def build_prompt(self, query: str, docs: List[str], trace: List[List[str]]) -> str:
        messages = list()
        messages.append({"role": "system", "content": get_rerank_system_prompt(self.prompt_mode)})

        # Rearank
        if self.prompt_mode == 'rearank':
            rearank_prompts = get_rerank_user_prompt(
                query, 
                docs, 
                self.prompt_mode, 
                n_docs=len(docs), 
            )
            assert isinstance(rearank_prompts, list)
            messages.extend(rearank_prompts)

        # Relevance & Reasoning and Ranking
        else:
            documents = ""
            for idx, doc in enumerate(docs):
                if 'content' in doc:
                    contents = doc['content'].strip()
                elif 'contents' in doc:
                    contents = doc['contents'].strip()
                else:
                    assert False
                contents = self.truncate(contents, self.max_dlen)
                documents += f"[{idx+1}]: {contents}\n"
            query = self.truncate(query, self.max_qlen)
            query = query.strip()
            documents = documents.strip()
            
            user_prompt = fix_text(get_rerank_user_prompt(
                query, 
                documents, 
                self.prompt_mode, 
                n_docs=len(docs), 
            ))
            messages.append({"role": "user", "content": user_prompt})

        if self.snowflake:
            prompt = messages
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if self.reuse and trace is not None:
                steps = "[Reasoning Trace]\n"
                if self.query2trace and query in self.query2trace and self.query2trace[query] is not None:
                    step_to_insert = self.query2trace[query]["trace"]
                    insert_idx = self.query2trace[query]["modified_step"] - 1
                    for idx, step in enumerate(trace):
                        if idx == insert_idx:
                            steps += f"Step {idx+1}: {step_to_insert}\n"
                        elif idx < insert_idx:
                            steps += f"Step {idx+1}: {step}\n"
                        else:
                            steps += f"Step {idx+2}: {step}\n"
                else:
                    for idx, step in enumerate(trace):
                        # if idx == len(trace) - 1:
                        #     step = " ".join(step.split()[:-2])
                        steps += f"Step {idx+1}: {step}\n"
                steps = fix_text(steps).strip()
                prompt += steps + "\n\n[Document Ranking]\n"
            elif self.query2trace:
                steps = "[Reasoning Trace]\n"
                if query in self.query2trace and self.query2trace[query] is not None:
                    steps += f"Step [{self.query2trace[query]['modified_step']}]: {self.query2trace[query]['trace']}\n\n"
        return prompt

    def transform_text_to_ids(self, response):
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        ids = new_response.strip().split()
        indices = []
        for x in ids:
            try:
                x = int(x) - 1
                if x < self.window_size:
                    indices.append(x)
            except:
                pass
        for i in range(self.window_size):
            if i not in indices:
                indices.append(i)
        return indices

    def parse_ranking(self, output: str):
        try:
            output = output.strip()
            if self.prompt_mode == 'relevance':
                indices = self.transform_text_to_ids(output)

            elif self.prompt_mode == 'rearank':
                if "<answer>" in output:
                    content_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
                    response = content_match.group(1).strip() if content_match else output.strip()
                # process for Qwen3 with think tags but no answer tags
                if "<think>" in output:
                    response = response.split("</think>")[-1]
                indices = self.transform_text_to_ids(response)

            elif self.prompt_mode == 'reasoning_and_ranking':
                trace = []
                if self.return_trace:
                    reasoning_section = output.split("[Reasoning Trace]")[-1].split("[Document Ranking]")[0]
                    trace = re.findall(r"Step \d+:\s*(.+)", reasoning_section)
                    trace = [step.strip() for step in trace]
                if len(trace) == 0:
                    trace = None
                response = output.split("[Document Ranking]")[-1]
                indices = self.transform_text_to_ids(response)

            if self.return_trace:
                return indices[:self.window_size//2], trace
            return indices[:self.window_size//2]

        except Exception:
            print(f"#> Error output sample: {output}")
            if self.return_trace:
                return [], None
            return []
    
    def rerank(self, queries: List[str], documents: List[List[str]]) -> List[List[int]]:
        """
        Args:
            queries: List of queries.
            documents: List of list of documents per query.

        Returns:
            List of list of indices
        """
        assert len(queries) == len(documents), f"{len(queries)}; {len(documents)}"
        dataset = ListwiseRerankDataset(queries, documents)

        if self.vllm or self.snowflake:
            sampler = None
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=collate_fn)

        results = []
        total_start_time = time.time()
        print(f"#> Rerank start")

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch_items = [
                {'query': q, 'docs': d}
                for q, d in zip(batch['query'], batch['docs'])
            ]
            processed_items = self.sliding_windows_batch(
                batch_idx=batch_idx, 
                batch_items=batch_items, 
                rank_start=0, 
                rank_end=min(len(batch_items[0]['docs']), self.rank_end), 
            )
            results.extend(processed_items)

            if batch_idx % 10 == 0:
                self.save_tmp_results(queries, documents, results)
            
        total_end_time = time.time()
        print(f"#> Total execution time: {total_end_time - total_start_time:.2f}s")

        analyze_path = self.output_path[:-5] + "_analyze_dist.json"
        json_dump(analyze_path, self.analyze_dist)
        print(f"#> Saving {analyze_path} done.")
        seen_path = self.output_path[:-5] + "_seen_doc_ids.json"
        json_dump(seen_path, self.seen_doc_ids)
        print(f"#> Saving {seen_path} done.")
        bt_model_path = self.output_path[:-5] + "_bt_model.pickle"
        pickle_dump(bt_model_path, self.bt_model)
        print(f"#> Saving {bt_model_path} done.")

        if self.vllm or self.snowflake:
            return results
        else:
            all_results = [None for _ in range(world_size)]
            dist.all_gather_object(all_results, results)
            if rank == 0:
                merged = [r for rank_results in all_results for r in rank_results]
                return merged
            else:
                return []

    def sliding_windows_batch(self, batch_idx, batch_items, rank_start, rank_end):
        """Process multiple items with sliding windows using batched inference."""
        items = [deepcopy(item) for item in batch_items]
        num_items = len(items)
        
        doc_ptrs = [item['docs'][:rank_end] for item in batch_items]
        
        done_flags = [False for _ in range(num_items)]
        # if self.cached_trace:
        #     reuse_trace = [self.query2trace.get(item['query']) for item in items]
        #     if isinstance(reuse_trace[0], list):
        #         print(f"#> Loaded cached trace for {sum([1 for t in reuse_trace if t is not None])}/{len(reuse_trace)} items")
        #     elif isinstance(reuse_trace[0], dict):
        #         reuse_trace = [item['trace'] if item is not None and 'trace' in item else None for item in reuse_trace]
        # else:
        reuse_trace = [None for _ in range(num_items)]
        selected_docs = [[] for _ in range(num_items)]
        final_rankings = [[] for _ in range(num_items)]
        seen_doc_ids = [set() for _ in range(num_items)]
        prev_window = [[] for _ in range(num_items)]
        
        iteration = 1
        while not all(done_flags):
            print(f"#> Sliding window Progress: iteration {iteration}\tBatch Progress: {sum(done_flags)}/{len(done_flags)}")

            prompts = []
            prompt_info = [] # (item_idx, current_doc_ids)

            for i in range(num_items):
                if done_flags[i]:
                    continue

                item = items[i]
                query = item['query']
                docs = item['docs']

                prev_relevant = selected_docs[i]
                ptr = doc_ptrs[i]
                new_docs = ptr[:self.window_size - len(prev_relevant)]
                
                if not new_docs:
                    done_flags[i] = True
                    continue
                
                if self.d2d and prev_relevant:
                    if iteration >= 6:
                        if i == 0:
                            print(f"#> Entered GAR at iteration {iteration}")
                        new_docs = self._update_new_docs(
                            cur_docs=prev_relevant, 
                            pool_docs=[doc for doc in ptr if doc['id'] not in seen_doc_ids[i]], 
                            seen_doc_ids=seen_doc_ids[i], 
                            max_num_docs=self.window_size - len(prev_relevant), 
                            trace=reuse_trace[i], 
                            new_docs=new_docs, 
                            original_query=query, 
                            prev_window=prev_window[i], 
                        )

                current_docs = prev_relevant + new_docs
                prompt = self.build_prompt(query, current_docs, trace=reuse_trace[i])

                current_doc_ids = set([doc['id'] for doc in current_docs])
                seen_doc_ids[i].update(current_doc_ids)

                prompts.append(prompt)
                prompt_info.append((i, query, current_docs, current_doc_ids))
            
            if not prompts:
                break

            outputs = self.model.llm_batch_inference(prompts, logprobs=1, return_n_tokens=True)

            for output, prompt, (item_idx, query, current_docs, current_doc_ids) in zip(outputs, prompts, prompt_info):
                output_text = output['text']
                n_tokens = output['n_tokens']
                logprobs = output['logprobs']

                ranked_local = self.parse_ranking(output_text)
                if self.return_trace:
                    ranked_local, trace = ranked_local
                    if trace is not None:
                        reuse_trace[item_idx] = trace

                if self.debug:
                    print(f"#> [DEBUG] Output: {output_text}; {ranked_local}")
                    assert False

                if iteration <= self.bt_iter:
                    if query not in self.bt_model:
                        self.bt_model[query] = {}
                    dids = [current_docs[i]['id'] for i in ranked_local]
                    remain_dids = [current_docs[i]['id'] for i in range(len(current_docs)) if i not in ranked_local]
                    for i in range(len(dids)):
                        for j in range(i+1, len(dids)):
                            self.bt_model[query][(dids[i],dids[j])] = self.bt_model[query].get((dids[i],dids[j]), 0) + 1
                        for did in remain_dids:
                            self.bt_model[query][(dids[i],did)] = self.bt_model[query].get((dids[i],did), 0) + 1

                # map local indices to global doc ids
                relevant_global_docs = [current_docs[i] for i in ranked_local if i < len(current_docs)]
                prev_window[item_idx] = current_docs.copy()

                # update
                selected_docs[item_idx] = relevant_global_docs
                doc_ptrs[item_idx] = [doc for doc in doc_ptrs[item_idx] if doc['id'] not in seen_doc_ids[item_idx]]

                # terminate condition
                if iteration == 9 or len(doc_ptrs[item_idx]) == 0:
                    final_rankings[item_idx] = relevant_global_docs
                    done_flags[item_idx] = True
                
                # Saving output info for analysis
                if 'output' not in items[item_idx]:
                    items[item_idx]['output'] = []
                if 'ndcg' not in items[item_idx]:
                    items[item_idx]['ndcg'] = []
                if 'completion_tokens' not in items[item_idx]:
                    items[item_idx]['completion_tokens'] = 0

                query = items[item_idx]['query']
                gold_ids = self.query2gold_ids[query]
                qrels = {"q1":{did:1 for did in gold_ids}}
                preds = {"q1":{doc['id']:1/(rank+1) for rank, doc in enumerate(selected_docs[item_idx])}}
                eval_results = calculate_retrieval_metrics(preds, qrels, k_values=[10])
                ndcg = eval_results["NDCG@10"]
                items[item_idx]['ndcg'].append(ndcg)

                items[item_idx]['output'].append(output_text)
                items[item_idx]['completion_tokens'] += n_tokens

                    
            if self.debug:
                print(f"iteration {iteration}, # selected_docs: {selected_docs}, # doc_ptrs: {doc_ptrs}")
            iteration += 1

        for item_idx, item in enumerate(items):
            query = item['query']
            output = item['output']
            ndcg = item['ndcg']
            completion_tokens = item['completion_tokens']
            self.seen_doc_ids[query] = list(seen_doc_ids[item_idx])
            
            # For Analysis
            if query not in self.analyze_dist:
                self.analyze_dist[query] = {}

            self.analyze_dist[query]['ndcg'] = ndcg
            self.analyze_dist[query]['output'] = output
            self.analyze_dist[query]['completion_tokens'] = completion_tokens

        return final_rankings

    def _update_new_docs(self, cur_docs: List[str], pool_docs: List[str], seen_doc_ids: List[str]=None, max_num_docs: int=10, trace: List[str]=None, new_docs=None, original_query=None, prev_window=None) -> List[int]:
        """Update new documents for GAR (Document-to-Document reranking)."""
        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        if 'contents' in cur_docs[0]:
            cur_docs = [{'id': doc['id'], 'content':doc['contents']} for idx, doc in cur_docs]
            pool_docs = [{'id': doc['id'], 'content':doc['contents']} for idx, doc in pool_docs]
            new_docs = [{'id': doc['id'], 'content':doc['contents']} for idx, doc in new_docs]
            prev_window = [{'id': doc['id'], 'content':doc['contents']} for idx, doc in prev_window]

        if self.d2d_strategy == 'd2d':
            neighbors = self.d2d_ranker.get_neighbours(
                queries=cur_docs, 
                docs=pool_docs, 
                excluded_ids=self.query2excluded_ids[original_query],
                k=16,
            )
            return neighbors[:max_num_docs]

        elif self.d2d_strategy == 't2d':
            queries = [{'id': -1, 'content': step} for step in trace]
        elif self.d2d_strategy == 'td2d':
            if trace is None:
                return new_docs, 0

            # Identify target step (i_star)
            steps = [{'id': idx, 'content': step} for idx, step in enumerate(trace)]
            docs_results = self.d2d_ranker.compute_scores(
                queries=steps, 
                docs=prev_window, 
            )

            s_rel, s_contri, s_consist = [], [], []
            cur_doc_ids = set([doc['id'] for doc in cur_docs])
            full_ranking = {"q1":{doc['id']:len(cur_docs)-rank for rank, doc in enumerate(cur_docs)}}
            for i, (step, ranked_docs) in enumerate(zip(steps, docs_results)):
                assert i == step['id']

                # Compute support scores
                rel = [sigmoid(doc['score']) for doc in ranked_docs if doc['id'] in cur_doc_ids]
                s_rel.append(round(float(np.mean(rel)),5))

                # Compute contribution scores
                ranked_docs = ranked_docs[:len(cur_docs)]
                stepwise_ranking = {"q1":{doc['id']:len(ranked_docs)-rank for rank, doc in enumerate(ranked_docs)}}
                ndcg = calculate_retrieval_metrics(stepwise_ranking, full_ranking)["NDCG@10"]
                s_contri.append(ndcg)
                
                # Compute consistency scores
                dids = [doc['id'] for doc in ranked_docs]
                consist = []
                for i in range(len(dids)):
                    for j in range(i+1, len(dids)):
                        s_i = self.bt_model[original_query].get((dids[i],dids[j]),0)
                        s_j = self.bt_model[original_query].get((dids[j],dids[i]),0)
                        e_i = np.exp(s_i)
                        e_j = np.exp(s_j)
                        consist.append(e_i / (e_i + e_j))
                s_consist.append(round(float(np.mean(consist)),5))

            s_final = []
            for rel, contri, consist in zip(s_rel, s_contri, s_consist):
                s = rel - self.alpha * contri - self.beta * consist
                s_final.append(s)
            i_star = int(np.argmin(s_final))
            s_i_star = s_final[i_star]

            # Use the d2d ranker to get new documents
            neighbors = self.d2d_ranker.get_neighbours(
                queries=cur_docs, 
                docs=pool_docs, 
                excluded_ids=self.query2excluded_ids[original_query],
                k=16,
            )
            # Padding with original ones
            new_doc_ids = set([doc['id'] for doc in new_docs])
            neighbors = new_docs + [doc for doc in neighbors if doc['id'] not in new_doc_ids]

            # Remove cur documents
            neighbors = [doc for doc in neighbors if doc['id'] not in cur_doc_ids]
            
            # Select best supporting documents for all steps
            did2doc = {doc['id']:doc for doc in neighbors}
            supporting_docs = self.d2d_ranker.compute_scores(
                # queries=[steps[i_star]], 
                queries=steps, 
                docs=neighbors, 
            )

            # For Analysis
            recall_per_step = []
            gold_ids = self.query2gold_ids[original_query]
            remain_gold_ids = set([did for did in gold_ids if did not in cur_doc_ids])
            for docs in supporting_docs:
                docs = docs[:max_num_docs]
                dids = set([doc['id'] for doc in docs])
                recall = len(gold_ids & dids) / len(dids)
                recall_per_step.append(recall)

            bm25_recall = len(gold_ids & new_doc_ids) / len(new_doc_ids)
            n_remain_gold_ids = len(remain_gold_ids)

            if original_query not in self.analyze_dist:
                self.analyze_dist[original_query] = {}
                self.analyze_dist[original_query]["i_star"] = []
                self.analyze_dist[original_query]["s_i_star"] = []
                self.analyze_dist[original_query]["s_rel"] = []
                self.analyze_dist[original_query]["s_contri"] = []
                self.analyze_dist[original_query]["s_consist"] = []
                self.analyze_dist[original_query]["s_final"] = []
                self.analyze_dist[original_query]["n_remain_gold_ids"] = []
                self.analyze_dist[original_query]["recall_per_step"] = []
                self.analyze_dist[original_query]["bm25_recall"] = []

            self.analyze_dist[original_query]["i_star"].append(i_star)
            self.analyze_dist[original_query]["s_i_star"].append(s_i_star)
            self.analyze_dist[original_query]["s_rel"].append(s_rel)
            self.analyze_dist[original_query]["s_contri"].append(s_contri)
            self.analyze_dist[original_query]["s_consist"].append(s_consist)
            self.analyze_dist[original_query]["s_final"].append(s_final)

            self.analyze_dist[original_query]["n_remain_gold_ids"].append(n_remain_gold_ids)
            self.analyze_dist[original_query]["recall_per_step"].append(recall_per_step)
            self.analyze_dist[original_query]["bm25_recall"].append(bm25_recall)


            # Collecting documents supporting i_star
            best_supporting_docs = supporting_docs[i_star][:max_num_docs]
            # best_supporting_docs = supporting_docs[0][:max_num_docs]
            next_window = [did2doc[doc['id']] for doc in best_supporting_docs]
            return next_window

