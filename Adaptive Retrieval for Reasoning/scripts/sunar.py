import re
import os
import time
import csv
import json
import torch
import random
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from ftfy import fix_text
from copy import deepcopy
from collections import defaultdict, Counter
from typing import Union, Tuple, Dict, List
from types import SimpleNamespace
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler

from scripts.llm import Base_LLM
from scripts.dataset import ListwiseRerankDataset, collate_fn
from utils.utils import json_load, json_dump, normalize_answer_qa
from utils.prompts import get_rerank_system_prompt, get_rerank_user_prompt
from utils.ranking_utils import InMemoryDocumentRanker


class SUNAR:
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
        self.d2d = d2d # Document-to-Document reranking mode

        self.reuse = self.kwargs.get("reuse", False)
        self.return_trace = self.kwargs.get("return_trace", False)
        self.rank_end = self.kwargs.get("top_k_docs", 100)
        self.max_sunar_tokens = self.kwargs.get("max_sunar_tokens", 256)
        self.peft_model_name = self.kwargs.get("peft_model_name")
        self.output_path = self.kwargs.get("output_path", "search_cache_tmp.json")
        self.tmp_results = self.kwargs.get("tmp_results", [])
        self.gen_final_answer = self.kwargs.get("gen_final_answer", False)

        self.vllm = self.kwargs.get("vllm", False)
        self.snowflake = self.kwargs.get("snowflake", False)
        self.debug = self.kwargs.get("debug", False)
        
        self.query2excluded_ids = None
        self.qa_path = self.kwargs.get("qa_path", "")
        dataset_name = os.path.basename(self.qa_path).split(".")[0].lower()
        if dataset_name in ["leetcode", "aops", "theoremqa_questions"]:
            self.query2excluded_ids = {}
            qa_list = json_load(self.qa_path)
            for qa in qa_list:
                query = qa['query']
                excluded_ids = qa['excluded_ids']
                self.query2excluded_ids[query] = set(excluded_ids)
            print(f"#> Load query2excluded_ids: {len(self.query2excluded_ids)} for {dataset_name}")

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
                    batch_size=100,
                    seed=42,
                    graph_path=self.graph_path, 
                    corpus_path=self.corpus_path, 
                )
            )

        self.model = Base_LLM(
            self.model_name_or_path, 
            **kwargs, 
        )
        if self.prompt_mode == 'pointwise':
            from sentence_transformers import CrossEncoder
            self.device = torch.cuda.current_device()
            self.cross_encoder = CrossEncoder("nreimers/mmarco-mMiniLMv2-L12-H384-v1", device=self.device, trust_remote_code=True)
            
        if self.snowflake:
            if os.path.exists('/data/models'):
                tokenizer_path = '/data/models/Llama-3.3-70B-Instruct'
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True, use_fast=True)

    def save_tmp_results(self, results):
        self.tmp_results.extend(results)
        output_path = self.output_path[:-5] + "_tmp.json"
        json_dump(output_path, self.tmp_results)
        print(f"#> Saving tmp {output_path} done")

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])

    def build_prompt(self, query: str, docs: List[str], trace: List[List[str]]) -> str:
        if self.prompt_mode == 'pointwise':
            if 'content' in docs[0]:
                pairs = [[query, doc['content']] for doc in docs]
            elif 'contents' in docs[0]:
                pairs = [[query, doc['contents']] for doc in docs]
            pairs = [[self.truncate(query, self.max_qlen), self.truncate(doc, self.max_dlen)] for query, doc in pairs]
            return pairs

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
                for idx, step in enumerate(trace):
                    steps += f"Step {idx+1}: {step}\n"
                steps = fix_text(steps).strip()
                prompt += steps + "\n[Document Ranking]\n"
        return prompt

    def parse_ranking(self, output: str):
        if self.prompt_mode == 'pointwise':
            scores = [(score, idx) for idx, score in enumerate(output)]
            scores = sorted(scores, reverse=True)
            indices = [idx for score, idx in scores]
            return indices[:self.window_size//2]

        try:
            output = output.strip()
            if self.prompt_mode == 'relevance':
                ids = output.split(">")

            elif self.prompt_mode == 'rearank' or self.prompt_mode == 'reasonrank':
                if "<answer>" in output:
                    content_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
                    response = content_match.group(1).strip() if content_match else output.strip()
                # process for Qwen3 with think tags but no answer tags
                if "<think>" in output:
                    response = response.split("</think>")[-1]
                new_response = ""
                for c in response:
                    if not c.isdigit():
                        new_response += " "
                    else:
                        new_response += c
                ids = new_response.strip().split()

            elif self.prompt_mode == 'reasoning_and_ranking':
                ids = output.split("[Document Ranking]")[-1]
                ids = ids.split(">")

                trace = None
                if self.return_trace and "[Reasoning Trace]" in output:
                    reasoning_match = re.search(r"\[Reasoning Trace\](.*?)\n\[Document Ranking\]", output, re.DOTALL)
                    reasoning_section = reasoning_match.group(1)
                    trace = re.findall(r"Step \d+:\s*(.+)", reasoning_section)

            indices = []
            for idx in ids:
                idx = idx.strip()
                idx = idx.replace("[","").replace("]","")
                if idx.isdigit():
                    idx = int(idx) - 1
                    if idx not in indices:
                        indices.append(idx)

            if self.return_trace:
                return indices[:self.window_size//2], trace
            return indices[:self.window_size//2]

        except Exception:
            print(f"#> Error output sample: {output}")
            if self.return_trace:
                return [], None
            return []

    def rerank(self, queries: List[str], documents: List[List[str]], original_queries: List[str]=None) -> List[List[int]]:
        """
        Args:
            queries: List of queries.
            documents: List of list of documents per query.

        Returns:
            List of list of indices
        """
        assert len(queries) == len(documents), f"{len(queries)}; {len(documents)}"
        results = []
        total_start_time = time.time()
        print(f"#> Rerank start")


        for batch_idx in range(0, len(queries), self.batch_size):
            batch_queries = queries[batch_idx:batch_idx+self.batch_size]
            batch_documents = documents[batch_idx:batch_idx+self.batch_size]

            if original_queries is None:
                batch_items = [
                    {'query': q, 'docs': d}
                    for q, d in zip(batch_queries, batch_documents)
                ]
            else:
                batch_original_queries = original_queries[batch_idx:batch_idx+self.batch_size]
                batch_items = [
                    {'query': q, 'docs': d, 'original_query': oq}
                    for q, d, oq in zip(batch_queries, batch_documents, batch_original_queries)
                ]

            processed_items = self.sliding_windows_batch(
                batch_idx=batch_idx, 
                batch_items=batch_items, 
                rank_start=0, 
                rank_end=min(len(batch_items[0]['docs']), self.rank_end), 
            )
            results.extend(processed_items)
            
        total_end_time = time.time()
        print(f"#> Total execution time: {total_end_time - total_start_time:.2f}s")

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
        
        # doc_ptrs = [list(range(rank_start, min(rank_end, len(item['docs'])))) for item in batch_items]
        doc_ptrs = [item['docs'][:rank_end] for item in batch_items]
        
        # Initialize selected_docs with the first window of documents
        done_flags = [False for _ in range(num_items)]
        reuse_trace = [None for _ in range(num_items)]
        selected_docs = [[] for _ in range(num_items)]
        final_rankings = [[] for _ in range(num_items)]
        seen_doc_ids = [set() for _ in range(num_items)]
        
        iteration = 1
        slidegar_done = [False for _ in range(num_items)]  # Track if GAR has been done for each item
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
                    if (not slidegar_done[i]) and (len(ptr) <= self.window_size - len(prev_relevant)) and self.frequency == -1:  # last window, use GAR
                        if i == 0:
                            print(f"#> Entered GAR at LAST iteration {iteration}")
                        slidegar_done[i] = True
                        new_docs = self._update_new_docs(
                            cur_docs=[docs[j] for j in prev_relevant],
                            cur_doc_ids=prev_relevant,
                            pool_docs=[doc for j, doc in enumerate(docs) if j not in ptr and j not in prev_relevant],
                            pool_doc_ids=[j for j in range(len(docs)) if j not in ptr and j not in prev_relevant],
                            max_num_docs=self.window_size - len(prev_relevant),
                            trace=reuse_trace[i],
                        )
                    elif self.frequency != -1 and iteration % self.frequency == 0: # GAR condition
                        if i == 0:
                            print(f"#> Entered GAR at iteration {iteration}")
                        new_docs = self._update_new_docs(
                            cur_docs=prev_relevant,
                            pool_docs=[doc for doc in ptr if doc['id'] not in seen_doc_ids[i]],
                            max_num_docs=self.window_size - len(prev_relevant),
                            trace=reuse_trace[i],
                            new_docs=new_docs,
                            original_query=query if 'original_query' not in item else item['original_query'], 
                        )

                current_docs = prev_relevant + new_docs
                prompt = self.build_prompt(query, current_docs, reuse_trace[i])

                current_doc_ids = set([doc['id'] for doc in current_docs])
                seen_doc_ids[i].update(current_doc_ids)

                prompts.append(prompt)
                prompt_info.append((i, current_docs, current_doc_ids))
            
            if not prompts:
                break

            if self.prompt_mode == 'pointwise':
                outputs = [self.cross_encoder.predict(pairs, show_progress_bar=False) for pairs in prompts]
            else:
                outputs = self.model.llm_batch_inference(prompts)

            for output, prompt, (item_idx, current_docs, current_doc_ids) in zip(outputs, prompts, prompt_info):
                ranked_local = self.parse_ranking(output)
                if self.return_trace:
                    ranked_local, trace = ranked_local
                    if trace is not None:
                        reuse_trace[item_idx] = trace

                if len(ranked_local) == 0:
                    final_rankings[item_idx] = []
                    done_flags[item_idx] = True
                    continue

                # map local indices to global doc ids
                relevant_global_docs = [current_docs[i] for i in ranked_local if i < len(current_docs)]

                # update
                selected_docs[item_idx] = relevant_global_docs
                doc_ptrs[item_idx] = [doc for doc in doc_ptrs[item_idx] if doc['id'] not in seen_doc_ids[item_idx]]

                # terminate condition
                if iteration == 9 or len(doc_ptrs[item_idx]) == 0:
                    final_rankings[item_idx] = relevant_global_docs
                    done_flags[item_idx] = True
                    
            iteration += 1

        return final_rankings

    def _update_new_docs(self, cur_docs: List[str], pool_docs: List[str], max_num_docs: int=10, trace: List[str]=None, new_docs=None, original_query=None) -> List[int]:
        """Update new documents for GAR (Document-to-Document reranking)."""
        if 'contents' in cur_docs[0]:
            cur_docs = [{'id': doc['id'], 'content':doc['contents']} for doc in cur_docs]
            pool_docs = [{'id': doc['id'], 'content':doc['contents']} for doc in pool_docs]

        if self.d2d_strategy == 'd2d':
            queries = cur_docs
        else:
            raise NotImplementedError
        
        # Use the d2d ranker to get new documents
        ranked_docs = self.d2d_ranker.get_neighbours(
            queries=queries, 
            docs=pool_docs, 
        )
        if self.query2excluded_ids is not None and original_query is not None:
            ranked_docs = [doc for doc in ranked_docs if doc['id'] not in self.query2excluded_ids[original_query]]
        if len(ranked_docs) == 0:
            ranked_doc_ids = set([doc['id'] for doc in ranked_docs])
            ranked_docs += [doc for doc in pool_docs[:max_num_docs] if doc['id'] not in ranked_doc_ids]

        return ranked_docs[:max_num_docs]
        
    def inference(self, qa_list, cache):
        # Initialize for saving history
        results = []
        for i in tqdm(range(0, len(qa_list), self.batch_size)):
            print(f"#> {i}-th batch")
            batch_items = qa_list[i:i+self.batch_size]

            processed_items = self.generate_topk_docs(batch_items)
            processed_items = self.rerank_topk_docs(processed_items)

            if self.gen_final_answer:
                processed_items = self.generate_final_answer(processed_items)

            for item in processed_items:
                results.append({
                    'id':item['id'], 
                    'query':item['query'], 
                    'sub_queries':item['sub_queries'], 
                    'sub_answers':item['sub_answers'], 
                    'sub_docs':item['sub_docs'], 
                    'final_docs':item['final_docs'], 
                })
                if self.gen_final_answer:
                    results[-1]['answer'] = item['answer']
                    results[-1]['pred_answer'] = item['pred_answer']
                    results[-1]['em'] = item['em']
                    results[-1]['f1'] = item['f1']
                    results[-1]['acc'] = item['acc']

        if self.gen_final_answer:
            avg_em = round(100*np.mean([item['em'] for item in results]), 1)
            avg_f1 = round(100*np.mean([item['f1'] for item in results]), 1)
            avg_acc = round(100*np.mean([item['acc'] for item in results]), 1)
            print(f"EM: {avg_em}\tF1: {avg_f1}\tACC: {avg_acc}\tTotal: {len(results)}")

        return results

    def generate_final_answer(self, items):
        batch_items = [deepcopy(item) for item in items]
        
        prompts, prompt_info = [], []
        for item_idx, item in enumerate(batch_items):
            qid = item['id']
            query = item['query']
            answers = item['answer']
            sub_queries = item['sub_queries']
            sub_answers = item['sub_answers']
            final_docs = item['final_docs']

            decomposed_path = ""
            for sub_query, sub_ans in zip(sub_queries, sub_answers):
                decomposed_path += f"Question: {sub_query}\n"
                decomposed_path += f"Intermediate Answer: {sub_ans}\n"

            prompt = self.build_sunar_ans_gen_messages(query, final_docs, decomposed_path)
            prompts.append(prompt)
            prompt_info.append((item_idx, qid, query, answers))

        outputs = self.model.llm_batch_inference(prompts)

        for pred_answer, (item_idx, qid, query, answers) in zip(outputs, prompt_info):
            if "[Final Answer]:" in pred_answer:
                pred_answer = pred_answer.split("[Final Answer]:")[-1]
            elif "answer is:" in pred_answer:
                pred_answer = pred_answer.split("answer is:")[-1]
            pred_answer = normalize_answer_qa(pred_answer)
            
            em, f1, acc = 0, 0, 0
            for ans in answers:
                ans = normalize_answer_qa(ans)
                if ans == pred_answer:
                    em = max(em, 1)
                if len(pred_answer) > 0 and pred_answer in ans:
                    acc = max(acc, 1)

                ans_tokens = ans.split()
                pred_tokens = pred_answer.split()
                common = Counter(pred_tokens) & Counter(ans_tokens)
                num_same = sum(common.values())
                if len(pred_tokens) == 0 or len(ans_tokens) == 0:
                    continue
                precision = 1.0 * num_same / len(pred_tokens)
                recall = 1.0 * num_same / len(ans_tokens)
                if precision + recall > 0:
                    f1 = max(f1, (2 * precision * recall) / (precision + recall))
            
            batch_items[item_idx]['pred_answer'] = pred_answer
            batch_items[item_idx]['em'] = em
            batch_items[item_idx]['f1'] = f1
            batch_items[item_idx]['acc'] = acc

        return batch_items

    def rerank_topk_docs(self, items):
        batch_items = [deepcopy(item) for item in items]

        queries, final_docs, inputs_info = [], [], []
        for item_idx, item in enumerate(batch_items):
            qid = item['id']
            query = item['query']
            first_stage_docs = item['docs']
            sub_docs = item['sub_docs']

            topk_docs = []
            topk_doc_ids = set()
            for docs in sub_docs:
                for doc in docs[:2]:
                    if doc['id'] in topk_doc_ids:
                        continue
                    topk_docs.append(doc)
                    topk_doc_ids.add(doc['id'])
            if len(topk_docs) == 0:
                topk_docs = first_stage_docs[:100]

            queries.append(query)
            final_docs.append(topk_docs)
            inputs_info.append((item_idx, qid, query))

        if self.prompt_mode == 'pointwise':
            outputs = []
            for query, docs in zip(queries, final_docs):
                if 'content' in docs[0]:
                    pairs = [[query, doc['content']] for doc in docs]
                elif 'contents' in docs[0]:
                    pairs = [[query, doc['contents']] for doc in docs]
                pairs = [[self.truncate(query, self.max_qlen), self.truncate(doc, self.max_dlen)] for query, doc in pairs]
                scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
                scores = [(score, idx) for idx, score in enumerate(scores)]
                scores = sorted(scores, reverse=True)
                ranked_docs = [docs[idx] for score, idx in scores]
                outputs.append(ranked_docs)
        else:
            outputs = self.rerank(queries, final_docs)

        for query, ranked_docs, (item_idx, qid, query) in zip(queries, outputs, inputs_info):
            batch_items[item_idx]['final_docs'] = ranked_docs[:10]

        return batch_items

    def generate_topk_docs(self, items):
        batch_items = [deepcopy(item) for item in items]
        for qa in batch_items:
            qa['sub_queries'] = []
            qa['sub_docs'] = []
            qa['sub_answers'] = []

        topk_docs = [qa['docs'] for qa in batch_items]
        done_flags = [False for _ in range(len(batch_items))]

        iteration = 1
        while not all(done_flags) and iteration <= 5: # Until the final answers are generated or iteration > 5
            print(f"#> SUNAR Progress: iteration {iteration}\tBatch Progress: {sum(done_flags)}/{len(done_flags)}")

            # 1. Generate sub-queries
            prompts, prompt_info = [], []
            for item_idx, qa in enumerate(batch_items):
                if done_flags[item_idx]:
                    continue
                query = qa['query']
                sub_queries = qa['sub_queries']
                sub_docs = qa['sub_docs']
                sub_answers = qa['sub_answers']

                prompt = self.build_sunar_messages(query, sub_queries=sub_queries, sub_answers=sub_answers, mode="gen_sub_query")
                prompts.append(prompt)
                prompt_info.append((item_idx, query))

            outputs = self.model.llm_batch_inference(prompts, max_tokens=self.max_sunar_tokens)
            for output, prompt, (item_idx, query) in zip(outputs, prompts, prompt_info):
                sub_query = self.extract_first_follow_up_question(output)
                if self.debug and item_idx == 0:
                    print(f"#> [DEBUG] Generate sub-queries: {sub_query if len(sub_query) else output}")
                if "[Final Answer]:" in sub_query:
                    final_ans = self.extract_final_answer(output)
                    batch_items[item_idx]['final_answer'] = final_ans
                    done_flags[item_idx] = True
                    continue
                batch_items[item_idx]['sub_queries'].append(sub_query)


            # 2. Ranking sub-docs for sub-queries
            original_queries, sub_queries, sub_topk_docs, inputs_info = [], [], [], []
            for item_idx, qa in enumerate(batch_items):
                if done_flags[item_idx]:
                    continue
                original_queries.append(qa['query'])
                sub_queries.append(qa['sub_queries'][-1])
                sub_topk_docs.append(topk_docs[item_idx])
                inputs_info.append(item_idx)
            results = self.rerank(sub_queries, sub_topk_docs, original_queries)

            for ranked_docs, item_idx in zip(results, inputs_info):
                batch_items[item_idx]['sub_docs'].append(ranked_docs)
                if self.debug and item_idx == 0:
                    print(f"#> [DEBUG] Ranking sub-docs: {len(batch_items[item_idx]['sub_docs'])}")


            # 3. Generate sub-answer
            prompts, prompt_info = [], []
            for item_idx, qa in enumerate(batch_items):
                if done_flags[item_idx]:
                    continue
                query = qa['query']
                sub_queries = qa['sub_queries']
                sub_docs = qa['sub_docs']
                sub_answers = qa['sub_answers']

                prompt = self.build_sunar_messages(query, sub_queries=sub_queries, sub_docs=sub_docs, sub_answers=sub_answers, mode="gen_sub_ans")
                prompts.append(prompt)
                prompt_info.append((item_idx, query))

            outputs = self.model.llm_batch_inference(prompts, max_tokens=self.max_sunar_tokens)
            for output, prompt, (item_idx, query) in zip(outputs, prompts, prompt_info):
                sub_answer = self.extract_first_intermediate_answer(output)
                if self.debug and item_idx == 0:
                    print(f"#> [DEBUG] Generate sub-answers: {sub_answer if len(sub_answer) else output}")
                batch_items[item_idx]['sub_answers'].append(sub_answer)

                # terminate condition
                if iteration == 10:
                    done_flags[item_idx] = True

            iteration += 1
            if self.debug:
                print("#> Iteration:", iteration)

        return batch_items

    def extract_first_follow_up_question(self, text):
        match = re.search(r"Follow up:\s*(.*)", text)
        return match.group(1).strip() if match else text.strip()

    def extract_first_intermediate_answer(self, text):
        match = re.search(r"Intermediate Answer:\s*(.*)", text)
        return match.group(1).strip() if match else text.strip()
        
    def extract_final_answer(self, text):
        match = re.search(r"\[Final Answer\]:\s*(.*)", text)
        return match.group(1).strip() if match else ""

    def build_sunar_messages(self, query, sub_queries: List[str]=None, sub_docs: List[str]=None, sub_answers: List[str]=None, mode: str=None):
        system_prompt = "Follow the given examples\n"
        prompt_pairs = [
            (
                "Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?\nAre follow up questions needed here:", 
                (
                    "Yes.\n"
                    "Follow up: How old was Theodor Haecker when he died?\n"
                    "Intermediate Answer: Theodor Haecker was 65 years old when he died.\n"
                    "Follow up: How old was Harry Vaughan Watkins when he died?\n"
                    "Intermediate Answer: Harry Vaughan Watkins was 69 years old when he died.\n"
                    "[Final Answer]: Harry Vaughan Watkins.\n\n"
                )
            ), 
            (
                "Question: Why did the founder of Versus die?\nAre follow up questions needed here:", 
                (
                    "Yes.\n"
                    "Follow up: Who founded Versus?\n"
                    "Intermediate Answer: Gianni Versace.\n"
                    "Follow up: Why did Gianni Versace die?\n"
                    "Intermediate Answer: Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997.\n"
                    "[Final Answer]: Shot.\n\n"
                )
            ), 
            (
                "Question: Who is the grandchild of Dambar Shah?\nAre follow up questions needed here:", 
                (
                    "Yes.\n"
                    "Follow up: Who is the child of Dambar Shah?\n"
                    "Intermediate Answer: Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. He was the father of Krishna Shah.\n"
                    "Follow up: Who is the child of Krishna Shah?\n"
                    "Intermediate Answer: Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. He was the father of Rudra Shah.\n"
                    "[Final Answer]: Rudra Shah.\n\n"
                )
            ), 
            (
                "Question: Who was born earlier director of Avengers or director of Frankenstein (1931 film)\nAre follow up questions needed here:", 
                (
                    "Yes.\n"
                    "Follow up: Who is the director of avengers?\n"
                    "Intermediate Answer: Russo brothers\n"
                    "Follow up: Who is the director of Frankenstein (1931 film)?\n"
                    "Intermediate Answer: James Whale\n"
                    "Follow up: Who was born earlier Russo brothers or James Whale?\n"
                    "Intermediate Answer: Russo brothers\n"
                    "[Final Answer]: Avengers\n\n"
                )
            ), 
            (
                "Question: Are both director of film FAQ: Frequently Asked Questions and director of film The Big Money from the same country?\nAre follow up questions needed here:", 
                (
                    "Yes.\n"
                    "Follow up: Who directed the film FAQ: Frequently Asked Questions?\n"
                    "Intermediate Answer: Carlos Atanes.\n"
                    "Follow up: Who directed the film The Big Money?\n"
                    "Intermediate Answer: John Paddy Carstairs.\n"
                    "Follow up: What is the nationality of Carlos Atanes?\n"
                    "Intermediate Answer: Carlos Atanes is Spanish.\n"
                    "Follow up: What is the nationality of John Paddy Carstairs?\n"
                    "Intermediate Answer: John Paddy Carstairs is British.\n"
                    "[Final Answer]: No.\n\n"
                )
            ), 
            (
                "Question: Are both the directors of Jaws and Casino Royale from the same country?\nAre follow up questions needed here:", 
                (
                    "Yes.\n"
                    "Follow up: Who is the director of Jaws?\n"
                    "Intermediate Answer: The director of Jaws is Steven Spielberg.\n"
                    "Follow up: Where is Steven Spielberg from?\n"
                    "Intermediate Answer: The United States.\n"
                    "Follow up: Who is the director of Casino Royale?\n"
                    "Intermediate Answer: The director of Casino Royale is Martin Campbell.\n"
                    "Follow up: Where is Martin Campbell from?\n"
                    "Intermediate Answer: New Zealand.\n"
                    "[Final Answer]: No\n\n"
                )
            ), 
            (
                f"Question: {query}\nAre follow up questions needed here:", 
                (
                    "Yes.\n"
                )
            )
        ]
        n = len(prompt_pairs)
        messages = [{"role":"system", "content":system_prompt}]
        for i, (user_prompt, assistant_prompt) in enumerate(prompt_pairs):
            messages.append({"role":"user", "content":user_prompt})
            if i < n-1:
                messages.append({"role":"assistant", "content":assistant_prompt})

        completion = "Yes.\n"

        # Generate sub-queries
        if mode == 'gen_sub_query':
            for sub_query, sub_ans in zip(sub_queries, sub_answers):
                completion += f"Follow up: {sub_query}\n"
                completion += f"Intermediate Answer: {sub_ans}\n"

        # Generate sub-answer
        elif mode == 'gen_sub_ans':
            assert sub_queries is not None and sub_docs is not None
            sub_query = sub_queries[-1]
            docs = sub_docs[-1]
            if len(docs) > 0 and 'content' in docs[0]:
                docs = [doc["content"] for doc in docs]
            elif len(docs) > 0 and 'contents' in docs[0]:
                docs = [doc["contents"] for doc in docs]
            docs = [self.truncate(doc, self.max_dlen) for doc in docs]
            docs = "\n".join(docs)

            completion += f"Follow up: {sub_query}\nEvidence: {docs}\n"
            # completion += f"Given the question and related evidence think step by step and generate the Intermediate Answer:"
            completion += f"Intermediate Answer:"

        if self.snowflake:
            if len(completion) > 0:
                assert messages[-1]['role'] == 'user'
                messages[-1]['content'] += completion
            prompt = messages
        elif self.vllm:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if len(completion) > 0:
                prompt += completion
        else:
            raise NotImplementedError
        
        return prompt


    def build_sunar_ans_gen_messages(self, query, final_docs: List[str], decomposed_path: str):
        if len(final_docs) > 0 and 'content' in final_docs[0]:
            final_docs = [doc["content"] for doc in final_docs]
        elif len(final_docs) > 0 and 'contents' in final_docs[0]:
            final_docs = [doc["contents"] for doc in final_docs]
        final_docs = [self.truncate(doc, self.max_dlen) for doc in final_docs]
        final_docs = "\n".join(final_docs)

        system_prompt = (
            "Follow the given examples and Given the question and context, reasoning path, think step by step extract key segments "
            "from given evidence relevant to question and give rationale, by forming your own reasoning path preceded by [Answer]: "
            "and output final answer for the question using information from given evidences and give concise precise answer preceded by [Final Answer]:\n"
        )
        user_prompt = (
            "[Question]: When does monsoon season end in the state the area code 575 is located?\n"
            "[Answer]: The area code 575 is located in New Mexico. Monsoon season in New Mexico typically ends in mid-September.\n"
            "[Final Answer]: mid-September.\n\n"

            "Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?\n"
            "[Answer]: Theodor Haecker was 65 years old when he died.Harry Vaughan Watkins was 69 years old when he died. Hence Harry Vaughan Watkins lived longer.\n"
            "[Final Answer]: Harry Vaughan Watkins.\n\n"

            "[Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?\n"
            "[Answer]: Ineabelle Diaz is from Peurto Rico, which is in the United States of America. The current official currency in the United States is the United States dollar.\n"
            "[Final Answer]: United States dollar.\n\n"

            "[Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?\n"
            "[Answer]: The person who founded the American Institute of Public Opinion in 1935 is George Gallup. George Gallup was born in Jefferson, Iowa.\n"
            "[Final Answer]: Jefferson.\n\n"

            "[Question]: What language is used by the director of Tiffany Memorandum?\n"
            "[Answer]: The director of Tiffany Memorandum is Sergio Grieco. Sergio Grieco speaks Italian.\n"
            "[Final Answer]: Italian.\n\n"

            "[Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?\n"
            "[Answer]: The player that scored the first touchdown in Superbowl 1 is Max McGee. Max McGee played for the Green Bay Packers.\n"
            "[Final Answer]: Green Bay Packers.\n\n"

            "[Question]: The birth country of Jayantha Ketagoda left the British Empire when?\n"
            "[Answer]: The birth country of Jayantha Ketagoda is Sri Lanka. Sri Lanka left the British Empire on February 4, 1948.\n"
            "[Final Answer]: February 4, 1948.\n\n"

            "Follow the above example and Given existing Reasoning path:\n"
            f"{decomposed_path}"
            "the evidence, Evidence: "
            f"{final_docs}"
            "and use the most relevant information for the question from the most relevant evidence from given Evidence: "
            "and form your own correct reasoning path to derive the answer thinking step by step preceded by [Answer]: "
            f"and subsequently give final answer as shown in above examples preceded by [Final Answer]: for the Question: {query}"
        )
        messages = [
            {"role":"system", "content":system_prompt}, 
            {"role":"user", "content":user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt