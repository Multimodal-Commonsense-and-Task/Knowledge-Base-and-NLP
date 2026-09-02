import re
import os
import time
import csv
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from ftfy import fix_text
from copy import deepcopy
from collections import defaultdict
from typing import Union, Tuple, Dict, List

from transformers import AutoTokenizer

from scripts.llm import Base_LLM
from utils.utils import json_dump
from utils.prompts import get_rerank_system_prompt, get_rerank_user_prompt


class Tracer:
    def __init__(self, model_name_or_path, max_qlen=1024, max_dlen=1024, 
                batch_size=8, window_size=20, prompt_mode='reasoning', **kwargs):
        self.kwargs = kwargs
        self.max_qlen = max_qlen
        self.max_dlen = max_dlen
        self.batch_size = batch_size
        self.window_size = window_size
        self.prompt_mode = prompt_mode
        self.model_name_or_path = model_name_or_path
        self.snowflake = self.kwargs.get("snowflake", False)
        self.debug = self.kwargs.get("debug", False)

        self.model = Base_LLM(
            self.model_name_or_path, 
            **kwargs, 
        )
        if self.snowflake:
            if os.path.exists('/data/models/Llama-3.2-1B-Instruct'):
                tokenizer_path = '/data/models/Llama-3.2-1B-Instruct'
            else:
                raise ValueError("Please provide the path to the tokenizer for Snowflake model.")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True, use_fast=True)

    def save_tmp_results(self, queries, results):
        trace_cache = {}
        for query, result in zip(queries, results):
            if len(result['trace']) == 0:
                continue
            if len(result['docs']) == 0 and self.prompt_mode == 'reasoning_and_ranking':
                continue
            trace_cache[query] = result

        dir_path = os.path.dirname(self.kwargs['cache_path'])
        model_short_name = self.model_name_or_path.split("/")[-1].lower()
        output_file = os.path.basename(self.kwargs['cache_path']).replace("search_cache", "trace_cache")[:-5] + f"_{self.prompt_mode}_{model_short_name}_tmp.json"
        output_path = os.path.join(dir_path, output_file)

        json_dump(output_path, trace_cache)
        print(f"#> Saving tmp {output_path} done\n")

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])

    def build_prompt(self, query: str, docs: List[str]) -> str:
        messages = list()
        messages.append({"role": "system", "content": get_rerank_system_prompt(
            self.prompt_mode, n_docs=len(docs), model_name_or_path=self.model_name_or_path)})

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
            model_name_or_path=self.model_name_or_path, 
        ))
        messages.append({"role": "user", "content": user_prompt})

        if self.snowflake:
            prompt = messages
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return prompt

    def parse_tracing(self, output: str, n_docs: int=None):
        try:
            if self.prompt_mode == 'reasoning_w_trace':
                assert "[Reasoning Trace]" in output
                assert "[Final Answer]" in output
                reasoning_match = re.search(r"\[Reasoning Trace\](.*?)\n\[Final Answer\]", output, re.DOTALL)
                reasoning_section = reasoning_match.group(1)
                trace = re.findall(r"Step \d+:\s*(.+)", reasoning_section)
                return trace, []

            elif self.prompt_mode == 'reasoning_and_ranking':
                assert "[Reasoning Trace]" in output
                assert "[Document Ranking]" in output
                assert "[Document Support Analysis]" in output
                reasoning_match = re.search(r"\[Reasoning Trace\](.*?)\n\[Document Ranking\]", output, re.DOTALL)
                reasoning_section = reasoning_match.group(1)
                trace = re.findall(r"Step \d+:\s*(.+)", reasoning_section)

                analysis = output.split("[Document Support Analysis]")[-1]
                analysis = analysis.split("[Document Ranking]")[0].strip()

                ids = output.split("[Document Ranking]")[-1]
                if "Note" in ids:
                    ids = ids.split("Note")[0]
                ids = ids.split(">")

                indices = []
                for idx in ids:
                    idx = idx.strip()
                    idx = idx.replace("[","").replace("]","")
                    assert idx.isdigit() == True
                    if (int(idx)-1 not in indices) and int(idx)-1 < min(n_docs, self.window_size):
                        indices.append(int(idx)-1)
                assert len(indices) >= min(n_docs, self.window_size//2)
                return trace, indices[:self.window_size], analysis

        except Exception:
            print(f"#> Error output sample: {output}")
            return [], [], ""

    def generate_data(self, queries: List[str], documents: List[List[str]]) -> List[Union[str, Dict[str, str]]]:
        """
        Args:
            queries: List of queries.

        Returns:
            List of list of reasoning steps.
        """
        assert len(queries) == len(documents)
        N = len(queries)
        results = []
        total_start_time = time.time()
        print(f"#> Genearte trace start")

        for i in tqdm(range(0, N, self.batch_size)):
            batch_items = [{'query':query,'docs':docs} for query, docs in zip(queries[i:i+self.batch_size], documents[i:i+self.batch_size])]
            prompts = []
            prompt_info = []

            for items in batch_items:
                query = items['query']
                docs = items['docs']

                prompt = self.build_prompt(query, docs)
                prompts.append(prompt)
                prompt_info.append((query, docs))

            outputs = self.model.llm_batch_inference(prompts)

            for output, prompt, (query, docs) in zip(outputs, prompts, prompt_info):
                trace, ranking, analysis = self.parse_tracing(output, n_docs=len(docs))
                if self.debug:
                    print(f"#> [DEBUG] System Prompt: {prompt[0]['content']}\n")
                    print(f"#> [DEBUG] User Prompt: {prompt[1]['content']}\n")
                    print(f"#> [DEBUG] Output: {output}\n\nTrace: {trace}\n\nRanking: {ranking}")
                    assert False

                ranked_docs = []
                for idx in ranking:
                    doc = docs[idx]
                    doc['idx_of_retrieval'] = idx
                    ranked_docs.append(doc)
                results.append({'query': query, 'trace':trace, 'analysis':analysis, 'docs':ranked_docs, 'raw_docs':docs})

                if len(results) % 100 == 0:
                    self.save_tmp_results(queries, results)
        
        assert len(results) == len(queries), f"{len(results)}; {len(queries)}"
        total_end_time = time.time()
        print(f"#> Total execution time: {total_end_time - total_start_time:.2f}s")
        
        return results
    