#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
python summarizer/compact_summarization.py --search_cache_path 'cache/hotpotqa/search_cache_500.json'
