
python scripts/convert.py --mode SLM_rewrite_to_cache --dataset_name msmarco \
   --base_qa_path  data/QA_datasets \
   --base_cache_path cache \
   --raw_cache_file "search_cache_abs_500.json" \
   --input_file "search_cache_abs_500_trained_rewriter_rewritten_llama-3.2-3b-step-1740_0.json"