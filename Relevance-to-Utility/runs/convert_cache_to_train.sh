python scripts/convert.py --mode LLM_rewrite_to_cache --dataset_name mmlu \
   --base_qa_path  data/QA_datasets \
   --base_cache_path cache \
   --raw_cache_file "search_cache_train_40000.json" \
   --input_file "search_cache_train_40000_conditional_cot2_rewritten_snowflake-llama-3.3-70b_*.json" \


python scripts/convert.py --mode cache_to_train_jsonl --dataset_name mmlu \
   --base_qa_path  data/QA_Datasets \
   --base_cache_path cache \
   --raw_cache_file "search_cache_train_40000.json" \
   --raw_qa_file "mmlu_train_40000.json" \
   --input_file "search_cache_train_40000_conditional_cot2_rewritten_snowflake-llama-3.3-70b_split.json" \
   --train_dataset_dir "FIRST/datasets" 

