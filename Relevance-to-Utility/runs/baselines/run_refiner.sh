export CUDA_VISIBLE_DEVICES=3

# PEFT SHOULD BE INSTALLED
# pip install -U peft
python summarizer/refiner_summarization.py --search_cache_path cache/msmarco/search_cache_abs_500.json
python summarizer/refiner_summarization.py --search_cache_path cache/hotpotqa/search_cache_500.json
python summarizer/refiner_summarization.py --search_cache_path cache/2wiki/search_cache_500.json