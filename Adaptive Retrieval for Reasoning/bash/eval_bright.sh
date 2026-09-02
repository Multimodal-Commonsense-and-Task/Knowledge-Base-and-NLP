# Evaluate results on BRIGHT



suffix="reasoning_and_ranking_qwen2.5-7b-instruct_512_v8_prefix_tevatron_72k_ndcg0.1_deepseek-r1_checkpoint-4500_contriever_td2d_freq2_alpha0.0_beta0.0"


retriever=bm25
categories=(
    "biology"
    "earth_science"
    "economics"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "pony"
    "leetcode"
    "aops"
    "theoremqa_questions"
    "theoremqa_theorems"
)

for category in "${categories[@]}"; do
    echo $category
    python -u scripts/evaluate.py \
        --qa_file /data/Search-o1/data/QA_Datasets/bright/$category.json \
        --base_cache_file /data/Search-o1/cache/bright/$retriever/search_cache_$retriever\_$category.json \
        --cache_file /data/Search-o1/cache/bright/$retriever/search_cache_$retriever\_$category\_$suffix.json
done

