VALID_DATASETS = { 'biology', 'earth_science', 'economics', 'psychology', 'robotics', 'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 'aops', 'theoremqa_questions', 'theoremqa_theorems' }
VALID_QUERY_TYPES = {'original', 'gpt4', 'llama3', 'gemini1', 'claude3', 'grit'}
VALID_METHODS = {'bm25', 'reasonir', 'smr', 'emr'}
VALID_RETRIEVERS = {'bm25', 'reasonir'}
OPENAI_MODELS = {'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'o1', 'o1-pro', 'o1-mini', 'o3', 'o3-pro', 'o3-mini', 'o4-mini'}
VLLM_MODELS = {'qwen3', 'qwen2.5'}
VALID_LLMS = OPENAI_MODELS | VLLM_MODELS
EVAL_METRICS = {'ndcg', 'ndcg_cut', 'map', 'map_cut', 'recall', 'P', 'recip_rank'}
METHOD_MAPPING = {}

def register_method(name: str):
    def decorator(cls):
        METHOD_MAPPING[name] = cls
        return cls
    return decorator
