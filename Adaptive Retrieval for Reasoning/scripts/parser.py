import argparse


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='/data/models/Llama-3.2-1B-Instruct')
    parser.add_argument('--peft_model_name', type=str, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--max_qlen', type=int, default=1024)
    parser.add_argument('--max_dlen', type=int, default=1024)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--snowflake',action='store_true')
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser


def add_task_specific_args(parser, task):
    if task == "rerank":
        parser.add_argument('--qa_path', type=str, default='../data/QA_Datasets/bright/aops.json')
        parser.add_argument('--cache_path', type=str, default='../cache/bright/bm25/search_cache_bm25_aops.json')
        parser.add_argument('--corpus_path', type=str, default=None)
        parser.add_argument('--graph_path', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--window_size', type=int, default=20)
        parser.add_argument('--top_k_docs', type=int, default=100)
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--beta', type=float, default=1.0)
        parser.add_argument('--bt_iter', type=int, default=5)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--prompt_mode', type=str, default='reasoning_and_ranking')
        parser.add_argument('--d2d', type=str, default = '', help = 'format should be gar_{d2d_model_name}_{concat_strategy}_freq{frequency}')
        parser.add_argument('--reuse', action='store_true')
        parser.add_argument('--slidegar', action='store_true')
        parser.add_argument('--return_trace', action='store_true')

    elif task == "generate_data":
        parser.add_argument('--cache_path', type=str, default='../cache/bright/bm25/search_cache_bm25_aops.json')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--window_size', type=int, default=20)
        parser.add_argument('--top_k_docs', type=int, default=100)
        parser.add_argument('--prompt_mode', type=str, default='reasoning_and_ranking')

    elif task in ["train_dpo", "train_sft", "train_grpo"]:
        parser.add_argument('--ds_config', type=str, default='./train/config/accelerate_configs/ds_zero0_config.json')
        parser.add_argument('--train_data_path', type=str, default='./train/data/train_sft_tevatron_v3_ndcg_0.5.jsonl')
        parser.add_argument('--output_dir', type=str, default='./train/models/sft')
        parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
        parser.add_argument('--num_train_epochs', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--prompt_mode', type=str, default='reasoning_and_ranking')
        
    elif task == "construct_graph":
        parser.add_argument('--qa_path', type=str, default='../data/QA_Datasets/bright/aops.json')
        parser.add_argument('--corpus_path', type=str, default='./data/bright/corpus.jsonl')
        parser.add_argument('--output_dir', type=str, default='./data/bright')
        parser.add_argument('--dataset', type=str, default='bright')
        parser.add_argument('--k', type=int, default=16)
        parser.add_argument('--batch_size', type=int, default=512)

    elif task == "sunar":
        parser.add_argument('--qa_path', type=str, default='/data/Search-o1/cache/hotpotqa/search_cache_500_wiki_top100.json')
        parser.add_argument('--cache_path', type=str, default='/data/Search-o1/cache/hotpotqa/search_cache_500_wiki_top100.json')
        parser.add_argument('--corpus_path', type=str, default=None)
        parser.add_argument('--graph_path', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--window_size', type=int, default=20)
        parser.add_argument('--top_k_docs', type=int, default=100)
        parser.add_argument('--max_sunar_tokens', type=int, default=2048)
        parser.add_argument('--prompt_mode', type=str, default='reasoning_and_ranking')
        parser.add_argument('--d2d', type=str, default = '', help = 'format should be gar_{d2d_model_name}_{concat_strategy}_freq{frequency}')
        parser.add_argument('--reuse', action='store_true')
        parser.add_argument('--return_trace', action='store_true')

    else:
        raise ValueError(f"Unsupported task: {task}")

    return parser

def parse_args():
    base_parser = get_base_parser()
    base_args, remaining_args = base_parser.parse_known_args()

    full_parser = get_base_parser()
    full_parser = add_task_specific_args(full_parser, base_args.task)

    args = full_parser.parse_args()
    return args
