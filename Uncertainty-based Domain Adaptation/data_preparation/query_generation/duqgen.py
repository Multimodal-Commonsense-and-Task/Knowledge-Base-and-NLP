from langchain_community.chat_models import ChatOllama

from tqdm import tqdm
from src.utils.data_utils import load_jsonl_cache, load_jsonl_corpus
from src.examples.common_prompts.summary_gen import duqgen_dataset
from src.examples.common_utils.save import save_duquery
import argparse
import os
from common_utils.ntfy import antfy
import asyncio
import json
import uuid
from common_utils.decide_host import allocate_host, release_host

uuid4 = uuid.uuid4()


@antfy
async def main(args, **kwargs):
    semaphore = asyncio.Semaphore(args.semaphore)
    host = allocate_host(uuid4)
    client = ChatOllama(
        model="llama3:8b-instruct-fp16",
        base_url=host,
        num_ctx=4096,
        seed=123,
        temperature=0.8,
    )

    try:
        datasets = args.datasets
        query_gen = duqgen_dataset
        for dataset in datasets:

            async def process_document(did, doc, client, semaphore):
                async with semaphore:
                    try:
                        summary = await query_gen(doc, client, dataset)
                        # save_duquery(did, summary, dataset, case=args.case)
                        return did, summary
                    except Exception as e:
                        print(e)
                        return did, None

            corpus = load_jsonl_corpus(dataset, case=args.case)
            # target_corpus = load_jsonl(dataset)[args.range[0] : args.range[1]]
            gen_queries = {}
            cached = load_jsonl_cache(dataset, type="query")
            target_corpus = list(corpus.items())[args.range[0] : args.range[1]]
            target_corpus_no_cache = []
            for did, doc in target_corpus:
                if did in cached:
                    gen_queries[did] = cached[did]
                else:
                    target_corpus_no_cache.append((did, doc))
            target_corpus = target_corpus_no_cache

            while True:
                failed = []
                tasks = []
                for did, doc in target_corpus:
                    doc = " ".join(doc.split()[: args.max_input_seq_length])
                    task = asyncio.create_task(
                        process_document(did, doc, client, semaphore)
                    )
                    tasks.append(task)

                for task in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="DUQuery generation",
                ):
                    did, query = await task
                    if query is not None and len(query) > 0:
                        gen_queries[did] = query
                    else:
                        failed.append((did, corpus[did]))
                if len(failed) == 0:
                    break
                target_corpus = failed

            with open(
                f"/output_path/data/{dataset}_query{args.case}.jsonl", "w"
            ) as f:
                for did, query in gen_queries.items():
                    json.dump(
                        {"docid": did, "doctext": corpus[did], "question": query}, f
                    )
                    f.write("\n")

            # with open(f"/output_path/data/{dataset}_query.jsonl.cache", "a") as f:
            #     for did, doc in target_corpus_no_cache:
            #         json.dump(
            #             {"docid": did, "doctext": doc, "question": gen_queries[did]}, f
            #         )
            #         f.write("\n")
    finally:
        release_host(uuid4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, required=False, help="host of the chat server"
    )
    parser.add_argument(
        "-r",
        "--range",
        type=int,
        nargs=2,
        help="Range of documents to generate queries",
    )
    parser.add_argument(
        "--fill_holes",
        type=bool,
        default=False,
        help="Fill holes in the already done queries",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="query_validate",
        help="Name of the job",
    )
    parser.add_argument(
        "--semaphore",
        type=int,
        default=8,
        help="Number of concurrent queries to generate",
    )
    parser.add_argument(
        "--max_input_seq_length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="",
        help="Case of the summary",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        nargs="+",
        help="Datasets to generate queries",
    )
    parser.add_argument(
        "--query_per_doc",
        required=False,
        type=int,
        default=1,
        help="Number of queries to generate per document",
    )
    args = parser.parse_args()
    asyncio.run(main(args, job_name=args.job_name))
