import argparse
import os
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import json

import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Dense Retrieval using Dense Passage Retriever (DPR) ####
# DPR implements a two-tower strategy i.e. encoding the query and document seperately.
# The DPR model was fine-tuned using dot-product (dot) function.

#########################################################
#### 1. Loading DPR model using SentenceTransformers ####
#########################################################
# You need to provide a ' [SEP] ' to seperate titles and passages in documents
# Ref: (https://www.sbert.net/docs/pretrained-models/dpr.html)



def main():
    parser = argparse.ArgumentParser(description='Evaluate DPR on a BEIR dataset')
    parser.add_argument('--beir_dataset_path', required=True, type=str,
                        help='Path to the BEIR dataset folder (containing corpus.jsonl, queries.jsonl, qrels/).')
    parser.add_argument('--model_dir', required=True, type=str,
                        help='Directory with fine-tuned question_encoder and ctx_encoder subfolders.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding queries and documents.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')

    args = parser.parse_args()

    corpus, queries, qrels = GenericDataLoader(data_folder=args.beir_dataset_path).load(split="test")
    model = DRES(models.SentenceBERT((
        os.path.join(args.model_dir, "question_encoder"),
        # "facebook-dpr-question_encoder-multiset-base",
        os.path.join(args.model_dir, "ctx_encoder"),
        # "facebook-dpr-ctx_encoder-multiset-base",
        " [SEP] "), batch_size=args.batch_size))

    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)
    with open(f"{args.model_dir}/results.json", "w") as f:
        json.dump(results, f)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    print("\n###################### Results ######################")
    print("NDCG@k:", ndcg)

    # save ndcg scores to the json file
    with open(f"{args.model_dir}/ndcg.json", "w") as f:
        json.dump(ndcg, f)

if __name__ == "__main__":
    main()
