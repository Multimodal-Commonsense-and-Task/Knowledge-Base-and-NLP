# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
Single-hop retrieval evaluation

## Use the unified model (trained with both hotpotQA and NQ)


python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-val-simplified.txt index/psg100_unified.npy index/psgs_w100_id2doc.json logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 1000 --shared-encoder --model-name bert-base-uncased --unified --save-pred nq-val-filtered-top50.txt --topk 50


# DPR shared-encoder baseline bsz256
python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/psg100_dpr_shared_baseline.npy index/psgs_w100_id2doc.json logs/08-23-2020/nq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --max-q-len 50 --save-pred nq-test-dpr-shared-b256-res.txt  

# shared encoder on merged corpus
python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/merged_all_single_only.npy index/merged_all_id2doc.json logs/08-23-2020/nq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --max-q-len 50

# to get negatives from DPR shared baseline
python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-val-simplified.txt index/psg100_dpr_shared_baseline.npy index/psgs_w100_id2doc.json logs/08-25-2020/wq_mhop_1_shared_dpr_neg_from_scratch-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --save-pred nq-val-shared-dpr-top100.txt --topk 100 

python eval_retrieval.py /private/home/xwhan/data/WebQ/WebQuestions-test.txt index/psg100_mhop_wq_1_from_baseline.npy index/psgs_w100_id2doc.json  logs/08-26-2020/wq_mhop_1_shared_dpr_neg_from_scratch-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --save-pred wq-test-res-type1.txt


python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/merged_all.npy index/merged_all_id2doc.json logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 1000 --shared-encoder --model-name bert-base-uncased --unified



"""

import numpy as np
import json
import faiss
import argparse
import logging
import torch
from tqdm import tqdm

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import defaultdict

from utils.utils import load_saved, move_to_cuda, para_has_answer
from utils.basic_tokenizer import SimpleTokenizer

from transformers import AutoConfig, AutoTokenizer
from models.retriever import BertRetrieverSingle, RobertaRetrieverSingle
from models.unified_retriever import UnifiedRetriever, BertNQRetriever
from pyserini.prf import DenseVectorRocchioPrf, DenseVectorAveragePrf

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

PROCESS_TOK = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--max-q-len', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save-pred", default="", type=str)
    parser.add_argument("--unified", action="store_true", help="test with unified trained model")
    # parser.add_argument("--plot_in_query", default="", type=str)

    # prf argument
    parser.add_argument('--prf-depth', type=int, metavar='num of passages used for PRF', required=False, default=0,
                        help="Specify how many passages are used for PRF, 0: Simple retrieval with no PRF, > 0: perform PRF")
    parser.add_argument('--prf-method', type=str, metavar='avg or rocchio', required=False, default='avg',
                        help="Choose PRF methods, avg or rocchio")
    parser.add_argument('--rocchio-alpha', type=float, metavar='alpha parameter for rocchio', required=False,
                        default=0.9,
                        help="The alpha parameter to control the contribution from the query vector")
    parser.add_argument('--rocchio-beta', type=float, metavar='beta parameter for rocchio', required=False, default=0.1,
                        help="The beta parameter to control the contribution from the average vector of the positive PRF passages")
    parser.add_argument('--rocchio-gamma', type=float, metavar='gamma parameter for rocchio', required=False, default=0.1,
                        help="The gamma parameter to control the contribution from the average vector of the negative PRF passages")
    parser.add_argument('--rocchio-topk', type=int, metavar='topk passages as positive for rocchio', required=False, default=3,
                        help="Set topk passages as positive PRF passages for rocchio")
    parser.add_argument('--rocchio-bottomk', type=int, metavar='bottomk passages as negative for rocchio', required=False, default=0,
                        help="Set bottomk passages as negative PRF passages for rocchio, 0: do not use negatives prf passages.")

    args = parser.parse_args()

    logger.info(f"Loading questions")
    qas = [json.loads(line) for line in open(args.raw_data).readlines()]
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in qas]
    qids = [_["_id"] for _ in qas]
    answers = [item["answer"] for item in qas]

    logger.info(f"Loading index")
    d = 768
    xb = np.load(args.indexpath).astype('float32')
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    if args.gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 1, index)

    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.unified:
        model = UnifiedRetriever(bert_config, args)
    elif "roberta" in args.model_name:
        model = RobertaRetrieverSingle(bert_config, args)
    else:
        model = BertRetrieverSingle(bert_config, args)

    model = load_saved(model, args.model_path, exact=False)
    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp

    model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info(f"Loading corpus")
    id2doc = json.load(open(args.corpus_dict))
    logger.info(f"Corpus size {len(id2doc)}")

    # Check PRF Flag
    if args.prf_depth > 0:
        PRF_FLAG = True
        if args.prf_method.lower() == 'avg':
            prfRule = DenseVectorAveragePrf()
        elif args.prf_method.lower() == 'rocchio':
            prfRule = DenseVectorRocchioPrf(args.rocchio_alpha, args.rocchio_beta, args.rocchio_gamma,
                                            args.rocchio_topk, args.rocchio_bottomk)
        # ANCE-PRF is using a new query encoder, so the input to DenseVectorAncePrf is different
        elif args.prf_method.lower() == 'ance-prf':
            pass
        print(f'Running FaissSearcher with {args.prf_method.upper()} PRF...')
    else:
        PRF_FLAG = False


    retrieved_results = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ans = answers[b_start:b_start + args.batch_size]
            batch_qids = qids[b_start:b_start + args.batch_size]
            batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, pad_to_max_length=True,
                                                          return_tensors="pt", is_pretokenized=True)
            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"],
                                      batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            if PRF_FLAG:
                D, I, V = index.search_and_reconstruct(
                    q_embeds_numpy, args.prf_depth)
                prf_embs_q = prfRule.get_batch_prf_q_emb(batch_qids,
                                                         q_embeds_numpy,
                                                         [(score, idx, vector) for
                                                          score, idx, vector in
                                                          zip(D, I, V)])
            D, I = index.search(prf_embs_q, args.topk)
            for b_idx in range(len(batch_q)):
                topk_docs = []
                for doc_id, doc_score in zip(I[b_idx], D[b_idx]):
                    topk_docs.append({"title": id2doc[str(doc_id)][0], "score": float(doc_score)})
                retrieved_results.append(topk_docs)

    assert len(qids) == len(questions) == len(answers) == len(retrieved_results)
    preds = {}
    for qid, ret_res in zip(qids, retrieved_results):
        preds[qid] = {d["title"]: d["score"] for d in ret_res}
    assert args.save_pred != ""
    with open(args.save_pred, "w") as writer:
        json.dump(preds, writer)
