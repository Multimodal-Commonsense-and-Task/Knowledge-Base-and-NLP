#!/bin/bash

CUDA_VISIBLE_DEVICES="0" \
python -m colbert.train \
--amp \
--query_maxlen 300 \
--doc_maxlen 300 \
--mask-punctuation \
--bsize 32 \
--accum 1 \
--experiment colbert-ours-beir \
--similarity cosine \
--run msmarco.psg.cosine \
--triples /data/trec/qidpidtriples.train.full.2.tsv \
--queries /data/trec/queries.train.tsv \
--collection /data/trec/collection.tsv \
--regularizer sensim \
--reg_lambda -0.3 \
--qidf /data/trec/qidf.pickle


