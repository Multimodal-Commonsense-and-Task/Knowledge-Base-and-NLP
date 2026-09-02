# HIL: Hybrid Isotropy Learning for Zero-shot Performance in Dense retrieval (NAACL 2024)

## Overview

Dense retrieval models such as ColBERT have achieved strong retrieval performance through fine-grained token-level interactions, but often struggle to generalize to unseen domains, where traditional methods such as BM25 can remain competitive. We investigate this limitation through the geometry of learned representations, observing complementary benefits of isotropy and anisotropy: isotropic representations facilitate similarity-based retrieval, while anisotropic representations can better preserve structures useful for generalization. Based on these observations, we propose Hybrid Isotropy Learning (HIL), which combines isotropic and anisotropic representations to improve zero-shot dense retrieval. Experiments on the BEIR benchmark show that HIL consistently outperforms ColBERT, demonstrating the benefit of balancing the two representation geometries for robust zero-shot retrieval.

## Instructions

The scripts expect data under `/data`.

```text
/data/
├── trec/
│   ├── qidpidtriples.train.full.2.tsv
│   ├── queries.train.tsv
│   ├── collection.tsv
│   └── qidf.pickle
├── beir/<dataset>/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   ├── qrels/{dev,test}.tsv
│   └── qidf.pickle
├── bm25/run.beir-bm25-flat.<dataset>.txt
└── colbert-prf/
    ├── experiments/
    ├── index/
    └── analysis/hypo/
```

Training is configured in `train.sh`. Indexing and retrieval use `index.sh` and `rank_first_full.sh`.

## Acknowledgments

This research was partially supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-2020-0-01789) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation). This work was also partially supported by IITP grant funded by MSIT [No.2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data and No.2021-0-01343-004, Artificial Intelligence Graduate School Program (Seoul National University)].
