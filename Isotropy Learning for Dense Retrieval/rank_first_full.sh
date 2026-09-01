#!/bin/bash

for dataset in "msmarco"
    do
        python -u rank_first_stage.py --beir_dataset $dataset \
                                    --model_path HIL/colbert-100000 \
                                    --index_path index/index-colbertv2-HIL-100k-len300 \
                                    --qidf /data/beir/$dataset/qidf.pickle;
done

