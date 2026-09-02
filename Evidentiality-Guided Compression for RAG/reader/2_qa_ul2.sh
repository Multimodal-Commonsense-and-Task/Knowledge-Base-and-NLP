# Usage: sh 2_qa_ul2.sh [task]
# Same evaluation with the open reader (Flan-UL2), which is also the LLM used for
# evidentiality mining.
task=${1:-NQ}

python qa_ul2.py \
    --eval_data ../data/reader/$task/test.json \
    --name $task/ECoRAG_ul2 \
    --n_context 20
