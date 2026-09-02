# Usage: sh 1_qa_gpt.sh [task]
# Answers with GPT-4o-mini over the final compression. Requires OPENAI_API_KEY.
task=${1:-NQ}

python qa_gpt.py \
    --eval_data ../data/reader/$task/test.json \
    --name $task/ECoRAG_gpt \
    --n_context 20 \
    --temperature 0.0
