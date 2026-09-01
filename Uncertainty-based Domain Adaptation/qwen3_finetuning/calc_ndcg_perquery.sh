ckpt=$1
dataset=$2
embedding_dir=${ckpt}/embeddings_beir

if [ -f $embedding_dir/evalperquery.${dataset}.txt ]; then
  echo "$embedding_dir/evalperquery.${dataset}.txt already exists, skipping evaluation."
else
  echo "Starting evaluation for $dataset..."
  python -m pyserini.eval.trec_eval -c -q -mndcg_cut.10 beir-v1.0.0-${dataset}-test $embedding_dir/rank.${dataset}.trec > $embedding_dir/evalperquery.${dataset}.txt
fi