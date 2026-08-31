OUT_FOLDER=results
INDEX_FOLDER=./common_index

for dataset in Books Movies
do
  for neg_set in all
	do
	  for method in bm25 pl2
	  do
	    config_path="test_config/${dataset}_${method}_${neg_set}.json"
	    out_path="$OUT_FOLDER/${dataset}/${neg_set}/${method}"
	    echo "$dataset::$neg_set::$method: Saving results to $out_path, using $config_path"
	    cmd="python run_lexical_benchmark.py evaluate_test --method terrier --common_index_path $INDEX_FOLDER --query_type all --dataset $dataset --negative_set $neg_set --config $config_path --out $out_path"
	    echo $cmd
	    $cmd
	  done
	done
done
