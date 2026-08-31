
SUBMISSIONS_CACHE=./.submissions_cache

SUFFIX=" --force_refresh"


python process_dataset.py --input ./dataset/Books/raw_queries.json --out ./dataset/Books/queries.json --cache $SUBMISSIONS_CACHE $SUFFIX
for split in train validation test
do
   python process_dataset.py --input ./dataset/Books/splits/$split/raw_queries.json --out ./dataset/Books/splits/$split/queries.json --cache $SUBMISSIONS_CACHE
done


python process_dataset.py --input ./dataset/Movies/raw_queries.json --out ./dataset/Movies/queries.json --cache $SUBMISSIONS_CACHE $SUFFIX
for split in train validation test
do
   python process_dataset.py --input ./dataset/Movies/splits/$split/raw_queries.json --out ./dataset/Movies/splits/$split/queries.json --cache $SUBMISSIONS_CACHE
done