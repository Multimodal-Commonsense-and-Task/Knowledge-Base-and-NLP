# Data




## Dataset Description

For each subset (Movies/Books) you can find the following files:

- `documents.json`  a JSONL file of known-items, each line is a JSON string representing a single item:
  ```json
    {
      "id": "item id",
      "text": "text description of the item",
      "title": "title of the item",
      "meta": {
        // dictionary of meta-data associated with this item
      }
    }
  ```
- `queries.json` A JSONL file where each line is a query with the JSON structure described below. 
Note that the `title` and `text` fields are not distributed, and need to be downloaded using 
the `fill_data.sh` or `process_data.py` script:
    ```json
    {
      "id": "submission / query id",
      "title": "raw title of the submission",
      "description": "raw description that accompanies the submission"
    }
    ``` 
- `qrels.txt` A QREL file which indicates the correct item for a particular query. Each line is one query/document pair 
    Note that they are tab seperated:  
    ```
        query_id_1	0	known_item_1	1
        query_id_2	0	known_item_2	1
        ...        
    ``` 
- ``


 
## Data collection

## Pre-requisites

- Setup the requirements mentioned in the [README](README.md)
- Download the following files: 
    - [Wikiplots](https://github.com/markriedl/WikiPlots)
        - Download both plots.zip and titles.zip, and move both unzipped files to `dataset/wikiplots`
    - [Geckodriver](https://github.com/mozilla/geckodriver/releases)
        - Download, unarchive and place in `./geckodriver`
    - [UCSD BookGraph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
        - Download `goodreads_book_series.json.gz`, `goodreads_book_works.json.gz` and `goodreads_books.json.gz`
        - move to `./dataset/ucsd_goodreads` and unarchive
  - Obtain a reddit client id and secret, and save them into a JSON `reddit_config.json`:

        {
            "client_id": "#####",
            "client_secret": "#####"
        }
        
        
## Reproducing data collection for 2017-2020

The data can be downloaded from [the data release folder](./data_release/). The script for collecting the data by yourself / reproducing it can be found in [download_paper_data.sh](download_paper_data.sh). 

If you want to collect your own dataset, or want to know in detail how it 
was done, see the next section: [Collecting Additional Data](#collecting-additional-data).   



## Collecting Additional Data

- Download submission ids and associated submissions using pushshift and PRAW, using the following command, which creates (a) a JSON file containing submission IDS and (b) a folder containing each submission (PRAW object) in a separate pickle. Note that it's a good idea to do this by year, so it can be done in parallel and is robust to network errors:
```
python download.py --start-time DD-MM-YYYY --end-time DD-MM-YYYY --conf /path/to/json --submissions /path/to/submissions/json --output /path/to/submissions/pickles 
```

-  The next command creates a JSON file with all submissions belonging to a particular category:

```
python create_solved_cat.py --input_folder /path/to/submissions/pickles  --status_cat ./dataset/status_dict.json --norm_cat ./dataset/raw_categories.tsv --cat "Movie" --out dataset/solved_Movies.json
python create_solved_cat.py --input_folder /path/to/submissions/pickles  --status_cat ./dataset/status_dict.json --norm_cat ./dataset/raw_categories.tsv --cat "Book/Story" --out dataset/solved_Books.json
```
- The next command applies the heuristic for finding the gold answers (see `--help` for other args):
```
python movies_extract_gt.py --input_json ./dataset/solved_Movies.json --ent_folder ./gt_Movies
python books_extract_gt.py --input_json ./dataset/solved_Books.json --ent_folder ./gt_Books
```
- The next command extracts negatives
```
python movies_extract_negatives.py --input_json ./dataset/solved_Movies.json --neg_ent_folder ./neg_Movies
python books_extract_negatives.py --input_json ./dataset/solved_Books.json --neg_ent_folder ./neg_Books
```
- The next file selects valid query/known-item pairs:
```
python filter_available.py --input_json ./dataset/solved_Movies.json --ent_folder ./gt_Movies/ --out ./dataset/GoldMovies.json
python filter_available.py --input_json ./dataset/solved_Books.json --ent_folder ./gt_Books/ --out ./dataset/GoldBooks.json
``` 
- Finally, the next group of commands downloads negatives and other candidates, and creates the dataset including the train/test/val split (third line)
```
python create_files.py ./dataset/GoldMovies.json ./dataset/Movies/ movie --negatives ./neg_Movies
python create_files.py ./dataset/GoldBooks.json ./dataset/Books/ book --negatives ./neg_Books
python clean_data.py ./dataset/Movies --sub_folders csv/of/paths/to/submission/pickles
python clean_data.py ./dataset/Books --sub_folders csv/of/paths/to/submission/pickles
python split.py --input_json_movies ./dataset/solved_Movies.json --ent_folder_movies ./gt_Movies --input_json_books ./dataset/solved_Books.json --ent_folder_books ./gt_Books
```
- The scripts above do not load the title / description, so execute the following commands to 
obtain the final data:

```
python process_dataset.py --input ./dataset/Books/raw_queries.json --out ./dataset/Books/queries.json --cache /path/to/submission/pickles
for split in train validation test
do
    python process_dataset.py --input ./dataset/Books/splits/$split/raw_queries.json --out ./dataset/Books/splits/$split/queries.json --cache /path/to/submission/pickles
done

python process_dataset.py --input ./dataset/Movies/raw_queries.json --out ./dataset/Movies/queries.json --cache /path/to/submission/pickles
for split in train validation test
do
    python process_dataset.py --input ./dataset/Movies/splits/$split/raw_queries.json --out ./dataset/Movies/splits/$split/queries.json --cache /path/to/submission/pickles
done
``` 