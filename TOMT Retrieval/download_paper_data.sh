# these lines downloads the data for 2017-2020
mkdir raw_dataset
# the following lines can be run in parallel
python download.py --start-time 01-01-2017 --end-time 31-12-2017 --conf ./reddit_config.json --submissions raw_dataset/2017_submissions.json --output raw_dataset/2017_submissions
python download.py --start-time 01-01-2018 --end-time 31-12-2018 --conf ./reddit_config.json --submissions raw_dataset/2018_submissions.json --output raw_dataset/2018_submissions
python download.py --start-time 01-01-2019 --end-time 31-12-2019 --conf ./reddit_config.json --submissions raw_dataset/2019_submissions.json --output raw_dataset/2019_submissions
python download.py --start-time 01-01-2020 --end-time 31-12-2020 --conf ./reddit_config.json --submissions raw_dataset/2020_submissions.json --output raw_dataset/2020_submissions

# copy all submission-pickles into a single folder
mkdir raw_dataset/2017_2020_submission_pickles/
cp raw_dataset/2017_submissions/* raw_dataset/2017_2020_submission_pickles/
cp raw_dataset/2018_submissions/* raw_dataset/2017_2020_submission_pickles/
cp raw_dataset/2019_submissions/* raw_dataset/2017_2020_submission_pickles/
cp raw_dataset/2020_submissions/* raw_dataset/2017_2020_submission_pickles/

# filter out unsolved submissions
python create_solved_cat.py --input_folder raw_dataset/2017_2020_submission_pickles/  --status_cat ./dataset/status_dict.json --norm_cat ./dataset/raw_categories.tsv --cat "Movie" --out raw_dataset/solved_Movies.json
python create_solved_cat.py --input_folder raw_dataset/2017_2020_submission_pickles/  --status_cat ./dataset/status_dict.json --norm_cat ./dataset/raw_categories.tsv --cat "Book/Story" --out raw_dataset/solved_Books.json

# extract answer candidates using the heuristic
python movies_extract_gt.py --input_json ./raw_dataset/solved_Movies.json --ent_folder ./raw_dataset/gt_Movies
python books_extract_gt.py --input_json ./raw_dataset/solved_Books.json --ent_folder ./raw_dataset/gt_Books

# extract negatives
python movies_extract_negatives.py --input_json ./raw_dataset/solved_Movies.json --neg_ent_folder ./raw_dataset/neg_Movies
python books_extract_negatives.py --input_json ./raw_dataset/solved_Books.json --neg_ent_folder ./raw_dataset/neg_Books

# keep only valid submissions, output gold dataset
python filter_available.py --input_json ./raw_dataset/solved_Movies.json --ent_folder ./raw_dataset/gt_Movies/ --out ./dataset/GoldMovies.json
python filter_available.py --input_json ./raw_dataset/solved_Books.json --ent_folder ./raw_dataset/gt_Books/ --out ./dataset/GoldBooks.json


# downloads negatives, create final data folders
python create_files.py ./dataset/GoldMovies.json ./dataset/Movies/ movie --negatives ./raw_dataset/neg_Movies
python create_files.py ./dataset/GoldBooks.json ./dataset/Books/ book --negatives ./raw_dataset/neg_Books

# split data into train/val/test (assumes data is in ./dataset/Movies and ./dataset/Books)
python split.py --input_json_movies ./raw_dataset/solved_Movies.json --ent_folder_movies ./raw_dataset/gt_Movies --input_json_books ./raw_dataset/solved_Books.json --ent_folder_books ./raw_dataset/gt_Books

# clean data
python clean_data.py ./dataset/Movies --sub_folders raw_dataset/2017_2020_submission_pickles/
python clean_data.py ./dataset/Books --sub_folders raw_dataset/2017_2020_submission_pickles/

# remove temp files in data
rm -rf ./dataset/Movies/temp
rm -rf ./dataset/Movies/temp_neg
rm -rf ./dataset/Books/temp
rm -rf ./dataset/Books/temp_neg


echo "Dataset for Movies: ./dataset/Movies/"
echo "Dataset for Books: ./dataset/Books/"

