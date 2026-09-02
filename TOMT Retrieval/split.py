import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from config import configure_logging, supress_log
from tomt.benchmarks.gt import GTData
from tomt.data.goodreads import format_isbn, GoodReadsData
from tomt.data.imdb import IMDBApi
from tomt.data.train_test_split import train_val_test_split


def find_gold_ent_mov(gt_ent, gold_doc_id):
    gold_ent = None
    for og_ent in gt_ent:
        ent = og_ent["entity"]
        if ent is None: continue
        if ent.get("imdb_id") == gold_doc_id:
            gold_ent = og_ent
    return gold_ent


def find_gold_ent_book(gt_ent, gold_doc_id):
    gold_ent = None
    for og_ent in gt_ent:
        ent = og_ent["entity"]
        if ent is None: continue
        if ent.get("work_id") == gold_doc_id:
            gold_ent = og_ent
        else:
            work = goodreads_data.get_work(isbn10=format_isbn(ent["isbn"]))
            if work is None:
                work = goodreads_data.get_work(isbn13=format_isbn(ent["isbn13"]))
            if work is None:
                raise ValueError()

            if work["work_id"] == gold_doc_id:
                gold_ent = og_ent

    return gold_ent


def get_answer_positions(gt_data, solved_gt_path, submissions_path, find_gold_ent):
    queries = gt_data.get_queries()
    qrels = gt_data.get_qrels()

    with open(submissions_path) as reader:
        submissions = json.load(reader)

    answer_positions = {}
    for query in queries:
        qid = query["id"]
        with open(os.path.join(solved_gt_path, f"{qid}.json")) as reader:
            gt_ent = json.load(reader)
        gold_doc_id = list(qrels[qid].keys())[0]

        gold_ent = find_gold_ent(gt_ent, gold_doc_id)

        assert gold_ent is not None
        reply_id = gold_ent["uttrance"]["id"]
        ans_pos = None
        for pos, ut in enumerate(submissions[qid]["solved_path"], 1):
            if reply_id == ut["id"]:
                ans_pos = pos

        answer_positions[qid] = ans_pos

    return answer_positions


def create_ans_pos_subids(root, solved_gt_path, submissions_path, find_gold_func):
    for split in {"train", "test", "validation"}:
        gt_path = os.path.join(root, split)
        gt_data = GTData(gt_path)
        answer_pos = get_answer_positions(gt_data, solved_gt_path, submissions_path, find_gold_func)
        print(f"Subset: {split}")
        for key, val in Counter(answer_pos.values()).items():
            print(f"\tAnswer Pos: {key} :: {val}")

        assert len(answer_pos) == len(gt_data.get_queries())

        pos_ids = defaultdict(list)
        for sub_id, pos in answer_pos.items():
            pos_ids[pos].append(sub_id)

        os.makedirs(os.path.join(gt_path, "subsets"), exist_ok=True)
        for pos, sub_ids in pos_ids.items():
            with open(os.path.join(gt_path, "subsets", f"answer_pos_ids_{pos}.json"), "w") as writer:
                json.dump(sub_ids, writer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Split")
    parser.add_argument("--input_json_movies", help="the location of the file output by create_solved_cat",
                        required=True)
    parser.add_argument("--input_json_books", help="the location of the file output by create_solved_cat",
                        required=True)
    parser.add_argument("--ent_folder_movies", help="location to dump entities", required=True)
    parser.add_argument("--ent_folder_books", help="location to dump entities", required=True)

    args = parser.parse_args()

    # First, create train/test/val splits
    np.random.seed(42)
    random.seed(42)
    configure_logging("Split", True)
    supress_log("imdbpy.parser.http.piculet")

    print("Creating Splits")
    if not os.path.exists("./dataset/Books/splits"):
        train_val_test_split("./dataset/Books")
    if not os.path.exists("./dataset/Movies/splits"):
        train_val_test_split("./dataset/Movies")

    print("Creating Splits complete")

    # imdb_api = IMDBApi("./imdb_cache")
    # goodreads_data = GoodReadsData("./dataset/ucsd_goodreads/")
    #
    # print("Analyzing position of answers")
    #
    # root = "./dataset/Books/splits"
    # solved_gt_path = args.ent_folder_books
    # submissions_path = args.input_json_books
    # create_ans_pos_subids(root, solved_gt_path, submissions_path, find_gold_ent_book)
    # print("Analyzing position of answers for Books: Done!")
    #
    # root = "./dataset/Movies/splits"
    # solved_gt_path = args.ent_folder_movies
    # submissions_path = args.input_json_movies
    # create_ans_pos_subids(root, solved_gt_path, submissions_path, find_gold_ent_mov)
    # print("Analyzing position of answers for Movies: Done!")
