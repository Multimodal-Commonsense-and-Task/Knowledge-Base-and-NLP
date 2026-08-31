import json
import os

import pandas as pd
from tqdm.autonotebook import tqdm
import argparse


def iterate_gt(folder, submissions):
    for f in sorted(os.listdir(folder)):
        sid = f.split(".")[0]
        with open(os.path.join(folder, f)) as reader:
            try:
                j = json.load(reader)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"{f} got a JSON decode error. Skipping!")
                continue

        yield sid, j, submissions[sid]


def write_gold(submissions_path, solved_gt_path, gold_path):
    with open(submissions_path) as reader:
        raw_submissions = json.load(reader)
        submissions = {}
        for sid in raw_submissions:
            submissions[sid] = raw_submissions[sid]["submission"]

    stats_rows = []

    MAX = len(os.listdir(solved_gt_path))

    for sid, entities, submission in tqdm(iterate_gt(solved_gt_path, submissions), total=MAX, leave=False,
                                          desc="Gathering stats"):

        n_confident = 0
        for e in entities:
            n_confident += e["confident"]

        stats_rows.append({
            "submission_id": sid,
            "n_entities": len(entities),
            "n_confident": n_confident,
        })

    gold_submissions = []

    sub_stats = pd.DataFrame(stats_rows)
    conf_stats = sub_stats[sub_stats.n_confident > 0]
    for _, row in tqdm((conf_stats[conf_stats.n_confident == 1]).iterrows(), desc="Creating gold",
                       total=sum(conf_stats.n_confident == 1)):
        submission = submissions[row["submission_id"]]

        with open(os.path.join(solved_gt_path, row["submission_id"] + ".json")) as reader:
            entities = json.load(reader)

        solved_ents = []
        for ent in entities:
            if ent["confident"]:
                solved_ents.append(ent)

        if len(solved_ents) != 1:
            print(entities)
            raise ValueError()

        submission["gold_entity"] = solved_ents[0]

        gold_submissions.append(submission)

    with open(gold_path, "w") as writer:
        # write one per line
        for g in gold_submissions:
            assert g["gold_entity"]["entity"] is not None
            writer.write(f"{json.dumps(g)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("WriteGold")
    parser.add_argument("--input_json", help="the location of the file output by create_solved_cat", required=True)
    parser.add_argument("--ent_folder", help="location of gt entites", required=True)
    parser.add_argument("--out", help="location to write JSON", required=True)

    args = parser.parse_args()

    if os.path.exists(args.out):
        raise ValueError(f"{args.out} exists!")
    write_gold(args.input_json, args.ent_folder, args.out)
