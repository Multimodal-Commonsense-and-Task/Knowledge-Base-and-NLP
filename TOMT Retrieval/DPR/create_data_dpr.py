import os
import json
from tomt.benchmarks.gt import GTData, get_documents

import argparse

parser = argparse.ArgumentParser("create_data_dpr")
parser.add_argument("--root", required=True)
parser.add_argument("--dataset", required=True, choices=("Movies", "Books"))
parser.add_argument("--file_path", required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.file_path, exist_ok=True)
    documents = get_documents(os.path.join(args.root, args.dataset), hard_negatives=True, negatives=True)
    for split in ["test", "train", "validation"]:
        gtdata = GTData(os.path.join(args.root, args.dataset, "splits", split))
        data = gtdata.get_grouped_data(os.path.join(args.root, args.dataset), "all",
                                       hn_source="tomt_hn")

        with open(args.file_path + split + "_dpr.json", 'w') as f:
            for record in data:
                neg_records = []
                for neg_record in record["bm25_hn_negatives"]:  # record["bm25_negatives"]:
                    if len(neg_record["text"].split()) < 1:
                        neg_records.append({"title": neg_record["id"], "text": neg_record["title"]})
                    else:
                        neg_records.append({"title": neg_record["id"], "text": neg_record["text"]})
                pos_records = []
                for pos_record in record["positive_documents"]:
                    pos_records.append({"title": pos_record["id"], "text": pos_record["text"]})
                f.write(json.dumps({'_id': record["query"]["id"], 'question': record["query"]["description"],
                                    'neg_paras': neg_records, 'pos_paras': pos_records,
                                    'answers': [record["positive_documents"][0]["title"]]}))
                f.write('\n')
