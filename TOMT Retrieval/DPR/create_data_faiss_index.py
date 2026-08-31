import os
import re
import json
from tomt.benchmarks.gt import GTData, get_documents

from create_data_dpr import parser

if __name__ == '__main__':
    args = parser.parse_args()
    documents = get_documents(os.path.join(args.root, args.dataset), hard_negatives=True, negatives=True)

    id2doc = dict()
    for record in documents:
        _input = re.sub(r"(?<![A-Z][a-z])([!?.])(?=\s*[A-Z])\s*", r"\1\n", record['text'])
        id2doc[record["id"]] = {"title": record["id"], "text": record['text'], "sents": _input.split("\n")}

    with open(args.file_path + "id2doc2.json", "w") as f:
        for iid, doc in id2doc.items():
            f.write(json.dumps(doc) + "\n")