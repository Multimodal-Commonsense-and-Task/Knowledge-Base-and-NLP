import os
import json
from tomt.benchmarks.gt import GTData, get_documents
from create_data_dpr import parser

if __name__ == '__main__':
    args = parser.parse_args()
    documents = get_documents(os.path.join(args.root, args.dataset), hard_negatives=True, negatives=True)
    split = "test"
    gtdata = GTData(os.path.join(args.root, args.dataset, "splits", split))
    data = gtdata.get_grouped_data(os.path.join(args.root, args.dataset), "all", hn_source="tomt_hn")

    with open(args.file_path + 'qas_test.json', 'w') as f:
        for record in data:
            f.write(json.dumps({'_id': record["query"]["id"],
                                'answer': [record["positive_documents"][0]["title"]],
                                'question': record["query"]["description"],
                                'sp': [i["id"] for i in record["positive_documents"]],
                                }))
            f.write('\n')
