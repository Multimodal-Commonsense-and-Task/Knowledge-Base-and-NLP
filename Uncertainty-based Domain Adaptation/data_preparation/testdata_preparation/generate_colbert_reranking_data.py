import json
import argparse
import numpy as np
from tqdm import tqdm
import csv
import sys
from pyserini import search
from pyserini.search.lucene import LuceneSearcher
from beir.datasets.data_loader import GenericDataLoader

csv.field_size_limit(sys.maxsize)
def tsv_reader(input_filepath):
    reader = csv.reader(open(input_filepath, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for idx, row in enumerate(reader):
        yield idx, row

def main_run(dataset_name, fields, save_testdata_filepath, ranking_tsv_filepath, collection_tsv_filepath):
    lexical_index_name = f'beir-v1.0.0-{dataset_name}.{fields}'
    topics_filename = f"beir-v1.0.0-{dataset_name}-test"
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    topics_dict = search.get_topics(topics_filename)

    results_dict = {}
    inv_map, results = {}, {}
    
    #### Document mappings (from original string to position in tsv file ####
    for idx, row in tsv_reader(collection_tsv_filepath):
        inv_map[str(idx)] = row[0]

    #### Results ####
    for _, row in tsv_reader(ranking_tsv_filepath):
        qid, doc_id, rank = row[0], row[1], int(row[2])
        if qid != inv_map[str(doc_id)]:
            if qid not in results:
                results[qid] = []
            results[qid].append((rank, inv_map[str(doc_id)]))

    for qid in results:
        results_dict[qid] = [docid for _, docid in sorted(results[qid])]

    docid_set = set()
    n_doc_occurance = 0
    for qid, info in results_dict.items():
        for docid in info:
            docid_set.add(docid)
            n_doc_occurance += 1


    docid_doctext_dict = {}
    n_missing_docs = 0
    for docid in tqdm(list(docid_set), total=len(docid_set)):
        try:
            doctext = searcher.doc(docid).raw()
        except AttributeError:
            n_missing_docs += 1
            continue

        # =============================================================================
        doctext_str = doctext.split('"text" : "')[-1].split('"metadata')[0].strip()
        if doctext_str[-2:] == '",':
            doctext_str = doctext_str.replace('",', '').strip()
        # =============================================================================

        doctitle_str = doctext.split('"title" : "')[-1].split('"text')[0].strip()
        if doctitle_str[-2:] == '",':
            doctitle_str = doctitle_str.replace('",', '').strip()

        # =============================================================================
        if fields == "multifield":
            docid_doctext_dict[docid] = doctitle_str + '. ' + doctext_str
        else:
            docid_doctext_dict[docid] = doctext_str

    assert len(docid_set) >= len(docid_doctext_dict)


    test_data = []
    n_doc_occurances_list = []
    for qid, info in results_dict.items():
        try:
            qtext = topics_dict[qid]
        except KeyError:
            qtext = topics_dict[int(qid)]

        docs_info = []
        n_doc = 0
        for docid in info:
            if not docid_doctext_dict.get(docid):
                continue
            doctext = docid_doctext_dict[docid]
            docs_info.append( [docid, doctext] )
            n_doc += 1

        n_doc_occurances_list.append( n_doc )

        test_data.append( {'qid': qid, 'qtext': qtext, 'passages': docs_info} )


    # 7. Save test data
    with open(save_testdata_filepath, 'w') as f:
        json.dump(test_data, f)


    print(f'Number of documents per query: minimum={np.min(n_doc_occurances_list)} || maximum={np.max(n_doc_occurances_list)}')


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Generate test reranking data of BM25')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--case', required=True, type=str,
                        help='case name')
    parser.add_argument('--collection_tsv_filepath', required=True, type=str,
                        help='collection tsv file path' )
    parser.add_argument('--ranking_tsv_filepath', required=True, type=str,
                        help='ranking tsv file path')
    parser.add_argument('--save_testdata_filepath', required=True, type=str,
                        help='file to save test reranking data')
    parser.add_argument('--fields', required=False, default='multifield', type=str,
                        help='whether single field or multifield option in pyserini to include text with title or not')
    args = parser.parse_args()


    main_run(args.dataset_name, args.fields, args.save_testdata_filepath, args.ranking_tsv_filepath, args.collection_tsv_filepath)