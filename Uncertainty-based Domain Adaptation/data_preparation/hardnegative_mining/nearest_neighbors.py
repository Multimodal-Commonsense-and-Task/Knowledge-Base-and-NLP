import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher
# set your custom local cache directory to store pyserini downloads
# os.environ['PYSERINI_CACHE'] = "/local/scratch/guest/pyserini_cache"



filterout_items = ['In each of these examples', 'In each example,', 'Explanation:', 'Example 1:', 'Example 2:', 'In this example', 'Note:',
                   'In the above examples', 'Note that in each example', 'In the first three examples', 'In the first example',
                   'Answer:', 'By analyzing the provided documents', 'Note that the examples', 'In both examples', 'In each case',
                   'Note that the relevant queries', 'In each of the examples']
TOPK = 10


def isEnglish(s):
    return s.isascii()


def get_doc_text(docid, searcher):
    try:
        doctext = searcher.doc(docid).raw()

        doctext_str = doctext.split('"text" : "')[-1].split('"metadata')[0].strip()
        if doctext_str[-2:] == '",':
            doctext_str = doctext_str.replace('",', '').strip()

        doctitle_str = doctext.split('"title" : "')[-1].split('"text')[0].strip()
        if doctitle_str[-2:] == '",':
            doctitle_str = doctitle_str.replace('",', '').strip()

        return doctitle_str + ' ' + doctext_str
    except AttributeError:
        return None


def main_run(dataset_name, generated_queries_filepath, save_nearest_data_filepath, num_nearest=4, contriever_index_path=None, intermediate_dir="."):
    if not contriever_index_path:
        negative_mining_retriever_index_name = f"beir-v1.0.0-{dataset_name}.contriever-msmarco"
    else:
        negative_mining_retriever_index_name = contriever_index_path

    lexical_index_name = f'beir-v1.0.0-{dataset_name}.multifield'


    if intermediate_dir != ".":
        os.makedirs(intermediate_dir, exist_ok=True)
    save_intermediate_topics_filepath = f"{intermediate_dir}/target_queries_{dataset_name}.tsv"
    save_intermediate_dense_results_filepath = f"{intermediate_dir}/target_denseresults_{dataset_name}.txt"


    # ====================================================================================
    #                           1. Load Queries & Docs
    # ====================================================================================
    did_doctext_dict = {}
    for line in open(generated_queries_filepath):
        data = json.loads(line)
        did_doctext_dict[ data['docid'] ] = data['doctext']


    with open(save_intermediate_topics_filepath, 'w') as f:
        for did, dtext in did_doctext_dict.items():
            f.write(f"{did}\t{dtext}\n")


    # ====================================================================================
    #                          # 2. Dense Retrieval
    # ====================================================================================
    python_cmd = f"python -m pyserini.search.faiss \
        --encoder-class contriever \
        --encoder facebook/contriever-msmarco \
        --index {negative_mining_retriever_index_name} \
        --topics {save_intermediate_topics_filepath} \
        --output {save_intermediate_dense_results_filepath} \
        --output-format trec \
        --batch 128 --threads 16 \
        --hits {TOPK}"
    os.system(python_cmd)


    dense_rank_data = defaultdict(list)
    for line in open(save_intermediate_dense_results_filepath):
        qid, _, docid, rank, score, _ = line.strip().split(' ')
        dense_rank_data[qid].append( docid )


    # ====================================================================================
    #                           # 3. Look up DocText
    # ====================================================================================
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)


    # ====================================================================================
    #                           # 4. Assemble Hard Negs
    # ====================================================================================
    did_nearest_dict = defaultdict(list)
    did_nearest_ids_dict = defaultdict(list)
    for _, (did, docids) in enumerate(tqdm(dense_rank_data.items(), total=len(dense_rank_data))):

        nearest_info = []
        for near_docid in [e for e in docids if e != did]:
            near_doctext = get_doc_text( near_docid, searcher=searcher)

            if near_doctext != None:
                nearest_info.append( [near_docid, near_doctext] )
        
        for near_info in nearest_info[:num_nearest]:
            did_nearest_dict[did].append( near_info[1] )
            did_nearest_ids_dict[did].append( near_info[0] )


    # ====================================================================================
    #                           # 5. Prepare Train Data
    # ====================================================================================
    # file_colbert_write = open(save_colbert_traindata_filepath, 'w')
    with open(save_nearest_data_filepath, 'w') as f:
        for _, (did, dtext) in enumerate(did_doctext_dict.items()):
            nearest_docs = did_nearest_dict.get(qid)
            nearest_doc_ids = did_nearest_ids_dict.get(qid)
            if not nearest_docs:
                continue

            entry_dict = {'docid': did, 'doctext': dtext, 'nearest_docs': nearest_docs, 'nearest_doc_ids': nearest_doc_ids}
            json.dump(entry_dict, f)
            f.write('\n')


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Hard Negative Mining')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--sampled_documents_filepath', required=True, type=str,
                        help='file containing sampled documents')
    parser.add_argument('--save_nearest_data_filepath', required=True, type=str,
                        help='file to save nearest data output')
    parser.add_argument('--num_nearest', type=int, default=3, required=False,
                        help='number')
    parser.add_argument('--contriever_index_path', type=str, required=False,
                        help='path to the contriever index')
    parser.add_argument('--intermediate_dir', type=str, required=False,
                        help='directory to save intermediate files')
    args = parser.parse_args()


    main_run(args.dataset_name, args.sampled_documents_filepath, args.save_nearest_data_filepath, args.num_nearest, args.contriever_index_path, args.intermediate_dir)