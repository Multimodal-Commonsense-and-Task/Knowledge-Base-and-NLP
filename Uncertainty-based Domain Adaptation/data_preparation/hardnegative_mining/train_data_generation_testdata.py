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
TOPK = 100


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


def main_run(dataset_name, generated_queries_filepath, save_reranker_traindata_filepath, save_colbert_traindata_filepath, num_pos_to_neg=4, contriever_index_path=None, intermediate_dir="."):
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
    qid_qtext_dict = {}
    qid_reldocs_dict = {}
    for line in open(generated_queries_filepath):
        data = json.loads(line)
        question = data['question']

        processed_query = question.replace('\n', '').rstrip()
        for item in filterout_items:
            processed_query = processed_query.split(item)[0]
        processed_query = processed_query.strip()

        if processed_query == '':
            continue
        if not isEnglish(processed_query):
            continue

        qid_qtext_dict[ data['qid'] ] = processed_query
        qid_reldocs_dict[ data['qid'] ] = data['reldocs']


    with open(save_intermediate_topics_filepath, 'w') as f:
        for qid, qtext in qid_qtext_dict.items():
            f.write(f"{qid}\t{qtext}\n")


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
    qid_hardnegs_dict = defaultdict(list)
    qid_pos_dict = defaultdict(list)
    for _, (qid, docids) in enumerate(tqdm(dense_rank_data.items(), total=len(dense_rank_data))):

        pos_docid = [docid for docid in docids if docid not in qid_reldocs_dict[qid]][0]
        pos_doctext = get_doc_text( pos_docid, searcher=searcher)
        qid_pos_dict[qid] = pos_doctext

        neg_doc_info = []
        for neg_docid in docids[len(docids)-num_pos_to_neg*2: ]:

            neg_doctext = get_doc_text( neg_docid, searcher=searcher)

            if neg_doctext != None:
                neg_doc_info.append( [neg_docid, neg_doctext] )

        for neg_info in neg_doc_info[:num_pos_to_neg]:
            qid_hardnegs_dict[qid].append( neg_info[1] )


    # ====================================================================================
    #                           # 5. Prepare Train Data
    # ====================================================================================
    file_colbert_write = open(save_colbert_traindata_filepath, 'w')
    with open(save_reranker_traindata_filepath, 'w') as f:
        for _, (qid, qtext) in enumerate(qid_qtext_dict.items()):

            doctext = qid_pos_dict.get(qid)

            neg_doctext_list = qid_hardnegs_dict.get(qid)
            if not neg_doctext_list:
                continue

            json.dump([qtext, doctext, 1.0], f)
            f.write('\n')

            for neg_doctext in neg_doctext_list:
                json.dump([qtext, neg_doctext, 0.0], f)
                f.write('\n')

                file_colbert_write.write(f"{qtext}\t{doctext}\t{neg_doctext}\n")
    file_colbert_write.close()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Hard Negative Mining')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--generated_queries_filepath', required=True, type=str,
                        help='generated synthetic queries file path')
    parser.add_argument('--save_reranker_traindata_filepath', required=True, type=str,
                        help='file to save training data with hard negatives for reranker')
    parser.add_argument('--save_colbert_traindata_filepath', required=True, type=str,
                        help='file to save training data with hard negatives for colbert')
    parser.add_argument('--num_pos_to_neg', type=int, default=4, required=False,
                        help='ratio of positive to negative pairs in the training data')
    parser.add_argument('--contriever_index_path', type=str, required=False,
                        help='path to the contriever index')
    parser.add_argument('--intermediate_dir', type=str, required=False,
                        help='directory to save intermediate files')
    args = parser.parse_args()


    main_run(args.dataset_name, args.generated_queries_filepath, args.save_reranker_traindata_filepath, args.save_colbert_traindata_filepath, args.num_pos_to_neg, args.contriever_index_path, args.intermediate_dir)