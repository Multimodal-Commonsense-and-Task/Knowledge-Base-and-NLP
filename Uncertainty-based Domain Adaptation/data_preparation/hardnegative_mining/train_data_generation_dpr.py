import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher

filterout_items = [
    'In each of these examples', 'In each example,', 'Explanation:', 'Example 1:', 
    'Example 2:', 'In this example', 'Note:', 'In the above examples', 
    'Note that the examples', 'In both examples', 'In each case', 
    'Note that the relevant queries', 'In each of the examples'
]
TOPK = 100

def is_english(s):
    # Basic check to ensure string is ASCII
    return s.isascii()

def get_doc_text(docid, searcher):
    """
    Utility to fetch the raw text from a docid in the Lucene index.
    """
    try:
        doctext = searcher.doc(docid).raw()
        # Parse out text and title from the JSON stored in the index
        doctext_str = doctext.split('"text" : "')[-1].split('"metadata')[0].strip()
        if doctext_str.endswith('",'):
            doctext_str = doctext_str[:-2].strip()

        doctitle_str = doctext.split('"title" : "')[-1].split('"text')[0].strip()
        if doctitle_str.endswith('",'):
            doctitle_str = doctitle_str[:-2].strip()

        return (doctitle_str + ' ' + doctext_str).strip()
    except AttributeError:
        return None

def main_run(args):
    dataset_name = args.dataset_name
    generated_queries_filepath = args.generated_queries_filepath
    save_reranker_traindata_filepath = args.save_reranker_traindata_filepath
    save_colbert_traindata_filepath = args.save_colbert_traindata_filepath
    num_pos_to_neg = args.num_pos_to_neg
    contriever_index_path = args.contriever_index_path
    intermediate_dir = args.intermediate_dir

    # If no custom Contriever index path is provided,
    # fallback to Pyserini's default name for the chosen dataset
    if not contriever_index_path:
        negative_mining_retriever_index_name = f"beir-v1.0.0-{dataset_name}.contriever-msmarco"
    else:
        negative_mining_retriever_index_name = contriever_index_path

    lexical_index_name = f'beir-v1.0.0-{dataset_name}.multifield'

    if intermediate_dir is None:
        intermediate_dir = "."
    os.makedirs(intermediate_dir, exist_ok=True)

    save_intermediate_topics_filepath = os.path.join(
        intermediate_dir, f"target_queries_{dataset_name}.tsv"
    )
    save_intermediate_dense_results_filepath = os.path.join(
        intermediate_dir, f"target_denseresults_{dataset_name}.txt"
    )

    # =====================================================================
    # 1. Load the generated queries
    # =====================================================================
    qid_qtext_dict = {}
    qid_doctext_dict = {}
    with open(generated_queries_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            docid = data['docid']
            question = data['question']

            processed_query = question.replace('\n', '').rstrip()
            for item in filterout_items:
                processed_query = processed_query.split(item)[0]
            processed_query = processed_query.strip()

            # Skip empty or non-English
            if not processed_query or not is_english(processed_query):
                continue

            qid_qtext_dict[docid] = processed_query
            qid_doctext_dict[docid] = data['doctext']

    # Write queries to an intermediate TSV (qid, query_text)
    with open(save_intermediate_topics_filepath, 'w', encoding='utf-8') as f:
        for qid, qtext in qid_qtext_dict.items():
            f.write(f"{qid}\t{qtext}\n")

    # =====================================================================
    # 2. Dense retrieval with Contriever
    # =====================================================================
    python_cmd = (
        f"python -m pyserini.search.faiss "
        f"--encoder-class contriever "
        f"--encoder facebook/contriever-msmarco "
        f"--index {negative_mining_retriever_index_name} "
        f"--topics {save_intermediate_topics_filepath} "
        f"--output {save_intermediate_dense_results_filepath} "
        f"--output-format trec "
        f"--batch 128 --threads 16 "
        f"--hits {TOPK} "
        f"--device cuda"
    )
    print(f"Running: {python_cmd}")
    os.system(python_cmd)

    dense_rank_data = defaultdict(list)
    with open(save_intermediate_dense_results_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            dense_rank_data[qid].append(docid)

    # =====================================================================
    # 3. Lookup doc text from the lexical (multifield) index
    # =====================================================================
    print(f"Loading LuceneSearcher for index: {lexical_index_name}")
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    # =====================================================================
    # 4. Assemble Hard Negatives
    # =====================================================================
    qid_hardnegs_dict = defaultdict(list)
    for qid, docids in tqdm(dense_rank_data.items(), total=len(dense_rank_data)):
        # We'll pick from the bottom portion to ensure they are "harder" hits
        neg_doc_info = []
        # For safety, pick from the last 2*num_pos_to_neg documents
        for neg_docid in docids[-(num_pos_to_neg*2):]:
            neg_doctext = get_doc_text(neg_docid, searcher)
            if neg_doctext:
                neg_doc_info.append((neg_docid, neg_doctext))

        # Just pick the first num_pos_to_neg from that subset
        for neg_info in neg_doc_info[:num_pos_to_neg]:
            qid_hardnegs_dict[qid].append(neg_info[1])

    # =====================================================================
    # 5. Write out training data
    # =====================================================================
    print(f"Writing reranker data to: {save_reranker_traindata_filepath}")
    print(f"Writing colbert data to: {save_colbert_traindata_filepath}")

    with open(save_reranker_traindata_filepath, 'w', encoding='utf-8') as f_reranker, \
         open(save_colbert_traindata_filepath, 'w', encoding='utf-8') as f_colbert:

        for qid, qtext in qid_qtext_dict.items():
            doctext = qid_doctext_dict[qid]
            neg_doctext_list = qid_hardnegs_dict.get(qid)

            if not neg_doctext_list:
                continue

            # Positive record
            json.dump([qtext, doctext, 1.0], f_reranker)
            f_reranker.write('\n')

            # Negative records
            for neg_doctext in neg_doctext_list:
                json.dump([qtext, neg_doctext, 0.0], f_reranker)
                f_reranker.write('\n')

                f_colbert.write(f"{qtext}\t{doctext}\t{neg_doctext}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Hard Negative Training Data for DPR')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='BEIR dataset name (e.g. trec-covid)')
    parser.add_argument('--generated_queries_filepath', required=True, type=str,
                        help='File path to the generated synthetic queries in JSONL format')
    parser.add_argument('--save_reranker_traindata_filepath', required=True, type=str,
                        help='Output file path for reranker training data (JSON lines)')
    parser.add_argument('--save_colbert_traindata_filepath', required=True, type=str,
                        help='Output file path for DPR/ColBERT training data (TSV)')
    parser.add_argument('--num_pos_to_neg', type=int, default=4,
                        help='Number of negative passages for each positive')
    parser.add_argument('--contriever_index_path', type=str, required=False, default=None,
                        help='Prebuilt Contriever index name or path (if not using default for dataset)')
    parser.add_argument('--intermediate_dir', type=str, default='.',
                        help='Directory for intermediate query/result files')

    args = parser.parse_args()
    main_run(args)