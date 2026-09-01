from hamu_tool.dataset import DataLoader

def unseen_ratio(dataset, remove_norel=True):
    loader = DataLoader.load(f'beir/{dataset}')

    qrels_gold = {}
    drels_gold = {}
    for qrel in loader.get_qrels('test'):
        qid = qrel.qid
        did = qrel.did
        if not qid in qrels_gold:
            qrels_gold[qid] = []
        qrels_gold[qid].append(did)
        if not did in drels_gold:
            drels_gold[did] = []
        drels_gold[did].append(qid)

    total_seen_terms = set()
    for doc in loader.get_docs():
        did = doc.id
        if remove_norel and (not did in drels_gold):
            continue
        doc = doc.text
        terms_doc = set(doc.split())
        total_seen_terms.update(terms_doc)

    ratio_seen_list = []
    ratio_unseen_d_list = []
    ratio_unseen_corpus_list = []
    for doc in loader.get_docs():
        did = doc.id
        if not did in drels_gold:
            continue
        doc = doc.text
        terms_doc = set(doc.split())
        qid_gold_list = drels_gold[did]
        terms_query_gold = set()
        for qid_gold in qid_gold_list:
            query_gold = loader.get_query(qid_gold).text
            terms_query_gold.update(query_gold.split())
        terms_seen = terms_query_gold & terms_doc
        terms_seen_corpus = terms_query_gold & total_seen_terms
        terms_unseen_d = terms_query_gold - terms_seen
        terms_unseen_corpus = terms_query_gold - terms_seen_corpus

        ratio_seen = 100 * len(terms_seen) / len(terms_query_gold)
        ratio_unseen_d = 100 * len(terms_unseen_d) / len(terms_query_gold)
        ratio_unseen_corpus = 100 * len(terms_unseen_corpus) / len(terms_query_gold)
        ratio_seen_list.append(ratio_seen)
        ratio_unseen_d_list.append(ratio_unseen_d)
        ratio_unseen_corpus_list.append(ratio_unseen_corpus)

    ratio_seen = sum(ratio_seen_list) / len(ratio_seen_list)
    ratio_unseen_d = sum(ratio_unseen_d_list) / len(ratio_unseen_d_list)
    ratio_unseen_corpus = sum(ratio_unseen_corpus_list) / len(ratio_unseen_corpus_list)
    print(f'{dataset}: seen={ratio_seen:.2f}, unseen_d={ratio_unseen_d:.2f}, unseen_corpus={ratio_unseen_corpus:.2f}')

dataset_list = ['arguana', 'bioasq', 'climate-fever', 'dbpedia', 'fever', 'fiqa', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq', 'quora', 'robust04', 'scidocs', 'scifact', 'signal1m', 'touche', 'touche-v2', 'trec-covid', 'trec-news']

for dataset in dataset_list:
    unseen_ratio(dataset, remove_norel=True)
    unseen_ratio(dataset, remove_norel=False)
