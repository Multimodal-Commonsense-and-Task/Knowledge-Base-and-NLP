import os
import argparse

import spacy

from tomt.benchmarks.gt import read_qrels
from tomt.benchmarks.lexical_utils import Utils
import shutil
from tomt.data import utils

import urlextract


def update_qrels_docs(queries_path, qrels_path, docs_path=None):
    queries = utils.read_jsonl(queries_path)
    qrels = utils.read_qrels(qrels_path)

    qids = {q["id"] for q in queries}

    upd = {}
    docids = set()
    for qid, v in qrels.items():
        if qid in qids:
            upd[qid] = (list(v.keys())[0])
            docids.add((list(v.keys())[0]))

    print(len(queries), len(upd))
    assert len(queries) == len(upd)

    with open(qrels_path, "w") as writer:
        for qid, doc_id in upd.items():
            writer.write(f"{qid}\t{0}\t{doc_id}\t{1}\n")

    if docs_path:
        upd_docs = []
        for doc in utils.read_jsonl(docs_path):
            if doc["id"] in docids:
                upd_docs.append(doc)

        print(len(upd_docs), len(docids))
        assert len(upd_docs) == len(docids)

        utils.write_jsonl(upd_docs, docs_path)


def strip_urls(desc):
    urls = url_extractor.find_urls(desc, get_indices=True)
    # there are no URLS found in the text
    if len(urls) == 0:
        return desc
    # there is atleast one
    urls.sort(key=lambda _: _[0][0])
    # build new desc with the URLs removed
    new_desc = ""
    prev_end = 0
    for u, (begin, end) in urls:
        new_desc = new_desc + " " + desc[prev_end: begin]
        prev_end = end

    return new_desc


def is_edited(submission_id, sub_folders):
    for fold in sub_folders:
        if os.path.exists(os.path.join(fold, submission_id + ".pkl")):
            return utils.load_pickle(os.path.join(fold, submission_id + ".pkl")).edited is not False

    raise ValueError(f"submission {submission_id} not found in {sub_folders}")


def remove_tokens(sents, gold_title_toks, sent_pos):
    sent = sents[sent_pos]

    already_removed = set()
    new_sent = ""
    for org_tok in sent:
        if org_tok.is_punct:
            new_sent += org_tok.text + " "
            continue

        tok = org_tok.text.lower()

        skip = False
        for g in gold_title_toks:
            if g.is_punct:
                continue

            if tok == g.text.lower() and tok not in already_removed:
                skip = True
                already_removed.add(tok)
                break

        if not skip:
            new_sent += org_tok.text + " "

    new_sent = new_sent.strip()
    # print(sent, new_sent)
    new_desc = ""
    for pos, sent in enumerate(sents):
        if pos == sent_pos:
            new_desc += new_sent + ". "
        else:
            new_desc += sent.text + ". "

    # print(new_desc.rstrip().rstrip("."))

    return new_desc


def has_answer_in_text(query, gold_doc, sub_folders):
    # overlap = % [0, 1] overlap
    # n_last -> number of sentences to consider from the last
    def has_title_tokens(text, overlap=0.8, n_last=None, n_first=None):

        if n_last is not None:
            assert n_first is None
            sents = [_ for _ in sent_nlp(text).sents]
            if n_last < len(sents) - 1:
                n_last = 1
            sents = sents[len(sents) - n_last:]
            text = ""
            for sent in sents:
                text = sent.text + ". "
        if n_first is not None:
            assert n_last is None
            sents = [_ for _ in sent_nlp(text).sents]
            if n_first > len(sents):
                n_first = len(sents)
            sents = sents[:n_first]
            text = ""
            for sent in sents:
                text = sent.text + ". "

        text = text.lower()
        n = 0
        for g in gold_title_toks:
            if g.is_punct or g.is_stop:
                continue
            if g.text.lower() in text:
                n += 1

        if (n / len(gold_title_toks)) > overlap:
            return True

        return False

    qid = query['id']

    if not is_edited(qid, sub_folders):
        return False, query["description"]

    # title of gold doc
    gold_title = gold_doc["title"]

    gold_title_toks = sent_nlp(gold_title)
    desc = query["description"]
    if not has_title_tokens(desc):
        return False, query["description"]

    doc = sent_nlp(desc)
    assert doc.has_annotation("SENT_START")
    sents = [_ for _ in doc.sents]

    # edits are at the beginning or end
    if has_title_tokens(desc, overlap=0.5, n_first=1):
        # at the beginning
        new_desc = remove_tokens(sents, gold_title_toks, 0)
    else:
        # at the end
        # attempt to remove tokens from last sentence
        new_desc = remove_tokens(sents, gold_title_toks, len(sents) - 1)

    # spacy sometimes splits the sentences wrong
    # so if the title tokens still exist after editing
    # the last sentence, then find the
    # position of certain key words that indicate 'solved'
    # and then try removing it *from* that position onwards
    if has_title_tokens(new_desc):
        # attempt to remove sentence after 'Edit' or similar
        last_index = -1
        for _ in {"edit", "solution", "update", "solved"}:
            i = desc.lower().rfind(_)
            if i > last_index:
                last_index = i
        if last_index != -1:
            new_desc = desc[:last_index]

    # if both approaches fail, raise an Error
    if has_title_tokens(new_desc, n_last=2):
        raise ValueError(f"unable to fix {gold_title_toks}:: {desc}")

    return True, new_desc


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Clean", description="Cleans up queries")
    parser.add_argument("folder", help="(root) location of data to clean up")
    parser.add_argument("--min_len", help="min length of query", type=int, default=2)
    parser.add_argument("--verbose", help="set flag for verbose logging", action="store_true")
    parser.add_argument("--sub_folders", help="csv of submission folders (which contain pickles of submissions)",
                        type=str, required=True)
    args = parser.parse_args()

    sub_folders = args.sub_folders.split(",")
    SEP_ = shutil.get_terminal_size((50, 20)).columns

    url_extractor = urlextract.URLExtract()

    folder = args.folder
    min_length = args.min_len
    lex_utils = Utils(remove_square_braces=True, incl_only_alphanumeric=True)
    sent_nlp = spacy.load("en_core_web_md")

    removed_qids = []
    count = 0
    all_queries = {}

    documents = utils.read_jsonl(os.path.join(folder, "documents.json"))
    documents = {d["id"]: d for d in documents}
    qrels = read_qrels(os.path.join(folder, "qrels.txt"), False)

    for split in ["test", "train", "validation"]:
        queries_path = os.path.join(folder, "splits", split, "queries.json")
        queries = utils.read_jsonl(queries_path)
        fixed_queries = []

        for q in queries:

            q["raw_description"] = q["description"]
            # answer in text -> remove answer
            try:
                has_ans, fixed_desc = has_answer_in_text(q, documents[qrels[q["id"]]], sub_folders)
            except ValueError as v:
                print(f"Skipping {q['id']}")
                removed_qids.append(q["id"])
                continue

            if has_ans:
                if args.verbose:
                    od = q['description'].replace('\n', ' ')
                    fd = fixed_desc.replace('\n', ' ')
                    print(f"{q['id']} had answer in text:\n>>>>>>>Original\n\n:{od}\n\n")
                    print(f" >>>>>>>After removal\n\n: {fd}\n" + ("#" * SEP_) + "\n\n")

                q["description"] = fixed_desc

            toks = lex_utils.tokenize(q["raw_description"], lemmatize=True)

            if len(toks) == 0:
                removed_qids.append(q['id'])
                continue
            elif len(toks) < min_length:

                desc = strip_urls(q["raw_description"])

                if len(desc) == 0:
                    removed_qids.append(q["id"])
                    continue

                toks = lex_utils.tokenize(desc, lemmatize=True)

                if len(toks) < min_length:
                    removed_qids.append(q["id"])
                    continue
                else:
                    fixed_queries.append(q)

            else:
                fixed_queries.append(q)

        print(split, len(queries) - len(fixed_queries), "removed")
        count += len(fixed_queries)

        for q in fixed_queries:
            assert q["id"] not in all_queries
            all_queries[q["id"]] = q

        # write queries for this split only
        # the 'global' queries will be updated at the end of this loop
        # the qrels / docs will also be updated at the end
        utils.write_jsonl(fixed_queries, queries_path)

    all_fixed = []
    for qid, q in all_queries.items():
        if qid in removed_qids:
            d = q['raw_description'].replace('\n', ' ')
            if args.verbose:
                print(f"removing query {q['id']} with desc: {d}")
                print()
            continue

        all_fixed.append(q)

    print(count, len(all_fixed), len(removed_qids), "\nRemoved QIDs:\n", removed_qids)
    assert count == len(all_fixed)

    utils.write_jsonl(all_fixed, os.path.join(folder, "queries.json"))

    update_qrels_docs(f"{folder}/queries.json",
                      f"{folder}/qrels.txt",
                      f"{folder}/documents.json")
    for split in ["test", "train", "validation"]:
        update_qrels_docs(f"{folder}/splits/{split}/queries.json",
                          f"{folder}/splits/{split}/qrels.txt")
