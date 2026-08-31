import argparse
import json
import logging
import os
import pickle as pkl
import sys
import time
from collections import defaultdict
from tqdm import tqdm
from config import configure_logging, supress_log
from tomt.data.goodreads import GoodReadsData, GoodreadsApi, format_isbn
from tomt.data.imdb import IMDBApi, ImdbID
from tomt.data.wiki import WikiApi
import tomt.data.utils as utils
from tomt.benchmarks.lexical_utils import Utils


def process_movie(entity):
    document_id = entity["imdb_id"]
    assert document_id is not None

    # since we obtain plot info from WikiPlots,
    # it can be incomplete or missing
    # attempt to (a) gather from wikipedia
    # if this fails, use IMDB plot summary
    if entity["plot"] is None:
        # log.info("Geting plot info from wiki")
        plot, fail_reason = wiki.get_plot_info_from_wikipedia(entity["wiki_title"])
        # log.info("Done: Geting plot info from wiki")

        if plot is None:
            # use IMDB plot info
            plot = imdb.get_plot(imdb.get_movie(ImdbID(document_id)))

        if plot is None:
            log.error(f"Unable to get plot for {entity['wiki_url']}, {document_id}")
            raise ValueError(f"Unable to get plot for {entity['wiki_url']}, {document_id}")
        entity["plot"] = plot

    # these properties as 'required'
    for req in {"plot", "imdb_id"}:
        assert entity[req] is not None, entity

    if len(entity["plot"].strip()) == 0:
        log.error(f"Unable to get plot for {entity['wiki_url']}, {document_id}")
        raise ValueError(f"Unable to get plot for {entity['wiki_url']}, {document_id}")

    wikidata = entity.get("wikidata_entity", None)
    if wikidata is not None and "id" in wikidata:
        wikidata_id = wikidata["id"]
    else:
        wikidata_id = None

    if entity["wiki_title"] is not None and len(entity["wiki_title"].strip()) > 0:
        title = entity["wiki_title"]
    else:
        title = None

    if title is None:
        raise ValueError(f"Unable to get title for {entity['wiki_url']}, {document_id}")

    document = {
        "id": document_id,
        "text": entity["plot"],
        "title": title,
        "meta": {
            "wikidata_id": wikidata_id
        }
    }

    return document


def process_book(entity):
    if entity["work_id"] is None:
        isbn10, isbn13 = entity["isbn"], entity["isbn13"]
        isbn10 = format_isbn(isbn10)
        isbn13 = format_isbn(isbn13)
        assert isbn10 is not None and isbn13 is not None

        work = goodreads_data.get_work(isbn13=isbn13)
        if work is None:
            work = goodreads_data.get_work(isbn10=isbn10)
            if work is None:
                # one last attempt to find via URL
                err = f"Unable to link {isbn10}, {isbn13}"
                log.error(err)
                raise ValueError(err)

        entity["work_id"] = work["work_id"]

    document_id = entity["work_id"]
    title = entity["title"]
    text = entity["description"]

    eng_language_codes = {"eng", "en-US", "en-GB"}
    if text is None:
        work = goodreads_data.get_work(work_id=entity["work_id"])

        books = goodreads_data.get_books(work["isbns"])
        text = ""

        find_title = title is None or len(title.strip()) == 0
        for book in books:
            if not book:
                continue
            if find_title and book.get("title") is not None and len(book["title"].strip()) > 0:
                title = book["title"]
                find_title = False

            if book["language_code"] in eng_language_codes and len(book["description"].strip()) > len(text):
                text = book["description"]

    text = text.strip()
    if len(text) == 0:
        err = f"Unable to get description for {entity['work_id']}"
        log.error(err)
        raise ValueError(err)

    if title is None or len(title.strip()) == 0:
        err = f"Unable to get title for {entity['work_id']}"
        log.error(err)
        raise ValueError(err)

    document = {
        "id": document_id,
        "text": text,
        "title": title,
        "meta": {
            "work_id": document_id,
            "url": entity["url"]
        }
    }

    return document


def process(j, config):
    assert config in {"movie", "book"}
    assert j["gold_entity"]["confident"]
    try:
        if config == "movie":
            document = process_movie(j["gold_entity"]["entity"])
        else:
            document = process_book(j["gold_entity"]["entity"])
    except ValueError as e:
        e = str(e)
        if e.startswith("Unable to get title"):
            e = "Unable to get title"
        elif e.startswith("Unable to link"):
            e = "Unable to link"
        elif e.startswith("Unable to get description") or e.startswith("Unable to get plot"):
            e = "Unable to get plot/description"
        else:
            e = str(e)
        ERR_TYPES[e] += 1

        return None, None, None, None

    document_id = document["id"]

    query_id = j["id"]
    qrel = (query_id, 0, document_id, 1)

    n_tokens = len(lex_utils.tokenize(j["description"], lemmatize=True))
    query = {
        "id": query_id,
    }

    return query, n_tokens, document, qrel


def line_count(file):
    c = 0
    with open(file) as reader:
        for _ in reader:
            c += 1
    return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CreateFiles",
                                     description="Splits a raw unprocessed JSON file into QRels, Documents and Queries")
    parser.add_argument("input", help="location of the input JSON file")
    parser.add_argument("output_folder", help="location to dump data")
    parser.add_argument("config", choices={"movie", "book"},
                        help="which config to use (used to figure out which fields to use)")
    parser.add_argument("--negatives", help="location of negatives json", required=True)
    parser.add_argument("--min_length_query", help="minimum length of query (tokens)", default=5, type=int)
    parser.add_argument("--min_length_document", help="minimum length of document (tokens)", default=5, type=int)

    args = parser.parse_args()

    configure_logging("CreateFiles", False)
    supress_log("imdbpy")
    log = logging.getLogger("CreateFiles")
    if args.config == "movie":
        wiki = WikiApi("wiki_ent_cache", 10)
        imdb = IMDBApi("imdb_cache")
    elif args.config == "book":
        goodreads_api = GoodreadsApi()
        goodreads_data = GoodReadsData("./dataset/ucsd_goodreads")

    qrels = []
    documents = {}
    queries = []
    query_docs = defaultdict(list)

    start_time = time.time()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "temp"), exist_ok=True)

    gold_ids = {}
    ERR_TYPES = defaultdict(int)
    log.info(f"Opening: {args.input}")
    lex_utils = Utils(remove_square_braces=True, incl_only_alphanumeric=True)
    with open(args.input) as reader:
        for i, line in enumerate(tqdm(reader, total=line_count(args.input))):
            if i % 1000 == 0:
                log.info(f"Processing: {i}. Obtained {len(queries)}. {round(time.time() - start_time, 2)}s elapsed")
            submission = json.loads(line)

            temp_file_path = os.path.join(args.output_folder, "temp", submission["id"])
            if not os.path.exists(temp_file_path):
                try:
                    query, n_query_tokens, document, qrel = process(submission, args.config)
                except KeyboardInterrupt:
                    log.info("Got a KBInterrupt. Hit Ctrl-C again to quit (within 5 s)")
                    try:
                        time.sleep(5)
                        log.info("Continuing")
                        continue
                    except KeyboardInterrupt:
                        log.info("Quitting")
                        sys.exit(-1)

                with open(temp_file_path, "wb") as w:
                    pkl.dump((query, n_query_tokens, document, qrel), w)
            else:
                with open(temp_file_path, "rb") as r:
                    (query, n_query_tokens, document, qrel) = pkl.load(r)

            if not all((query, document, qrel)):
                continue

            if n_query_tokens < args.min_length_query:
                log.info(f"Too few tokens for query: {submission['id']}")
                continue

            document_text = document["text"]
            doc_tokens = lex_utils.tokenize(document_text, lemmatize=True)
            if len(doc_tokens) < args.min_length_document:
                log.info(f"Too few tokens for query: {document_text}")
                continue

            queries.append(query)
            qrels.append(qrel)
            documents[document["id"]] = document
            gold_ids[query["id"]] = document["id"]

            query_docs[document["id"]].append((query["id"], query))

    os.makedirs(args.output_folder, exist_ok=True)

    with open(os.path.join(args.output_folder, "qrels.txt"), "w") as writer:
        for (q, it, doc, rel) in qrels:
            writer.write(f"{q}\t{it}\t{doc}\t{rel}\n")

    req_fields_doc = {"id", "text"}
    with open(os.path.join(args.output_folder, "documents.json"), "w") as writer:
        for doc in documents.values():
            for req in req_fields_doc:
                assert req in doc and doc[req] is not None
            writer.write(json.dumps(doc) + "\n")

    req_fields_query = {"id"}
    with open(os.path.join(args.output_folder, "raw_queries.json"), "w") as writer:
        for q in queries:
            for req in req_fields_query:
                assert req in q and q[req] is not None
            writer.write(json.dumps(q) + "\n")

    log.info(f"Wrote {len(queries)} queries, {len(documents)} documents, {len(qrels)} qrels")

    log.info("Gathering Negatives + other candidates")
    all_negatives = []
    all_negative_ids = set()
    hard_negative_ids = set()
    sub_negative = defaultdict(set)
    n_hard_negatives = 0

    os.makedirs(os.path.join(args.output_folder, "temp_neg"), exist_ok=True)
    for file_name in os.listdir(os.path.join(args.output_folder, "temp_neg")):
        if file_name.startswith("."):
            continue
        sub_id = file_name.split(".")[0]
        j = utils.read_json(os.path.join(args.negatives, file_name))

        gold_id = gold_ids.get(sub_id)
        if args.config == "movie" and gold_id:
            gold_id = ImdbID(gold_id)

        hard_negatives = False
        key_id = "imdb_id" if args.config == "movie" else "isbn13"
        for neg_j in j["negatives"]:
            if args.config == "movie":
                key = neg_j["imdb_id"]
            else:
                key = format_isbn(neg_j["isbn13"])

            temp_file_path = os.path.join(args.output_folder, "temp_neg", key)

            if not os.path.exists(temp_file_path):
                try:
                    neg = {
                        "movie": process_movie,
                        "book": process_book
                    }[args.config](neg_j)
                    with open(temp_file_path, "wb") as w:
                        pkl.dump(neg, w)
                except ValueError:
                    continue
            else:
                with open(temp_file_path, "rb") as r:
                    neg = pkl.load(r)

            # don't add the 'gold' answer to the
            # set of negatives!
            if gold_id and neg["id"] == gold_id:
                continue

            all_negatives.append(neg)
            all_negative_ids.add(neg["id"])
            sub_negative[sub_id].add(neg["id"])

        if gold_id and len(sub_negative[sub_id]) > 0:
            # since for this submission we know the
            # answer, other answers found in this thread can be
            # considered 'hard' negatives
            assert gold_id not in sub_negative[sub_id]
            hard_negative_ids.update(sub_negative[sub_id])
            n_hard_negatives += len(sub_negative[sub_id])
            log.info(f"Found {len(sub_negative[sub_id])} Hard Negative(s) for {sub_id}")

    log.info(f"Found a total of {n_hard_negatives} Hard Negatives")
    pos_ids = set(documents.keys())
    # remove hard_negatives from the list of negatives
    final_negative_ids = all_negative_ids - hard_negative_ids - pos_ids
    # remove positive docs from list of hard negatives
    final_hard_negative_ids = hard_negative_ids - pos_ids

    final_hard_negatives = {d["id"]: d for d in all_negatives if d["id"] in final_hard_negative_ids}
    final_negatives = {d["id"]: d for d in all_negatives if d["id"] in final_negative_ids}

    with open(os.path.join(args.output_folder, "other_candidates.json"), "w") as writer:
        for doc in final_negatives.values():
            for req in req_fields_doc:
                assert req in doc and doc[req] is not None
            writer.write(json.dumps(doc) + "\n")

    with open(os.path.join(args.output_folder, "negative_documents.json"), "w") as writer:
        for doc in final_hard_negatives.values():
            for req in req_fields_doc:
                assert req in doc and doc[req] is not None
            writer.write(json.dumps(doc) + "\n")

    sub_negative = {k: list(v) for (k, v) in sub_negative.items()}
    utils.write_json(sub_negative, os.path.join(args.output_folder, "neg_doc_ids.json"))

    # shutil.rmtree(os.path.join(args.output_folder, "temp"))
    # shutil.rmtree(os.path.join(args.output_folder, "temp_neg"))

    log.info("Error counts")
    for k, v in sorted(ERR_TYPES.items(), key=lambda _: -_[1]):
        log.info(f"Error Count:: {k}: {v}")
