import json
import os
import sys
import time
import traceback

import argparse
import urlextract
from tqdm import tqdm

from books_extract_gt import filter_urls, GTResult
from tomt.data import wiki
from tomt.data.goodreads import GoodreadsApi


class BooksNegatives:
    def __init__(self, hn_path, wiki_cache_location,
                 wiki_search_limit):

        self.hn_path = hn_path
        os.makedirs(hn_path, exist_ok=True)
        self.url_extractor = urlextract.URLExtract()
        self.wiki_api = wiki.WikiApi(wiki_cache_location, wiki_search_limit)
        self.goodreads_api = GoodreadsApi()

    def link_data(self, gr_urls, wikipedia_urls):
        candidates = []
        sources = []

        if gr_urls and len(gr_urls) > 0:
            for url in gr_urls:
                try:
                    res, fail_reason = self.goodreads_api.get(url)
                except AttributeError:
                    traceback.print_exc()
                    raise AttributeError()
                except ValueError:
                    traceback.print_exc()
                    raise ValueError()

                if res:
                    (description, isbn, isbn13, work_id, title) = res

                    cand = GTResult(isbn, isbn13, work_id, title,
                                    description, url)
                    candidates.append(cand)
                    sources.append("goodreads_url")

        if wikipedia_urls and len(wikipedia_urls) > 0:
            for url in wikipedia_urls:
                titles = wiki.extract_wiki_titles([url])
                for title in titles:
                    qids = self.wiki_api.get_qids_from_title(title)
                    for qid in qids:
                        entity = self.wiki_api.get_entity(qid["id"])
                        (isbn10, isbn13), fail_reason = self.wiki_api.get_isbns(entity)
                        if not isbn10 or not isbn13:
                            continue

                        plot_info, fail_reason = self.wiki_api.get_plot_info_from_wikipedia(qid["title"])
                        # this will be linked later
                        work_id = None
                        title = None
                        cand = GTResult(isbn10, isbn13, work_id, title, plot_info, url)
                        candidates.append(cand)
                        sources.append("wiki_url")

        return candidates, sources

    def find_negatives(self, submission):
        # forest -> list
        replies = []
        reply_stack = submission["replies"][::]
        while len(reply_stack) > 0:
            reply = reply_stack.pop(0)
            replies.append(reply["body"])
            if reply["replies"] is not None:
                reply_stack.extend(reply["replies"])

        all_candidates = []
        all_sources = []
        for text in replies:
            found_urls = self.url_extractor.find_urls(text)

            gr_urls = None
            wikipedia_urls = None

            if len(found_urls) > 0:
                gr_urls = filter_urls(found_urls, "goodreads")
                wikipedia_urls = filter_urls(found_urls, "wikipedia")

            candidates, sources = self.link_data(gr_urls, wikipedia_urls)

            all_candidates.extend(candidates)
            all_sources.extend(sources)

        negatives = []

        for candidate, source in zip(all_candidates, all_sources):
            negatives.append(candidate._to_json())

        return {
            "negatives": negatives
        }

    def extract(self, submissions_path):
        with open(submissions_path) as reader:
            raw_submissions = json.load(reader)
            submissions = {}
            for sid in raw_submissions:
                submissions[sid] = raw_submissions[sid]["submission"]

        pbar = tqdm(sorted(submissions.items(),
                           key=lambda _: _[0]), colour="green")

        for sub_id, sub in pbar:
            hn_subpath = os.path.join(
                self.hn_path, sub_id + ".json")

            if os.path.exists(hn_subpath):
                pbar.write(f"{sub_id} done.")
                continue
            try:
                hn = self.find_negatives(sub)
                pbar.write(f"Found {len(hn['negatives'])} for {sub_id}")

                with open(hn_subpath, "w") as writer:
                    json.dump(hn, writer, indent=1)
            except KeyboardInterrupt:
                pbar.write("Keyboard interrupt!")
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    pbar.write("Keyboard interrupt! x2... Quitting")
                    # quit on KB interrupt made twice in <5 s
                    sys.exit(-1)
            except:
                traceback.print_exc()
                pbar.write(
                    f"Some error occured for {sub_id}. Skipping it for now!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("BooksNeg")
    parser.add_argument("--input_json", help="the location of the file output by create_solved_cat", required=True)
    parser.add_argument("--neg_ent_folder", help="location to dump entities", required=True)

    parser.add_argument("--wiki_cache", help="location to cache wikidata/pedia calls", default="./wiki_ent_cache")

    args = parser.parse_args()

    neg = BooksNegatives(hn_path=args.neg_ent_folder,
                         wiki_cache_location=args.wiki_cache,
                         wiki_search_limit=10)

    neg.extract(args.input_json)
