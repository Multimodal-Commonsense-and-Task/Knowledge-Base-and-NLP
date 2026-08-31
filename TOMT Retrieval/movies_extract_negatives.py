import json
import os
import sys
import time
import traceback
import argparse

import urlextract
from tqdm import tqdm

from movies_extract_gt import filter_urls, GTResult
from tomt.data import wiki
from tomt.data.imdb import IMDBApi, extract_imdb_ids


class MoviesNegatives:
    def __init__(self, hn_path, imdb_cache_location, wikiplots_path, wiki_cache_location,
                 wiki_search_limit):

        self.hn_path = hn_path
        self.wikiplots_data = wiki.read_wikiplots(wikiplots_path)
        os.makedirs(hn_path, exist_ok=True)
        self.url_extractor = urlextract.URLExtract()
        self.imdb_api = IMDBApi(imdb_cache_location)
        self.wiki_api = wiki.WikiApi(wiki_cache_location, wiki_search_limit)

    def get_plot_info(self, title):
        # Given the title of the wiki page, return the plot information from Wikipedia
        plot_info = self.wikiplots_data.get(title)
        return plot_info, "" if plot_info else f"{title} not found"

    def link_data(self, imdb_ids, wikipedia_titles):
        candidates = []
        sources = []
        if imdb_ids and len(imdb_ids) > 0:
            # for each imdb id found in the reply,
            # generate candidates
            for i in imdb_ids:
                movie = self.imdb_api.get_movie(i)
                results, fail_str = self.wiki_api.get_wiki_entities_from_imdb(
                    i, movie, self.imdb_api)
                if results:
                    for title, url, entity in results:
                        plot, _ = self.get_plot_info(title=title)
                        cand = GTResult(i,
                                        title,
                                        url,
                                        plot,
                                        entity)
                        candidates.append(cand)
                        sources.append("imdb")
                elif len(movie.data) != 0:
                    # this means that there isn't a corresponding
                    # movie entry in wikipedia, but is present in ImDB
                    # we can now extract the plot information from ImDB
                    # instead of wikipedia, and set the URL and entity to
                    # NULL
                    plot = self.imdb_api.get_plot(movie)
                    # if plot information is unavailable,
                    # skip
                    if plot:
                        # URL and entity unavailable
                        cand = GTResult(i,
                                        movie.data["title"],
                                        None,
                                        plot,
                                        None)
                        candidates.append(cand)
                        sources.append("imdb")

        if wikipedia_titles and len(wikipedia_titles) > 0:
            for title in wikipedia_titles:
                qids = self.wiki_api.get_qids_from_title(title)
                if not qids:
                    continue
                plot, _ = self.get_plot_info(title)
                for qid in qids:
                    ent = self.wiki_api.get_entity(qid["id"])
                    imdb_id, fail_str = self.wiki_api.get_imdb_id(
                        ent, self.imdb_api)

                    if imdb_id is None:
                        continue

                    cand = GTResult(imdb_id,
                                    qid["title"],
                                    self.wiki_api.get_wikipedia_url_from_wikidata_id(
                                        qid),
                                    plot,
                                    ent)
                    candidates.append(cand)
                    sources.append("wiki")
                    # we are confident here,
                    # since we get it from the ImDB id

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

            imdb_ids = None
            wikipedia_titles = None

            if len(found_urls) == 0:
                continue
            else:
                # see if it's an imdb url
                imdb_urls = filter_urls(found_urls, "imdb")
                if len(imdb_urls) > 0:
                    imdb_ids = extract_imdb_ids(imdb_urls)

                # now see if there is a wikipedia link
                wikipedia_urls = filter_urls(found_urls, "wikipedia")
                if len(wikipedia_urls) > 0:
                    wikipedia_titles = wiki.extract_wiki_titles(wikipedia_urls)

            candidates, sources = self.link_data(imdb_ids, wikipedia_titles)

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
    parser = argparse.ArgumentParser("MoviesNeg")
    parser.add_argument("--input_json", help="the location of the file output by create_solved_cat", required=True)
    parser.add_argument("--neg_ent_folder", help="location to dump entities", required=True)

    parser.add_argument("--imdb_cache", help="location to cache imdb calls", default="./imdb_cache")
    parser.add_argument("--wiki_cache", help="location to cache wikidata/pedia calls", default="./wiki_ent_cache")
    parser.add_argument("--wikiplots_path", help="path to folder containing titles/plots", default="dataset/wikiplots")

    args = parser.parse_args()

    neg = MoviesNegatives(hn_path=args.neg_ent_folder,
                          wikiplots_path=args.wikiplots_path,
                          imdb_cache_location=args.imdb_cache,
                          wiki_cache_location=args.wiki_cache,
                          wiki_search_limit=10)

    neg.extract(args.input_json)
