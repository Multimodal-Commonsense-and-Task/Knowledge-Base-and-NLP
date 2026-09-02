import os
import sys
import json
import traceback
from urllib.parse import urlparse

import urlextract
from tqdm import tqdm

from tomt.data import wiki
from tomt.data.imdb import IMDBApi, extract_imdb_ids, ImdbID
import argparse


def filter_urls(urls, netloc):
    allowed = set()
    for u in urls:
        up = urlparse(u)
        if netloc in up.netloc:
            allowed.add(u)
    return allowed


class GTResult:
    def __init__(self, imdb_id, wiki_title, wiki_url, plot, wikidata_entity):
        self.imdb_id = imdb_id
        self.plot = plot
        self.wiki_title = wiki_title
        self.wiki_url = wiki_url
        self.wikidata_entity = wikidata_entity

    def _to_json(self):
        imdb_id = self.imdb_id
        if self.imdb_id and isinstance(self.imdb_id, ImdbID):
            imdb_id = self.imdb_id.id
        return {
            "imdb_id": imdb_id,
            "wiki_title": self.wiki_title,
            "wiki_url": self.wiki_url,
            "plot": self.plot,
            "wikidata_entity": self.wikidata_entity.data if self.wikidata_entity else None
        }

    @staticmethod
    def _from_json(j):
        return None

    def __str__(self):
        return str(self._to_json())

    def __repr__(self):
        return self.__str__()


class GTExtractor:

    def __init__(self,
                 gt_entities_folder,
                 wikiplots_path="dataset/wikiplots/",
                 imdb_cache_location="./imdb_cache",
                 wiki_entity_cache_location="./wiki_ent_cache",
                 wiki_search_limit=10):

        os.makedirs(gt_entities_folder, exist_ok=True)

        self.gt_entities_folder = gt_entities_folder
        self.url_extractor = urlextract.URLExtract()

        # read wikiplots data
        self.wikiplots_data = wiki.read_wikiplots(wikiplots_path)
        self.wiki_api = wiki.WikiApi(
            wiki_entity_cache_location, wiki_search_limit)

        self.imdb_api = IMDBApi(imdb_cache_location)

    def get_plot_info(self, title):
        # Given the title of the wiki page, return the plot information from Wikipedia
        plot_info = self.wikiplots_data.get(title)
        return plot_info, "" if plot_info else f"{title} not found"

    def resolve(self, imdb_ids, wikipedia_titles, text):
        # Heursitic for resolving entity.
        # The input to this process is the list of imdb ids found using URLs in the text,
        # wikipedia titles (also extracted from URLs in text), and entities
        # found by a entity linker

        candidates = []
        sources = []
        reason = []
        confident = []

        # first preference to imdb id
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
                        reason.append("")
                        sources.append("imdb")
                        # we are confident here,
                        # since we get it from the ImDB id
                        confident.append(True)
                elif len(movie.data) != 0:
                    # this means that there isn't a corresponding
                    # movie entry in wikipedia, but is present in ImDB
                    # we can now extract the plot information from ImDB
                    # instead of wikipedia, and set the URL and entity to
                    # NULL
                    plot = self.imdb_api.get_plot(movie)
                    # if plot information is unavailable,
                    # skip
                    if not plot:
                        candidates.append(None)
                        reason.append("plot information unavaible in IMDB")
                        sources.append("imdb")
                        confident.append(False)
                    else:
                        # URL and entity unavailable
                        cand = GTResult(i,
                                        movie.data["title"],
                                        None,
                                        plot,
                                        None)
                        candidates.append(cand)
                        reason.append("")
                        sources.append("imdb")
                        confident.append(True)
                else:
                    candidates.append(None)
                    reason.append(fail_str)
                    sources.append("imdb")
                    confident.append(False)

        if wikipedia_titles and len(wikipedia_titles) > 0:
            for title in wikipedia_titles:
                qids = self.wiki_api.get_qids_from_title(title)
                if not qids:
                    candidates.append(None)
                    reason.append("Unable to resolve title->QID")
                    sources.append("wiki")
                    confident.append(False)
                    continue
                plot, _ = self.get_plot_info(title)
                for qid in qids:
                    ent = self.wiki_api.get_entity(qid["id"])
                    imdb_id, fail_str = self.wiki_api.get_imdb_id(
                        ent, self.imdb_api)

                    if imdb_id is None:
                        candidates.append(None)
                        reason.append(fail_str)
                        confident.append(False)
                        sources.append("wiki")
                        continue

                    cand = GTResult(imdb_id,
                                    qid["title"],
                                    self.wiki_api.get_wikipedia_url_from_wikidata_id(
                                        qid),
                                    plot,
                                    ent)
                    candidates.append(cand)
                    reason.append("")
                    sources.append("wiki")
                    # we are confident here,
                    # since we get it from the ImDB id
                    confident.append(True)

        return candidates, reason, confident, sources

    def extract_gt_entity(self, submission_path):
        results = []
        for ut in submission_path:
            # ignore all posts made by the OP
            if ut["is_op"]:
                continue
            text = ut["utterance"]

            # First: Find if there is an IMDB Link or Wikipedia link
            # in the text
            found_urls = self.url_extractor.find_urls(text)

            imdb_ids = None
            wikipedia_titles = None

            if len(found_urls) == 0:
                is_confident = False
            else:
                # see if it's an imdb url
                imdb_urls = filter_urls(found_urls, "imdb")
                if len(imdb_urls) > 0:
                    imdb_ids = extract_imdb_ids(imdb_urls)

                # now see if there is a wikipedia link
                wikipedia_urls = filter_urls(found_urls, "wikipedia")
                if len(wikipedia_urls) > 0:
                    wikipedia_titles = wiki.extract_wiki_titles(wikipedia_urls)

            # resolve the entity, using imdb and or wiki and or entities maybe found in text
            candidates, reason, confident, sources = self.resolve(
                imdb_ids, wikipedia_titles, text)

            for c, r, cc, s in zip(candidates, reason, confident, sources):
                results.append({
                    "entity": c._to_json() if c else None,
                    "reason": r,
                    "confident": cc,
                    "source": s,
                    "uttrance": ut
                })

        return results

    def extract(self, submissions_path):
        # load in the submissions data
        with open(submissions_path) as reader:
            submissions = json.load(reader)

        pbar = tqdm(sorted(submissions.items(),
                           key=lambda _: _[0]), colour="green")
        for sub_id, sub in pbar:
            gt_ent_path = os.path.join(
                self.gt_entities_folder, sub_id + ".json")
            if os.path.exists(gt_ent_path):
                continue

            try:
                gt_entities = self.extract_gt_entity(sub["solved_path"])
                pbar.write(f"Found {len(gt_entities)} for {sub_id}")

                with open(gt_ent_path, "w") as writer:
                    json.dump(gt_entities, writer, indent=1)
            except KeyboardInterrupt:
                pbar.write("Keyboard interrupt!")
                sys.exit(-1)
            except:
                traceback.print_exc()
                pbar.write(
                    f"Some error occured for {sub_id}. Skipping it for now!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MoviesGT")
    parser.add_argument("--input_json", help="the location of the file output by create_solved_cat", required=True)
    parser.add_argument("--ent_folder", help="location to dump entities", required=True)

    parser.add_argument("--imdb_cache", help="location to cache imdb calls", default="./imdb_cache")
    parser.add_argument("--wiki_cache", help="location to cache wikidata/pedia calls", default="./wiki_ent_cache")
    parser.add_argument("--wikiplots_path", help="path to folder containing titles/plots", default="dataset/wikiplots")

    args = parser.parse_args()

    gt_extractor = GTExtractor(gt_entities_folder=args.ent_folder,
                               wikiplots_path=args.wikiplots_path,
                               imdb_cache_location=args.imdb_cache,
                               wiki_entity_cache_location=args.wiki_cache,
                               wiki_search_limit=10)

    gt_extractor.extract(args.input_json)
