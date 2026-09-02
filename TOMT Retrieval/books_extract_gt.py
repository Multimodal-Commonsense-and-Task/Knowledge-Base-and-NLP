import json
import os
import sys
import traceback
import argparse
import urlextract
from tqdm import tqdm
from movies_extract_gt import filter_urls
from tomt.data.goodreads import GoodreadsApi
from tomt.data.wiki import WikiApi, extract_wiki_titles


class GTResult:
    def __init__(self, isbn, isbn13, work_id, title, description, url):
        self.isbn = isbn
        self.isbn13 = isbn13
        self.work_id = work_id
        self.description = description
        self.url = url
        self.title = title

    def _to_json(self):
        return {
            "isbn": self.isbn,
            "isbn13": self.isbn13,
            "description": self.description,
            "url": self.url,
            "work_id": self.work_id,
            "title": self.title
        }


class GTExtractor:
    def __init__(self, gt_entities_folder, wiki_cache, wiki_search_limit):
        self.gt_entities_folder = gt_entities_folder
        self.goodreads_api = GoodreadsApi()
        self.wiki_api = WikiApi(wiki_cache, wiki_search_limit)
        self.url_extractor = urlextract.URLExtract()

        os.makedirs(self.gt_entities_folder, exist_ok=True)

    def resolve(self, gr_urls, wikipedia_urls, text):
        candidates = []
        sources = []
        reason = []
        confident = []

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

                if not res:
                    candidates.append(None)
                    sources.append("goodreads_url")
                    reason.append(fail_reason)
                    confident.append(False)
                else:
                    (description, isbn, isbn13, work_id, title) = res

                    cand = GTResult(isbn, isbn13, work_id, title,
                                    description, url)
                    candidates.append(cand)
                    sources.append("goodreads_url")
                    reason.append("")
                    confident.append(True)

        if wikipedia_urls and len(wikipedia_urls) > 0:
            for url in wikipedia_urls:
                titles = extract_wiki_titles([url])
                for title in titles:
                    qids = self.wiki_api.get_qids_from_title(title)
                    for qid in qids:
                        entity = self.wiki_api.get_entity(qid["id"])
                        (isbn10, isbn13), fail_reason = self.wiki_api.get_isbns(entity)
                        if not isbn10 or not isbn13:
                            candidates.append(None)
                            sources.append("wiki_url")
                            reason.append(fail_reason)
                            confident.append(False)
                            continue

                        plot_info, fail_reason = self.wiki_api.get_plot_info_from_wikipedia(qid["title"])
                        # this will be linked later
                        work_id = None
                        title = None
                        cand = GTResult(isbn10, isbn13, work_id, title, plot_info, url)
                        candidates.append(cand)
                        sources.append("wiki_url")
                        reason.append("")
                        confident.append(True)

        return candidates, reason, confident, sources

    def extract_gt_entity(self, solved_path):
        results = []
        for ut in solved_path:
            # ignore all posts made by the OP
            if ut["is_op"]:
                continue
            text = ut["utterance"]

            # First: Find if there is an IMDB Link or Wikipedia link
            # in the text
            found_urls = self.url_extractor.find_urls(text)

            gr_urls = None
            wikipedia_urls = None

            if len(found_urls) == 0:
                is_confident = False
            else:
                # see if it's an imdb url
                gr_urls = filter_urls(found_urls, "goodreads")

                # now see if there is a wikipedia link
                wikipedia_urls = filter_urls(found_urls, "wikipedia")

            # resolve the entity, using imdb and or wiki and or entities maybe found in text
            candidates, reason, confident, sources = self.resolve(
                gr_urls, wikipedia_urls, text)

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
            except NotImplementedError:
                pass
            except:
                traceback.print_exc()
                pbar.write(
                    f"Some error occured for {sub_id}. Skipping it for now!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BooksGT")
    parser.add_argument("--input_json", help="the location of the file output by create_solved_cat", required=True)
    parser.add_argument("--ent_folder", help="location to dump entities", required=True)
    parser.add_argument("--wiki_cache", help="location to cache wikidata/pedia calls", default="./wiki_ent_cache")

    args = parser.parse_args()
    gt_extractor = GTExtractor(gt_entities_folder=args.ent_folder,
                               wiki_cache="./wiki_ent_cache",
                               wiki_search_limit=10)
    gt_extractor.extract(args.input_json)
