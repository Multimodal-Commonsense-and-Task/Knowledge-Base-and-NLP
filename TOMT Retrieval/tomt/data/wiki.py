import os
import pickle as pkl
from hashlib import md5
from urllib.parse import urlparse

import requests
import wikipedia as wikipedia_api
from requests import utils as requests_utils
from wikidata.client import Client
from wikidata.entity import Entity, EntityId, EntityState

from tomt.data.imdb import ImdbID

WIKIDATA_CLIENT = Client()
ISBN_10_PROP = "P957"
ISBN_13_PROP = "P212"


def read_wikiplots(path):
    with open(os.path.join(path, "titles")) as reader:
        titles = []
        for title in reader:
            titles.append(title.strip())

    with open(os.path.join(path, "plots")) as reader:
        plots = []
        current_plot = []
        for line in reader:
            if line.strip() == "<EOS>":
                plots.append("\n".join(current_plot))
                current_plot = []
            else:
                current_plot.append(line.strip())

        if len(current_plot) > 0:
            plots.append("\n".join(current_plot))

    assert len(titles) == len(plots)

    return {t: p for (t, p) in zip(titles, plots)}


def extract_wiki_titles(urls):
    titles = set()
    for url in urls:
        url = urlparse(url)
        titles.add(url.path.split("/")[-1])
    return titles


class WikiApi:
    def __init__(self, cache_location, wiki_search_limit):
        self.cache_location = cache_location
        os.makedirs(self.cache_location, exist_ok=True)
        os.makedirs(os.path.join(self.cache_location, "page_failures"), exist_ok=True)
        self.wiki_search_limit = wiki_search_limit

    def get_qids_from_title(self, wiki_title):
        # Given a wikipedia title, this function makes
        # an API call to wikipedia and searches for
        # candidate entities. It then extracts the
        # wikidata-qid from each result and returns it
        payload = {
            "action": "query",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": "1",
            "titles": wiki_title,
            "format": "json"
        }
        r = requests.get(f"https://en.wikipedia.org/w/api.php", payload)
        if r.status_code != 200:
            raise ValueError("Error with HTTP Request!")

        jj = r.json()
        if "query" not in jj:
            return None

        query = jj["query"]
        pages = query.get("pages", dict())

        if len(pages) == 0:
            return None

        qids = []
        for page_id, page in pages.items():
            if "pageprops" not in page:
                continue
            if "wikibase_item" not in page["pageprops"]:
                continue
            rr = {
                "title": page["title"],  # the normalized title,
                "id": page["pageprops"]["wikibase_item"]  # WikiData Q-ID
            }
            qids.append(rr)

        return qids

    def get_entity(self, qid):
        # Returns the wikidata entitiy associated with
        # the given QID
        file_loc = os.path.join(self.cache_location, qid)
        if os.path.exists(file_loc):
            with open(file_loc, "rb") as reader:
                ent_data = pkl.load(reader)
            ent = Entity(EntityId(qid), WIKIDATA_CLIENT)
            ent.data = ent_data
            ent.state = EntityState.loaded
        else:
            ent = WIKIDATA_CLIENT.get(qid, load=True)
            with open(file_loc, "wb") as writer:
                pkl.dump(ent.data, writer)
        return ent

    def write_page(self, page_id, page):
        ppath = os.path.join(self.cache_location, page_id)
        if not os.path.exists(ppath):
            try:
                with open(ppath, "wb") as writer:
                    pkl.dump(page, writer)
            except FileNotFoundError:
                # happens with weird file names
                with open(os.path.join(self.cache_location, md5(page_id.encode('utf-8')).hexdigest()),
                          "wb") as writer:
                    pkl.dump(page, writer)

    def read_page(self, page_id):
        ppath = os.path.join(self.cache_location, page_id)
        if os.path.exists(ppath):
            with open(ppath, "rb") as reader:
                return pkl.load(reader)

        ppath = os.path.join(self.cache_location, md5(page_id.encode('utf-8')).hexdigest())
        if os.path.exists(ppath):
            with open(ppath, "rb") as reader:
                return pkl.load(reader)

        return None

    def write_page_failure(self, page_id, reason):
        ppath = os.path.join(self.cache_location, "page_failures", page_id)
        if not os.path.exists(ppath):
            try:
                with open(ppath, "w") as writer:
                    writer.write(reason)
            except FileNotFoundError:
                # happens with weird file names
                with open(os.path.join(self.cache_location, "page_failures", md5(page_id.encode('utf-8')).hexdigest()),
                          "w") as writer:
                    writer.write(reason)

    def check_page_failure(self, page_id):
        ppath = os.path.join(self.cache_location, "page_failures", page_id)
        if os.path.exists(ppath):
            with open(ppath, "r") as reader:
                return True, reader.read()

        ppath = os.path.join(self.cache_location, "page_failures", md5(page_id.encode('utf-8')).hexdigest())
        if os.path.exists(ppath):
            with open(ppath, "r") as reader:
                return True, reader.read()
        return False, ""

    def delete_page_or_failure(self, page_id):
        ppaths = [os.path.join(self.cache_location, "page_failures", page_id),
                  os.path.join(self.cache_location, page_id)]
        for ppath in ppaths:
            if os.path.exists(ppath):
                os.remove(ppath)

    def get_plot_info_from_wikipedia(self, page_id, keys_to_try=None):
        if keys_to_try is None:
            keys_to_try = ["Plot", "Plot summary"]

        failed, reason = self.check_page_failure(page_id)
        if failed:
            return None, reason

        page = self.read_page(page_id)

        if not page:
            try:
                page = wikipedia_api.page(page_id)
                self.write_page(page_id, page)
            except wikipedia_api.PageError:
                reason = f"Page with page ID '{page_id}' not found"
                self.write_page_failure(page_id, reason)
                return None, reason
            except wikipedia_api.exceptions.DisambiguationError:
                reason = f"Page with page ID '{page_id}' not found was unable to be disambiguated"
                self.write_page_failure(page_id, reason)
                return None, reason

        plot = None
        for key in keys_to_try:
            try:
                if page.section(key) is None:
                    continue
            except KeyError:
                self.delete_page_or_failure(page_id)
                return None, "Page failed, try again"
            plot = page.section(key)
            break

        if not plot:
            reason = f"no plot found for '{page_id}'"
            self.write_page_failure(page_id, reason)
            return plot, reason

        return plot, ""

    def get_isbns(self, entity):

        def _get_prop(item):
            if item["mainsnak"]["snaktype"] == "novalue":
                return None, "mainsnak:novalue"
            else:
                if "datavalue" not in item["mainsnak"]:
                    print(item)
                return item["mainsnak"]["datavalue"]["value"], ""

        def _get_prop_from_claim(claims, prop_name):
            if prop_name not in claims:
                return None, f"prop {prop_name} doesn't exist"

            prop = claims[prop_name]

            if len(prop) == 1:
                prop_value, fail_reason = _get_prop(prop[0])
                if not prop_value:
                    return None, fail_reason
                return prop_value, ""
            elif len(prop) > 1:
                raise ValueError(f"Multiple prop values for {prop}")
            elif len(prop) == 0:
                return None, f"Empty prop {prop}"

        claims = entity.data["claims"]

        isbn10, fail_reason = _get_prop_from_claim(claims, ISBN_10_PROP)
        if not isbn10:
            return (None, None), fail_reason

        isbn13, fail_reason = _get_prop_from_claim(claims, ISBN_13_PROP)
        if not isbn13:
            return (None, None), fail_reason

        return (isbn10, isbn13), ""

    def get_imdb_id(self, entity, imdb_api):
        # Extracts the IMDB ID from the wikidata
        # entity. This is the 'P345' property
        # Returns <imdb id or None>, <empty string if found OR reason for not finding it>

        def _get_from_p345_item(item):
            # input - item from the p345 array
            if item["mainsnak"]["snaktype"] == "novalue":
                return None, "mainsnak:novalue"
            else:
                if "datavalue" not in item["mainsnak"]:
                    print(item)
                return ImdbID(item["mainsnak"]["datavalue"]["value"]), ""

        ll = entity.data["claims"]

        if "P345" not in ll:
            return None, "Doesn't have ImDB property!"

        p345 = ll["P345"]

        imdb_id = None
        reason = ""

        if len(p345) > 1:
            # in some cases (see Q5227700), there are multiple IMDB
            # ids associated with a single wikidata item.

            imdb_ids = set()
            fail_reasons = []
            for l in p345:
                i, fail = _get_from_p345_item(l)
                if i and i.startswith("tt"):
                    imdb_ids.add(ImdbID(i))
                else:
                    fail_reasons.append(fail)

            if len(imdb_ids) == 0:
                return None, "No IMDB Movie Ids found" + "::".join(fail_reasons)

            # see if the IDs resolve to a single one
            imdb_ids = imdb_api.resolve_redirects(imdb_ids)

            if len(imdb_ids) > 1:
                # TODO in other cases, handle this
                raise ValueError(
                    f"Too many IMDB ids for entity {entity.data['id']}: {p345}")

            imdb_id = list(imdb_ids)[0]
        else:
            imdb_id, reason = _get_from_p345_item(p345[0])

        return imdb_id, reason

    def get_wiki_entities_from_imdb(self, imdb_id, movie, imdb_api):
        # Given an IMDB movie entity, this function queries
        # wikipedia to find wikipedia entries with  similar
        # titles. It then retrieves the corresponding entity
        # and checks if the entity has the same ID as the
        # input movie. This is a bit (probably very) roundabout,
        # but the  alternative is to query SPARQL which is very
        # slow!
        # It returns list of ( (title, url, ent), "")  OR (None, "reason")

        # the api returns an empty object if not found
        if len(movie.data) == 0:
            return None, f"Unable to find imdb id {imdb_id}"

        imdb_id = ImdbID(movie.movieID)
        title = movie["title"]

        # now query wikidata to get the wiki title
        wiki_search_url = "https://en.wikipedia.org/w/api.php"
        payload = {
            "action": "opensearch",
            "search": title,
            "limit": self.wiki_search_limit,
            "namespace": 0,
            "format": "json"
        }

        results = requests.get(wiki_search_url, payload)

        if results.status_code != 200:
            raise ValueError(
                f"Recieved a non-200 response for {imdb_id}<>{title}")

        wiki_json = results.json()

        selected_titles = []
        _, res_titles, _, res_links = wiki_json
        for title, url in zip(res_titles, res_links):
            qids = self.get_qids_from_title(title)
            for qid in qids:
                entity = self.get_entity(qid["id"])
                ent_imdb_id, fail_reason = self.get_imdb_id(entity, imdb_api)
                if not ent_imdb_id:
                    continue
                if imdb_id == ent_imdb_id:
                    selected_titles.append((title, url, entity))

        if len(selected_titles) == 0:
            return None, f"Unable to resolve {imdb_id}<>{title}"

        return selected_titles, ""

    def get_wikipedia_url_from_wikidata_id(self, wikidata_id, lang='en'):
        # From https://stackoverflow.com/a/60811917
        # Given a QID, this function queries wikidata
        # and returns a list of URLS corresponding to this
        # particular QID

        url = (
            'https://www.wikidata.org/w/api.php'
            '?action=wbgetentities'
            '&props=sitelinks/urls'
            f'&ids={wikidata_id}'
            '&format=json')

        json_response = requests.get(url).json()

        entities = json_response.get('entities')
        if entities:
            entity = entities.get(wikidata_id)
            if entity:
                sitelinks = entity.get('sitelinks')
                if sitelinks:
                    if lang:
                        # filter only the specified language
                        sitelink = sitelinks.get(f'{lang}wiki')
                        if sitelink:
                            wiki_url = sitelink.get('url')
                            if wiki_url:
                                return requests_utils.unquote(wiki_url)
                    else:
                        # return all of the urls
                        wiki_urls = {}
                        for key, sitelink in sitelinks.items():
                            wiki_url = sitelink.get('url')
                            if wiki_url:
                                wiki_urls[key] = requests_utils.unquote(
                                    wiki_url)
                        return wiki_urls
        return None
