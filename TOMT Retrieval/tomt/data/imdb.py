import re
import os
import pickle as pkl
from urllib.parse import urlparse

import imdb

imdb_id_re = re.compile("^tt[0-9]+")


def extract_imdb_ids(urls):
    ids = set()
    for url in urls:
        url = urlparse(url)
        for p in url.path.split("/"):
            match = imdb_id_re.match(p)
            if match:
                ids.add(ImdbID(match.string))
    return ids


class IMDBApi:
    def __init__(self, imdb_cache_location):
        os.makedirs(imdb_cache_location, exist_ok=True)
        self.imdb_cache_location = imdb_cache_location
        self.ia = imdb.IMDb()

    def get_plot(self, movie):
        plots = movie.data.get("plot", [])
        if len(plots) == 0:
            return None
        return "\n\n".join(plots)

    def get_movie(self, imdb_id):
        # returns either: None, "<reason for failure>"
        # or (list of (title, links), "")
        if isinstance(imdb_id, ImdbID):
            imdb_id = imdb_id.id
        else:
            raise ValueError(f"Invalid type: {type(imdb_id)}")

        file_loc = os.path.join(self.imdb_cache_location, imdb_id)
        if os.path.exists(file_loc):
            with open(file_loc, "rb") as reader:
                movie = pkl.load(reader)
        else:
            movie = self.ia.get_movie(imdb_id[2:])
            with open(file_loc, "wb") as writer:
                pkl.dump(movie, writer)

        return movie

    def resolve_redirects(self, imdb_ids):
        resolved = set()
        for imdb_id in imdb_ids:
            mov = self.get_movie(imdb_id)
            # if we query with one (deprecated/duplicate) ID
            # and it redirects it to the latest ID
            # mov.movieID -> query ID
            # mov.data["imdbID"] -> new ID
            resolved.add(ImdbID(mov.data["imdbID"]))

        return resolved


class ImdbID:
    IMDB_RE = re.compile("^[a-z]{2}[0-9]+", flags=re.IGNORECASE)

    def __init__(self, imdb_id):
        if isinstance(imdb_id, int):
            self.id = f"tt{imdb_id}"
        elif isinstance(imdb_id, str) and self.IMDB_RE.fullmatch(imdb_id):
            self.id = imdb_id.lower()
        elif isinstance(imdb_id, str) and re.fullmatch("^[0-9]+", imdb_id):
            self.id = f"tt{imdb_id}"
        elif isinstance(imdb_id, ImdbID):
            self.id = imdb_id.id
        else:
            raise ValueError(f"Invalid IMDB id {imdb_id}")

    def __eq__(self, obj):
        if isinstance(obj, int) or isinstance(obj, str):
            obj = ImdbID(obj)
        elif isinstance(obj, ImdbID):
            pass
        elif obj is None:
            return False
        else:
            raise ValueError(f"Unsupported comparision {self}<>{obj}")
        return self.id == obj.id

    def __ne__(self, obj):
        return not self == obj

    def __str__(self):
        return self.id

    def __repr__(self):
        return f"ImdbID({self.id})"

    def __hash__(self):
        return hash(self.id)

    def startswith(self, sub):
        return self.id.startswith(sub)
