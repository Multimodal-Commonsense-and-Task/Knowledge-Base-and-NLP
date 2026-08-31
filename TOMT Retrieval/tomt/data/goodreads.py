import json
import logging
import os
import pickle as pkl
import re

import bs4
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options

OPTIONS = Options()
OPTIONS.add_argument('-headless')

ISBN10 = re.compile("^[0-9]{10}$")
ISBN13 = re.compile("^[0-9]{13}$")

log = logging.getLogger(__name__)


def format_isbn(i):
    return re.sub("[^0-9]", "", i)


class GoodreadsApi:
    def __init__(self, geckodriver_path='./geckodriver'):
        self.geckodriver_path = geckodriver_path

    def get(self, url):
        with webdriver.Firefox(executable_path=self.geckodriver_path, options=OPTIONS) as driver:
            try:
                driver.set_page_load_timeout(10)
                driver.get(url)
            except TimeoutException:
                raise ValueError("Timeout occurred")
            page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'lxml')

        desc_container = soup.find(
            name="div", attrs={"id": "descriptionContainer"})
        if desc_container and desc_container.find(name="span"):
            description = desc_container.find(name="span").text
        else:
            # some books have no description - we can find them
            # later with another API
            description = ""

        isbn, isbn13 = None, None

        isbn_div = None
        clear_floats = soup.find_all(
            name="div", attrs={"class": "clearFloats"})
        for clear_float in clear_floats:
            if not "ISBN" in clear_float.text:
                continue
            ibrt = clear_float.find(name="div", attrs={"class": "infoBoxRowTitle"})
            if not ibrt:
                continue
            if "ISBN" == ibrt.text.strip():
                isbn_div = clear_float.find(
                    name="div", attrs={"class": "infoBoxRowItem"})

        if isbn_div:
            for c in isbn_div.children:
                if isinstance(c, bs4.element.NavigableString):
                    c = c.strip()
                    if ISBN10.match(c):
                        isbn = c.strip()
                elif isinstance(c, bs4.element.Tag) and c.attrs["class"] == ["greyText"]:
                    isbn13_tag = c.find(name="span", attrs={
                        "itemprop": "isbn"})
                    if isbn13_tag and ISBN13.match(isbn13_tag.text.strip()):
                        isbn13 = isbn13_tag.text.strip()

        if (not isbn) or (not isbn13):
            return None, f"Unable to find either ISBN10 or ISBN13 for: {url}"

        title_h1 = soup.find(name="h1", attrs={"id": "bookTitle"})
        title = None
        if title_h1:
            title = title_h1.text.strip()

        # get Work ID
        ed_div = soup.find(name="div", attrs={"class": "otherEditionsActions"})
        work_id = None
        if ed_div:
            for link in ed_div.find_all(name="a"):
                if "editions" in link.attrs["href"]:
                    link = link.attrs["href"]
                    work_id = link.split("editions/")[-1]
                    work_id = work_id.split("-")[0]

        return (description, isbn, isbn13, work_id, title), ""


class GoodReadsData:
    # https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/books

    def __init__(self, gr_folder):
        self.graph_file = os.path.join(gr_folder, "goodreads_books.json")
        self.works_file = os.path.join(gr_folder, "goodreads_book_works.json")
        self.gr_folder = gr_folder

        self.isbn10_to_work = {}
        self.isbn13_to_work = {}
        # a work is an abstract concept of a book
        # i.e multiple ISBNs (different editions) can map to the same work
        self.works = {}
        # since the files is massive, this contains
        # the position of the book - as returned by .tell
        # which can then be used to .seek()
        # the format is ISBN13 -> (start_byte, end_byte)
        self.book_index10 = {}
        self.book_index13 = {}

        self._read_data()

    def get_work(self, work_id=None, isbn13=None, isbn10=None):
        if work_id:
            assert isbn13 is None and isbn10 is None, "Provide one arg only"
        if isbn10:
            assert work_id is None and isbn13 is None, "Provide one arg only"
        if isbn13:
            assert work_id is None and isbn10 is None, "Provide one arg only"

        if work_id:
            return self.works.get(work_id)

        if isbn13:
            work_id = self.isbn13_to_work.get(str(isbn13))
            if work_id:
                return self.works.get(work_id)

        if isbn10:
            work_id = self.isbn10_to_work.get(str(isbn10))
            if work_id:
                return self.works.get(work_id)

        return None

    def get_books(self, isbns):
        books = []
        with open(self.graph_file) as reader:
            for i in isbns:
                isbn10, isbn13 = i.get("isbn"), i.get("isbn13")
                pos = None
                if isbn13:
                    pos = self.book_index13.get(isbn13)

                if not pos and isbn10:
                    pos = self.book_index10.get(isbn10)

                if not pos:
                    books.append(None)
                    continue
                start, _ = pos
                reader.seek(start)
                books.append(json.loads(reader.readline()))

        return books

    def _read_data(self):

        pkl_path = os.path.join(self.gr_folder, "cached_pkl")

        if os.path.exists(pkl_path):
            log.info("Loading pickle")
            with open(pkl_path, "rb") as reader:
                dat = pkl.load(reader)

            self.works = dat["works"]
            self.isbn10_to_work = dat["isbn10_to_work"]
            self.isbn13_to_work = dat["isbn13_to_work"]
            self.book_index10 = dat["book_index10"]
            self.book_index13 = dat["book_index13"]
            return

        # read in data, linking books <> works
        with open(self.works_file) as reader:
            for line in reader:
                w = json.loads(line)
                work_id = w["work_id"]
                if work_id is None or len(work_id) == 0:
                    continue
                self.works[work_id] = w

        with open(self.graph_file) as reader:
            line_no = 0
            log.info("Starting to read the BookGraph data")
            missing_works = 0
            while True:
                start_pos = reader.tell()

                line = reader.readline()
                if line is None or len(line) == 0:
                    break

                b = json.loads(line)

                work_id = b.get("work_id")
                if work_id is None or len(work_id) == 0 or work_id not in self.works:
                    missing_works += 1
                    line_no += 1
                    continue

                if "isbns" not in self.works[work_id]:
                    self.works[work_id]["isbns"] = []

                isbn = {}

                end_pos = reader.tell()
                if len(b["isbn"]) > 0:
                    self.isbn10_to_work[str(b["isbn"])] = work_id
                    self.book_index10[b["isbn"]] = (start_pos, end_pos)
                    isbn["isbn"] = str(b["isbn"])
                if len(b["isbn13"]) > 0:
                    self.isbn13_to_work[str(b["isbn13"])] = work_id
                    self.book_index13[b["isbn13"]] = (start_pos, end_pos)
                    isbn["isbn13"] = str(b["isbn13"])

                self.works[work_id]["isbns"].append(isbn)

                line_no += 1

                if line_no % 10000 == 0:
                    log.info(f"\t {line_no} done")

            log.info(f"Missing Works: {missing_works}")
            log.info("Done reading the BookGraph data")

        with open(pkl_path, "wb") as writer:
            pkl.dump({
                "works": self.works,
                "isbn10_to_work": self.isbn10_to_work,
                "isbn13_to_work": self.isbn13_to_work,
                "book_index10": self.book_index10,
                "book_index13": self.book_index13
            }, writer)
