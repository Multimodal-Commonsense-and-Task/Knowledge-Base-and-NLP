import re
import os
from tomt.data import utils as data_utils
import spacy

DISABLE = ["tok2vec", "parser", "ner"]
SPACY_MODEL = "en_core_web_md"

SQ_RE = re.compile(r"\[.*\]")
CACHE_FOLDER = "./dataset/tokenized_data/"
os.makedirs(CACHE_FOLDER, exist_ok=True)


def get_std_utils():
    return Utils(False, False)


class Utils:
    def __init__(self, remove_square_braces=False, incl_only_alphanumeric=False):
        self.remove_square_braces = remove_square_braces
        self.incl_only_alphanumeric = incl_only_alphanumeric
        self.nlp = spacy.load(SPACY_MODEL, exclude=DISABLE)

    def tokenize_with_cache(self, text, id_, lemmatize=True):
        ppath = os.path.join(CACHE_FOLDER,
                             f"{self.remove_square_braces}_{self.incl_only_alphanumeric}_{lemmatize}_{id_}.pkl")

        if os.path.exists(ppath):
            return data_utils.load_pickle(ppath)

        tokens = self.tokenize(text, lemmatize)
        data_utils.write_pickle(tokens, ppath)

        return tokens

    def tokenize(self, text, lemmatize=True):
        if self.remove_square_braces:
            text = SQ_RE.sub("", text)
        text = self.nlp(text)
        toks = []
        for t in text:
            if t.is_stop or t.is_punct:
                continue

            if self.incl_only_alphanumeric and not (t.is_alpha or t.is_digit):
                continue

            toks.append(t.lemma_.lower() if lemmatize else t.text.lower())
        return toks
