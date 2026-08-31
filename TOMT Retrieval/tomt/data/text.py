import re

INLINE_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
MULT_SPACE = re.compile("\s\s+")

CLEAN_TEXT_RE_LIST = [
    (re.compile("<[a][^>]*>(.+?)</[a]>"), "Link"),
    (re.compile('&gt;'), ""),
    (re.compile('&#x27;'), "'"),
    (re.compile('&quot;'), '"'),
    (re.compile('&#x2F;'), " "),
    (re.compile('<p>'), " "),
    (re.compile('</i>'), ""),
    (re.compile('&#62;'), ""),
    (re.compile('<i>'), " "),
    (re.compile("\n"), ". ")
]


def replace_md_links_with_title(text):
    # This method replaces "<start> [title](url) <end>"
    # occurences in text with "<start> title <end>"

    r = INLINE_LINK_RE.search(text)
    if not r:
        return text

    result = ""
    while r:
        result = result + text[:r.start()] + " " + r.group(1) + " "
        text = text[r.end():]
        r = INLINE_LINK_RE.search(text, r.start() + 1)
    result += text

    return result


def clean_text(text):
    # took this from: https://rileymjones.medium.com/sentiment-anaylsis-with-the-flair-nlp-library-cfe830bfd0f4
    """ Remove hyperlinks and markup """

    text = replace_md_links_with_title(text)

    for r, sub in CLEAN_TEXT_RE_LIST:
        text = r.sub(sub, text)

    # replace multiple spaces with a single one
    return MULT_SPACE.sub(" ", text)
