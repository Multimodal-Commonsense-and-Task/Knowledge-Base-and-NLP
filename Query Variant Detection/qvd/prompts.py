from __future__ import annotations


REWRITE_INSTRUCTIONS = (
    "Rewrite the natural-language question as a concise query someone might type "
    "into a search engine. Preserve the user's original search intent."
)

VARIANT_INSTRUCTIONS = (
    "Treat two queries as equivalent only when their search intent matches and showing "
    "either query's results for the other would leave the user equally satisfied. "
    "Decide whether they are equivalent. End with exactly Yes or No."
)

VARIANT_WITH_EF_INSTRUCTIONS = VARIANT_INSTRUCTIONS + (
    " You will also receive the top search results for both queries. Analyze similarities "
    "and differences in their titles and snippets. Search engines can fail, and ranking "
    "also carries information, so do not treat result overlap as an infallible label."
)


def rewrite_input(query: str) -> str:
    return f"Question: {query}\nAnswer:"


def plain_variant_input(q1: str, q2: str) -> str:
    return f"Query 1: {q1}\nQuery 2: {q2}\nAnswer:"


def _format_results(results: list[dict]) -> str:
    blocks = []
    for rank, result in enumerate(results, start=1):
        title = result.get("title") or ""
        snippet = result.get("snippet") or result.get("text") or ""
        blocks.append(f"Rank {rank}\nTitle: {title}\nSnippet: {snippet}")
    return "\n".join(blocks)


def ef_variant_input(record: dict) -> str:
    return (
        f"Query 1: {record['q1']}\n"
        f"Query 2: {record['q2']}\n\n"
        "[Search results for Query 1]\n"
        f"{_format_results(record['q1_results'])}\n\n"
        "[Search results for Query 2]\n"
        f"{_format_results(record['q2_results'])}\n\n"
        "Would user satisfaction stay the same if these result sets were exchanged?\n"
        "Answer:"
    )


def parse_yes_no(text: str) -> int:
    normalized = text.strip().lower().strip("` .:;!\n\t")
    first = normalized.split(maxsplit=1)[0] if normalized else ""
    if first == "yes":
        return 1
    if first == "no":
        return 0
    raise ValueError(f"Could not parse Yes/No response: {text!r}")
