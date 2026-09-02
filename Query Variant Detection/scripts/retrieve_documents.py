#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
import trafilatura

from qvd.io import append_jsonl, completed_ids, read_records


SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


def search(session: requests.Session, query: str, api_key: str, cse_id: str, top_k: int) -> list[dict]:
    response = session.get(
        SEARCH_URL,
        params={"q": query, "key": api_key, "cx": cse_id, "num": top_k},
        timeout=30,
    )
    response.raise_for_status()
    return [
        {
            "rank": rank,
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for rank, item in enumerate(response.json().get("items", []), start=1)
    ]


def add_page_text(session: requests.Session, results: list[dict], max_chars: int) -> list[dict]:
    for result in results:
        try:
            response = session.get(result["link"], timeout=30)
            response.raise_for_status()
            result["text"] = (trafilatura.extract(response.text) or "")[:max_chars]
        except requests.RequestException:
            result["text"] = ""
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect top-k Google CSE results and page text.")
    parser.add_argument("--input", type=Path, required=True, help="PAWS TSV or rewritten JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--max-page-chars", type=int, default=20000)
    parser.add_argument("--no-crawl", action="store_true")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        raise SystemExit("Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID")

    session = requests.Session()
    session.headers["User-Agent"] = "query-variant-detection-repro/0.1"
    done = completed_ids(args.output)
    written = 0
    for record in read_records(args.input):
        if str(record["id"]) in done:
            continue
        for query_key, results_key in (("q1", "q1_results"), ("q2", "q2_results")):
            results = search(session, record[query_key], api_key, cse_id, args.top_k)
            record[results_key] = (
                results if args.no_crawl else add_page_text(session, results, args.max_page_chars)
            )
        append_jsonl(args.output, [record])
        written += 1
        if args.max_records is not None and written >= args.max_records:
            break


if __name__ == "__main__":
    main()

