#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from openai import OpenAI

from qvd.io import append_jsonl, completed_ids, read_records
from qvd.prompts import REWRITE_INSTRUCTIONS, rewrite_input


def request_rewrite(client: OpenAI, model: str, query: str, retries: int) -> str:
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=REWRITE_INSTRUCTIONS,
                input=rewrite_input(query),
                temperature=0,
                store=False,
            )
            return response.output_text.strip()
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite PAWS questions as concise search queries.")
    parser.add_argument("--input", type=Path, required=True, help="PAWS TSV or JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        first = next(read_records(args.input))
        print(REWRITE_INSTRUCTIONS)
        print(rewrite_input(first["q1"]))
        return

    client = OpenAI()
    done = completed_ids(args.output)
    written = 0
    for record in read_records(args.input):
        if str(record["id"]) in done:
            continue
        record["q1_original"] = record["q1"]
        record["q2_original"] = record["q2"]
        record["q1"] = request_rewrite(client, args.model, record["q1"], args.retries)
        record["q2"] = request_rewrite(client, args.model, record["q2"], args.retries)
        append_jsonl(args.output, [record])
        written += 1
        if args.max_records is not None and written >= args.max_records:
            break


if __name__ == "__main__":
    main()

