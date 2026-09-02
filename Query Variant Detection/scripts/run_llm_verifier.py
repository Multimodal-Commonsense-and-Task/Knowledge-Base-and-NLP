#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from openai import OpenAI

from qvd.io import append_jsonl, completed_ids, read_records
from qvd.prompts import (
    VARIANT_INSTRUCTIONS,
    VARIANT_WITH_EF_INSTRUCTIONS,
    ef_variant_input,
    parse_yes_no,
    plain_variant_input,
)


def request(client: OpenAI, model: str, instructions: str, prompt: str, retries: int) -> str:
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=prompt,
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
    parser = argparse.ArgumentParser(description="Run the paper's LLM-only or LLM+EF prompt.")
    parser.add_argument("--input", type=Path, required=True, help="TSV for plain mode; JSONL for EF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("plain", "ef"), default="plain")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    done = completed_ids(args.output)
    records = read_records(args.input)
    client = None if args.dry_run else OpenAI()
    written = 0
    for record in records:
        if str(record["id"]) in done:
            continue
        if args.mode == "ef":
            instructions = VARIANT_WITH_EF_INSTRUCTIONS
            prompt = ef_variant_input(record)
        else:
            instructions = VARIANT_INSTRUCTIONS
            prompt = plain_variant_input(record["q1"], record["q2"])
        if args.dry_run:
            print(instructions)
            print(prompt)
            return
        response = request(client, args.model, instructions, prompt, args.retries)
        try:
            decision = parse_yes_no(response)
            parse_error = None
        except ValueError as error:
            decision = None
            parse_error = str(error)
        append_jsonl(
            args.output,
            [
                {
                    "id": record["id"],
                    "label": int(record["label"]),
                    "decision": decision,
                    "response": response,
                    "parse_error": parse_error,
                    "model": args.model,
                    "mode": args.mode,
                }
            ],
        )
        written += 1
        if args.max_records is not None and written >= args.max_records:
            break


if __name__ == "__main__":
    main()

