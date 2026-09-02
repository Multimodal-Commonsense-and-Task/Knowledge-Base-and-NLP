from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Iterable, Iterator


def decode_paws_text(value: str) -> str:
    """Decode the byte-literal strings emitted by the official PAWS script."""
    value = value.strip()
    if value.startswith(("b'", 'b"')):
        decoded = ast.literal_eval(value)
        if isinstance(decoded, bytes):
            return decoded.decode("utf-8")
    return value


def read_paws_tsv(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            yield {
                "id": row["id"],
                "q1": decode_paws_text(row["sentence1"]),
                "q2": decode_paws_text(row["sentence2"]),
                "label": int(row["label"]),
            }


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in read_jsonl(path)}


def read_records(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".tsv":
        return read_paws_tsv(path)
    return read_jsonl(path)

