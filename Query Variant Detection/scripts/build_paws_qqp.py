#!/usr/bin/env python3
# Adapted from google-research-datasets/paws/qqp_generate_data.py.
# The upstream file is licensed under Apache License 2.0.
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import nltk


def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text)


def read_original_qqp(path: Path) -> dict[int, list[str]]:
    fieldnames = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"]
    questions: dict[int, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, fieldnames=fieldnames, delimiter="\t")
        next(reader)
        for row in reader:
            for qid_key, question_key in (("qid1", "question1"), ("qid2", "question2")):
                qid = int(row[qid_key])
                questions.setdefault(qid, tokenize(row[question_key]))
    return questions


def get_token(qid: int, index: int, questions: dict[int, list[str]]) -> str:
    try:
        return questions[qid][index]
    except (KeyError, IndexError) as error:
        raise ValueError(f"Cannot resolve qid={qid}, token={index}") from error


def build_sentence(specification: str, qid: int, questions: dict[int, list[str]]) -> str:
    output = []
    for index in specification.split("/"):
        if not index.startswith("("):
            output.append(get_token(qid, int(index), questions))
            continue
        merged = ""
        for sub_index in index.split(":"):
            source_qid, source_index = ast.literal_eval(sub_index)
            merged += get_token(source_qid, source_index, questions)
        output.append(merged)
    return " ".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct PAWS-QQP from its index and QQP.")
    parser.add_argument("--original-qqp", type=Path, required=True)
    parser.add_argument("--paws-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = read_original_qqp(args.original_qqp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "qid1", "sentence1", "qid2", "sentence2", "label"]
    with args.paws_index.open(encoding="utf-8", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source, fieldnames=fields, delimiter="\t")
        next(reader)
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "sentence1", "sentence2", "label"])
        for row in reader:
            writer.writerow(
                [
                    row["id"],
                    build_sentence(row["sentence1"], int(row["qid1"]), questions),
                    build_sentence(row["sentence2"], int(row["qid2"]), questions),
                    row["label"],
                ]
            )


if __name__ == "__main__":
    main()

