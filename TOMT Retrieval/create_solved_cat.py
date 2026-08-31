import argparse
import os
import logging
from datetime import datetime
import logging.handlers
import json
import re
from config import configure_logging
from tomt.data import reddit, utils
from tomt.data.submissions import *

log = logging.getLogger(__name__)


# See https://stackoverflow.com/a/22238613
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def traverse_and_add(reply):
    reply_tree = comment_to_json(reply)

    sub_replies = []
    for child in reply.replies:
        sub_replies.append(traverse_and_add(child))

    reply_tree["replies"] = None if len(sub_replies) == 0 else sub_replies

    return reply_tree


def comment_to_json(comment):
    tree = {
        "id": comment.id,
        "body": comment.body,
        "created_date": reddit.datetime_from_utc(comment.created_utc)
    }

    if comment.author:
        if isinstance(comment.author.name, tuple):
            author_name = comment.author.name
        else:
            author_name = comment.author.name
        tree["author_name"] = author_name

    return tree


def get_comment_forest(comments):
    forest = []

    for comment in comments:
        tree = comment_to_json(comment)

        replies = []
        for reply in comment.replies:
            replies.append(traverse_and_add(reply))

        tree["replies"] = None if len(replies) == 0 else replies
        forest.append(tree)
    return forest


def submission_to_json(submission, thread_status, raw_category, normalized_category):
    j = {
        "id": submission.id,
        "category": normalized_category,
        "author": submission.author.name if submission.author is not None else None,
        "raw_category": raw_category,
        "status": thread_status,
        "title": submission.title,
        "description": submission.selftext,
        "created_utc": reddit.datetime_from_utc(submission.created_utc),
        "replies": get_comment_forest(submission.comments)
    }

    return j


def get_raw_category(title):
    cat = re.search(r'\[TOMT\](\s*)\[([^]]*)\]',
                    title, re.IGNORECASE)
    if cat:
        cat = cat.group(2)
        return cat.lower()
    else:
        # ignore NSFW requests
        if re.search(r"^\s*\[TOMP].*", title, re.IGNORECASE) is not None:
            return "nsfw"
        else:
            # unresolved
            return None


def get_status(submission, status_dict):
    if submission.selftext == "[removed]":
        status = "removed"
    elif submission.link_flair_text is None:
        status = "unknown"
    else:
        status = submission.link_flair_text

    return status_dict[status]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CreateSolveCat",
                                     description="Creates JSON file of solved submissions of a particular category")
    parser.add_argument("--input_folder", help="location of the submission pickles", required=True)
    parser.add_argument("--status_cat", help="location of the JSON file containing standardized statuses",
                        required=True)
    parser.add_argument("--norm_cat", help="location of TSV file containing raw text category -> standardized category",
                        required=True)
    parser.add_argument("--cat",
                        help="category to keep (others are discarded)", required=True)
    parser.add_argument("--out",
                        help="path to output JSON file", required=True)
    configure_logging(__name__, False)
    args = parser.parse_args()

    if os.path.exists(args.out):
        raise ValueError(f"{args.out} already exists!")

    normalized_categories = {}

    with open(args.norm_cat) as reader:
        # skip first line
        reader.readline()
        for line in reader:
            raw_category, _, cat = line.strip().split("\t")
            normalized_categories[raw_category] = cat

    status_dict = utils.read_json(args.status_cat)

    data = {}
    for submission in reddit.iterate_raw_submissions([args.input_folder]):
        # skip posts which have no description
        if submission.selftext.strip() == "[deleted]":
            continue
        thread_status = get_status(submission, status_dict)
        if thread_status != "Solved":
            continue
        raw_category = get_raw_category(submission.title)
        normalized_category = normalized_categories.get(raw_category)

        if not normalized_category:
            continue
        if normalized_category != args.cat:
            continue

        submission_json = submission_to_json(submission, thread_status, raw_category, normalized_category)

        solved_nodes = find_solved_node(submission_json)

        if len(solved_nodes) == 1:
            match_id = list(solved_nodes)[0]
            gather_descendants(submission_json)

            path, c = find_path_to_node(submission_json, match_id)
            if path is None:
                continue

            data[submission_json["id"]] = {
                "submission": submission_json,
                "solved_path": c,
                "solved_path_ids": path
            }

    log.info(f"Writing {len(data)} entries")

    with open(args.out, "w") as writer:
        json.dump(data, writer, default=json_serial)
