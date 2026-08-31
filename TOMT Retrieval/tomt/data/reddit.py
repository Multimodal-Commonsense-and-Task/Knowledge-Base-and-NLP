import os
import json
import time
import logging
import argparse
import pickle as pkl
from datetime import datetime, timezone

import pytz
import praw
import requests
from tqdm.autonotebook import tqdm
from prawcore import exceptions

log = logging.getLogger(__name__)


def utc_timestamp(dt):
    """
        Converts a datetime object into a timestamp
    """
    timezone = pytz.timezone("utc")
    dt = timezone.localize(dt)
    return int(dt.timestamp())


def datetime_from_utc(utc):
    """
        Converts a UTC timestamp into a datetime object in the UTC timezone
    """
    return datetime.fromtimestamp(utc, tz=pytz.utc)


def pushshift_api(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?&size=1000&after=' + \
          str(after) + '&before=' + str(before) + '&subreddit=' + str(sub)
    log.info(f"URL :{url}")
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def pushshift_submission(subreddit, **kwargs):
    top_url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}"

    # filter out None terms first
    kwargs_cleaned = {}
    for k, v in kwargs.items():
        if v is not None:
            kwargs_cleaned[k] = v
    verbose = True
    if kwargs_cleaned.get("before") is not None and verbose:
        log.info(
            f"Getting threads before time {datetime_from_utc(kwargs_cleaned['before'])}")

    if kwargs_cleaned.get("after") is not None and verbose:
        log.info(
            f"Getting threads after time {datetime_from_utc(kwargs_cleaned['after'])}")

    payload_str = "&".join("%s=%s" % (k, v) for k, v in kwargs_cleaned.items())

    try:
        response = requests.get(top_url, params=payload_str, timeout=5)
    except requests.exceptions.Timeout:
        log.error("Timeout!")
        return None

    if response.status_code == 200:
        if verbose:
            log.info(response.url)

        return response.json()
    else:
        log.warning(f"{top_url} with payload {payload_str} failed with response: {response.status_code}!")
        return None


def get_all_submissions(start_time, end_time, output, sleep_time=0.5, retry_count=10):
    true_start_time = datetime.strptime(start_time, "%d-%m-%Y")
    true_end_time = datetime.strptime(end_time, "%d-%m-%Y")

    log.info(f"Start Time: {true_start_time}; End Time: {true_end_time}")

    assert true_end_time > true_start_time

    # we go backwards, so please forgive the name mixup
    start_time = utc_timestamp(true_end_time)
    last_time = utc_timestamp(true_start_time)

    # download 1000 submissions at a time
    limit = 1000

    # initialize before and after
    before = start_time  # start from this time and go backward until last_time is achieved
    after = None  # this is ignored in the first call
    assert last_time < start_time, "last_time must occur before start_time"
    all_submissions = []

    while True:
        retries = retry_count
        # get all submissions, with the created time
        submissions = pushshift_submission("tipofmytongue",
                                           before=before,
                                           limit=limit,
                                           fields=",".join(("created_utc", "id")))

        if submissions is None:
            retries = 0
            while retries < retry_count:
                log.info(f"Request failed: Retry {retries + 1}/{retry_count}")
                retries += 1
                submissions = pushshift_submission("tipofmytongue",
                                                   before=before,
                                                   limit=limit,
                                                   fields=",".join(("created_utc", "id")))

                if submissions is not None:
                    break

                time.sleep(sleep_time)

        if submissions is None:
            raise ValueError("Failed to get valid response from Pushshift")

        data = submissions["data"]

        for d in data:
            d["created_time"] = datetime_from_utc(d["created_utc"])

        all_submissions.extend(data)
        first_post = max(data, key=lambda _: _["created_utc"])
        last_post = min(data, key=lambda _: _["created_utc"])

        before = last_post["created_utc"]

        if before < last_time:
            log.info("Done!")
            break
        else:
            log.info(f"{len(all_submissions)} downloaded")

        time.sleep(sleep_time)

    # filter out all posts that occur before last_time
    all_submissions = [
        s for s in all_submissions if s["created_utc"] > last_time]
    # filter out all posts that occur after start_time
    all_submissions = [
        s for s in all_submissions if s["created_utc"] < start_time]

    log.info(
        f"First Post has Time: {all_submissions[0]['created_time']}\nLast Post has Time: {all_submissions[-1]['created_time']}")

    log.info(f"A total of {len(all_submissions)} submissions downloaded")

    # remove datetime objects
    for s in all_submissions:
        del s["created_time"]

    with open(output, "w") as writer:
        json.dump(all_submissions, writer, indent=2)


def get_submission(submission_id: str, config: dict, get_comments: bool):
    # create reddit instance
    # we want only read only, so no need to provide username / password
    reddit = praw.Reddit(client_id=config["client_id"],
                         client_secret=config["client_secret"],
                         user_agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1")

    assert reddit.read_only
    try:
        submission = reddit.submission(submission_id)

        if get_comments:
            # extract all comments
            submission.comments.replace_more(limit=None)
        return submission
    except exceptions.NotFound:
        log.warning(f"not found: {submission_id}")


def download_submissions(config_file, input_submissions, output_folder):
    with open(config_file) as reader:
        config = json.load(reader)

    # create reddit instance
    # we want only read only, so no need to provide username / password
    reddit = praw.Reddit(client_id=config["client_id"],
                         client_secret=config["client_secret"],
                         user_agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1")

    assert reddit.read_only

    # create directory
    os.makedirs(output_folder, exist_ok=True)

    with open(input_submissions) as reader:
        submissions = json.load(reader)

    n_submissions = len(submissions)
    not_found = []
    for idx in tqdm(range(n_submissions)):
        submission_id = submissions[idx]["id"]

        pkl_path = os.path.join(output_folder, f"{submission_id}.pkl")

        # if path exists, skip
        if os.path.exists(pkl_path):
            continue

        try:
            submission = reddit.submission(submission_id)
            # extract all comments
            submission.comments.replace_more(limit=None)
        except exceptions.NotFound:
            not_found.append(submission_id)
            continue

        # dump the object
        with open(pkl_path, "wb") as writer:
            pkl.dump(submission, writer)

    log.info(f"{len(not_found)}  documents not found")


def pprint_tree(node, _prefix="", _last=True):
    # Source: https://vallentin.io/2016/11/29/pretty-print-tree
    print(_prefix, "`- " if _last else "|- ", repr(node.body), sep="")
    _prefix += "   " if _last else "|  "
    child_count = len(node.replies)
    for i, child in enumerate(node.replies):
        _last = i == (child_count - 1)
        pprint_tree(child, _prefix, _last)


def iterate_raw_submissions(folder_paths, include_ids=None, exclude_ids=None):
    assert isinstance(folder_paths, list)
    if include_ids:
        assert exclude_ids is None
    if exclude_ids:
        assert include_ids is None
    for folder in tqdm(folder_paths, desc="Folders", leave=False):
        for f in tqdm(os.listdir(folder), desc="Submissions", leave=False):
            submission_id = f.split(".")[0]
            if include_ids and submission_id not in include_ids:
                continue
            if exclude_ids and submission_id in exclude_ids:
                continue
            with open(os.path.join(folder, f), "rb") as reader:
                submission = pkl.load(reader)

                # skip nsfw posts
                if submission.over_18:
                    continue
                yield submission
