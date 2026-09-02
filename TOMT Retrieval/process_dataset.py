import os
import argparse
import time
import logging
from config import configure_logging
from tomt.data import utils, reddit
from tqdm import tqdm

log = logging.getLogger(__name__)

def save_submission(submission, path):
    j = {
        "title": submission.title,
        "selftext": submission.selftext,
        "url": submission.url
    }
    utils.write_json(j, path)
    return j

def get_data(submission_id, cache_dir, reddit_config, force_refresh=False):
    p = os.path.join(cache_dir, submission_id + ".json")
    # load from cache
    if os.path.exists(p) and not force_refresh:
        return utils.read_json(p)
    else:
        submission = reddit.get_submission(submission_id, reddit_config, get_comments=False)
        if submission is None:
            raise ValueError("not found!")
        return save_submission(submission, p)

def load_fields(data, cache_dir, reddit_config_path, force_refresh):
    reddit_config = utils.read_json(reddit_config_path)
    not_found = 0
    for d in tqdm(data, unit="doc"):
        try:
            submission = get_data(d["id"], cache_dir,
                                  reddit_config=reddit_config,
                                  force_refresh=force_refresh)
        except KeyboardInterrupt as e:
            raise ValueError(e)
        except:
            not_found += 1
            continue
    
        # todo: clean?
        d["title"] = submission["title"]
        d["description"] = submission["selftext"]
        d["url"] = submission["url"]

    log.info(f"not found: {not_found}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("process_dataset")
    parser.add_argument("--input", help="location to input .json file", required=True)
    parser.add_argument("--out", help="location to output .json file", required=True)
    parser.add_argument("--cache", help="directory to cache data", default="./.submissions_cache")
    parser.add_argument("--force_refresh", action="store_true", default=False)
    parser.add_argument(
        "--reddit_config_path", help="location of reddit config", default="./reddit_config.json")
    configure_logging("CreateFiles", False)

    args = parser.parse_args()
    log.info(f"args: {args}")
    assert args.input != args.out, "input == out!"
    os.makedirs(args.cache, exist_ok=True)
    start_time = time.time()
    if not os.path.exists(args.out):
        data = utils.read_jsonl(args.input)
        load_fields(data, args.cache, args.reddit_config_path, args.force_refresh)
        utils.write_jsonl(data, args.out)
        log.info(f"took {time.time() - start_time} seconds!")
    else:
        log.info(f"{args.out} exists. skipping!")
