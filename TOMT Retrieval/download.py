import os
import logging
import argparse

import config
from tomt.data import reddit

log = logging.getLogger("reddit.download")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download",
                                     description="Downloads submissions from PushShift and then downloads submission by submission using PRAW")
    parser.add_argument("--start-time", dest="start_time", help="start time in DD-MM-YYYY format", required=True)
    parser.add_argument("--end-time", dest="end_time", help="end time in DD-MM-YYYY format", required=True)
    parser.add_argument(
        "--conf", help="location of the JSON config", required=True)
    parser.add_argument("--submissions", help="output location for the submissions dump",
                        default="dataset/submissions.json")
    parser.add_argument("--output", help="output location for the threads", default="dataset/tomt/")
    parser.add_argument("--sleep_time", help="sleep time in seconds between each request for Pushshift", type=float,
                        default=1.)

    config.add_common_args(parser)

    args = parser.parse_args()

    config.configure_logging("download", args.verbose)
    config.supress_log("urllib3")
    config.supress_log("prawcore")

    if not os.path.exists(args.submissions):
        # first, download the submissions file
        reddit.get_all_submissions(args.start_time, args.end_time, args.submissions, sleep_time=args.sleep_time)

    # use the submission ids to get the threads from reddit
    reddit.download_submissions(args.conf, args.submissions, args.output)
