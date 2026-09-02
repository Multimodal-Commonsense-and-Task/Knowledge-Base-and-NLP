"""Helpers for reading the sharded LLM outputs written by ``2_run_llm.py``."""

import glob
import json
import os


def load_exactmatch(result_dir):
    """Concatenate ``result_*.json`` shards back into one input-ordered EM list."""
    paths = glob.glob(os.path.join(result_dir, 'result_*.json'))
    if not paths:
        raise FileNotFoundError(f'no result_*.json under {result_dir}')

    shards = []
    for path in paths:
        with open(path) as f:
            shards.append(json.load(f))
    shards.sort(key=lambda s: s.get('start', s.get('shard_id', 0)))

    exactmatch = []
    for shard in shards:
        if shard.get('start') is not None and shard['start'] != len(exactmatch):
            raise ValueError(
                f"shard starting at {shard['start']} does not continue from "
                f'{len(exactmatch)}; a shard is missing or incomplete')
        exactmatch.extend(shard['exactmatch'])
    return exactmatch


def dedup(ctxs):
    """Deduplicate context dicts while preserving order."""
    seen, out = set(), []
    for ctx in ctxs:
        key = frozenset(ctx.items())
        if key not in seen:
            seen.add(key)
            out.append(ctx)
    return out
