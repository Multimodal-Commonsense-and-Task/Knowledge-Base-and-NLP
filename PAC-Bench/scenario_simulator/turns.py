from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class AggregateStats:
	"""Accumulates max-turn counts for any grouping."""

	total_turns: int = 0
	samples: int = 0

	def add(self, turn_count: int) -> None:
		self.total_turns += turn_count
		self.samples += 1

	@property
	def average(self) -> float:
		return self.total_turns / self.samples if self.samples else 0.0


def find_result_files(root: str) -> Iterable[str]:
	for dirpath, _, filenames in os.walk(root):
		if "result.json" in filenames:
			yield os.path.join(dirpath, "result.json")


def extract_model_pair(result_path: str) -> Optional[str]:
	scenario_dir = os.path.dirname(result_path)
	model_pair_dir = os.path.dirname(scenario_dir)
	if not model_pair_dir or model_pair_dir == scenario_dir:
		return None
	return os.path.basename(model_pair_dir)


def split_model_pair(model_pair: str) -> Tuple[str, ...]:
	"""Return component models for names formatted as `modelA_modelB`."""

	if model_pair.count("_") != 1:
		return (model_pair,)

	left, right = model_pair.split("_", 1)
	if not left or not right:
		return (model_pair,)
	return (left, right)


def max_turn_from_file(path: str) -> Optional[int]:
	try:
		with open(path, "r", encoding="utf-8") as stream:
			payload = json.load(stream)
	except (OSError, json.JSONDecodeError):
		return None

	history = payload.get("response_history")
	if not isinstance(history, list):
		return None

	max_turn: Optional[int] = None
	for item in history:
		if not isinstance(item, dict):
			continue
		turn = item.get("turn")
		if isinstance(turn, int):
			max_turn = turn if max_turn is None else max(max_turn, turn)

	return max_turn


def summarize(
	results_roots: Sequence[str],
) -> Tuple[
	Dict[str, AggregateStats],
	Dict[str, AggregateStats],
	Dict[str, AggregateStats],
	Dict[str, AggregateStats],
	int,
	int,
]:
	pair_stats: DefaultDict[str, AggregateStats] = defaultdict(AggregateStats)
	model_stats: DefaultDict[str, AggregateStats] = defaultdict(AggregateStats)
	agent_a_stats: DefaultDict[str, AggregateStats] = defaultdict(AggregateStats)
	agent_b_stats: DefaultDict[str, AggregateStats] = defaultdict(AggregateStats)
	processed = 0
	skipped = 0

	for root in results_roots:
		if not os.path.isdir(root):
			print(f"Warning: '{root}' is not a directory, skipping.", file=sys.stderr)
			continue

		for file_path in find_result_files(root):
			model_pair = extract_model_pair(file_path)
			max_turn = max_turn_from_file(file_path)

			if not model_pair or max_turn is None:
				skipped += 1
				continue

			pair_stats[model_pair].add(max_turn)
			
			parts = split_model_pair(model_pair)
			if len(parts) == 2:
				a, b = parts
				agent_a_stats[a].add(max_turn)
				agent_b_stats[b].add(max_turn)
				for model in set(parts):
					model_stats[model].add(max_turn)
			else:
				model_stats[parts[0]].add(max_turn)
				
			processed += 1

	return pair_stats, model_stats, agent_a_stats, agent_b_stats, processed, skipped


def main(argv: Optional[Iterable[str]] = None) -> int:
	parser = argparse.ArgumentParser(
		description="Aggregate average max turn count per model pair and per model."
	)
	parser.add_argument(
		"results_root",
		nargs="?",
		default=os.path.join(os.path.dirname(__file__), "results"),
		help="Fallback path to the root results directory when --input is not provided (default: ./results)",
	)
	parser.add_argument(
		"-i",
		"--input",
		dest="input_dirs",
		action="append",
		type=str,
		help="Specific directory to scan for result files. Can be repeated to aggregate multiple folders.",
	)
	args = parser.parse_args(list(argv) if argv is not None else None)

	results_roots: List[str]
	if args.input_dirs:
		results_roots = args.input_dirs
	else:
		results_roots = [args.results_root]

	pair_stats, model_stats, agent_a_stats, agent_b_stats, processed, skipped = summarize(results_roots)
	if not pair_stats:
		joined_roots = ", ".join(results_roots)
		print(
			f"No valid result.json files found under: {joined_roots}.",
			file=sys.stderr,
		)
		return 1

	print("Per-model-pair averages")
	header = f"{'Model Pair':<40} {'Samples':>8} {'Avg Turns':>12}"
	print(header)
	print("-" * len(header))

	for model_pair in sorted(pair_stats):
		stats_row = pair_stats[model_pair]
		print(f"{model_pair:<40} {stats_row.samples:>8} {stats_row.average:>12.2f}")

	if model_stats:
		print()
		print("Per-model averages (Comparison: Agent A vs Agent B)")
		header_model = f"{'Model':<30} {'Total Avg':>10} {'As Agent A':>12} {'As Agent B':>12} {'Samples':>8}"
		print(header_model)
		print("-" * len(header_model))

		for model in sorted(model_stats):
			total = model_stats[model]
			a_stat = agent_a_stats.get(model, AggregateStats())
			b_stat = agent_b_stats.get(model, AggregateStats())
			
			a_avg_str = f"{a_stat.average:12.2f}" if a_stat.samples > 0 else f"{'-':>12}"
			b_avg_str = f"{b_stat.average:12.2f}" if b_stat.samples > 0 else f"{'-':>12}"
			
			print(f"{model:<30} {total.average:10.2f} {a_avg_str} {b_avg_str} {total.samples:>8}")
	else:
		print()
		print("No per-model statistics could be derived (unrecognized naming convention).")

	print()
	print(f"Processed files: {processed}")
	print(f"Skipped files:   {skipped}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
