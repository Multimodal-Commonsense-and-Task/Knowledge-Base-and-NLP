#!/usr/bin/env python3
"""
Split scenarios by `merge_source`.

Input format:
- A JSON dict with a `scenarios` list (plus optional `metadata`)
  OR a bare JSON list of scenarios.

Output:
- One JSON file per merge_source value, written under `--output-dir`.
  Each output file is wrapped as {"metadata": ..., "scenarios": ...}.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def _load_input(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        return data, data["scenarios"]

    if isinstance(data, list):
        return {"metadata": {}}, data

    raise ValueError("Expected dict with 'scenarios' list or a list of scenarios")


def _write_output(path: str, output_data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split merged scenarios by merge_source")
    parser.add_argument("--input", required=True, help="Input merged JSON (e.g., graphs_final.json)")
    parser.add_argument("--output-dir", required=True, help="Directory to write per-source JSON files")
    parser.add_argument(
        "--output-template",
        default="graphs_{merge_source}.json",
        help="Filename template; supports {merge_source} (default: graphs_{merge_source}.json)",
    )
    parser.add_argument(
        "--unknown-label",
        default="unknown",
        help="Label to use when merge_source is missing/empty (default: unknown)",
    )

    args = parser.parse_args()

    input_wrapper, scenarios = _load_input(args.input)
    input_metadata = input_wrapper.get("metadata", {}) if isinstance(input_wrapper, dict) else {}

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        merge_source = scenario.get("merge_source") if isinstance(scenario, dict) else None
        merge_source = merge_source.strip() if isinstance(merge_source, str) else ""
        groups[merge_source or args.unknown_label].append(scenario)

    counts = Counter({k: len(v) for k, v in groups.items()})
    print(f"Loaded {len(scenarios)} scenarios from {args.input}")
    print("Split counts:")
    for label, count in counts.most_common():
        print(f"  - {label}: {count}")

    for merge_source, items in sorted(groups.items(), key=lambda kv: kv[0]):
        out_name = args.output_template.format(merge_source=merge_source)
        out_path = os.path.join(args.output_dir, out_name)
        output_data = {
            "metadata": {
                "total_scenarios": len(items),
                "split_key": "merge_source",
                "merge_source": merge_source,
                "input_file": os.path.basename(args.input),
                "input_metadata": input_metadata,
            },
            "scenarios": items,
        }
        _write_output(out_path, output_data)

    print(f"Output written to: {args.output_dir}")


if __name__ == "__main__":
    main()

