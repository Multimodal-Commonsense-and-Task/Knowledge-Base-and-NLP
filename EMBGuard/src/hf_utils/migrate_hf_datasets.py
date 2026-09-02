#!/usr/bin/env python3
"""
Migrate EMBGuard Hugging Face datasets to the public column schema.

This script reads existing Hub datasets and pushes transformed DatasetDicts.
It does not require local CSV/image files; the Hub dataset is the source.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub import login
from requests.exceptions import RequestException


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.test_types import (  # noqa: E402
    TEST_TYPE_CODES,
    hf_split_candidates,
    is_benign_test_type,
    is_risky_test_type,
    normalize_test_type_code,
    test_type_label,
    test_type_slug,
)


DATASET_ALIASES = {
    "embguardtest": "EMBGuardTest",
    "embguardtest_v2": "EMBGuardTest_v2",
    "embguardtest-v2": "EMBGuardTest_v2",
    "test": "EMBGuardTest_v2",
    "test_v2": "EMBGuardTest_v2",
    "test-v2": "EMBGuardTest_v2",
    "heldout": "heldout_set",
    "heldout_set": "heldout_set",
    "embhazard": "EMBHazard",
    "hazard": "EMBHazard",
}

PUBLIC_SCHEMA_COLUMNS = [
    "id",
    "image",
    "image_path",
    "risk",
    "type",
    "scenario_type",
    "risk_type",
    "risk_subtype",
    "situation",
    "hazard",
    "action",
    "room",
    "multi_scenario",
    "pair_item_id",
]


def canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name.strip().lower(), name.strip())


def _load_dataset_dict(repo_id: str, token: Optional[str], revision: Optional[str]) -> DatasetDict:
    print(f"Loading source dataset: {repo_id}")
    dataset = load_dataset(repo_id, token=token, revision=revision)
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def _label_for_value(value: object) -> str:
    code = normalize_test_type_code(value)
    return test_type_label(code) if code in TEST_TYPE_CODES else ""


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _column_values(dataset: Dataset, *names: str) -> list[str]:
    for name in names:
        if name in dataset.column_names:
            return [_as_text(value) for value in dataset[name]]
    return [""] * len(dataset)


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = _as_text(value).strip()
        if text:
            return text
    return ""


def _normalize_risk_value(value: object, fallback_code: object = None, fallback_type: object = None) -> str:
    text = _as_text(value).strip().upper()
    if text in {"O", "X"}:
        return text

    type_text = _as_text(fallback_type).strip().lower()
    if type_text == "unsafe":
        return "O"
    if type_text == "safe":
        return "X"

    code = normalize_test_type_code(fallback_code)
    if is_risky_test_type(code):
        return "O"
    if is_benign_test_type(code):
        return "X"
    return ""


def _risk_label_from_values(risk: object, legacy_type: object = None, scenario_code: object = None) -> str:
    type_text = _as_text(legacy_type).strip().lower()
    if type_text in {"safe", "unsafe"}:
        return type_text

    risk_text = _as_text(risk).strip().upper()
    if risk_text == "O":
        return "unsafe"
    if risk_text == "X":
        return "safe"

    code = normalize_test_type_code(scenario_code)
    if is_risky_test_type(code):
        return "unsafe"
    if is_benign_test_type(code):
        return "safe"
    return ""


def _is_yes(value: object) -> bool:
    return _as_text(value).strip().lower() in {"yes", "y", "true", "1"}


def _scenario_type_from_values(*values: object, multi_scenario: object = None) -> str:
    if _is_yes(multi_scenario):
        return test_type_slug("MHR")

    for value in values:
        code = normalize_test_type_code(value)
        if code in TEST_TYPE_CODES:
            return test_type_slug(code)

    for value in values:
        text = _as_text(value).strip().lower().replace("-", "_").replace(" ", "_")
        if text in {test_type_slug(code) for code in TEST_TYPE_CODES}:
            return text

    return ""


def _normalize_public_schema(
    dataset: Dataset,
    *,
    split_name: str,
    force_scenario_code: Optional[str] = None,
) -> Dataset:
    """Return a compact public dataset schema without legacy display columns."""
    ids = _column_values(dataset, "id", "ID")
    image_paths = _column_values(dataset, "image_path", "image_url", "source_path", "source_url", "URL", "url")
    risks = _column_values(dataset, "risk", "Risk")
    legacy_types = _column_values(dataset, "type", "Type")
    legacy_subtypes = _column_values(dataset, "scenario_type", "Subtype")
    risk_types = _column_values(dataset, "risk_type", "Risk Type", "Category")
    risk_subtypes = _column_values(dataset, "risk_subtype", "Subcategory")
    situations = _column_values(dataset, "situation", "Situation")
    hazards = _column_values(dataset, "hazard", "Related Hazard", "Hazard")
    actions = _column_values(dataset, "action", "Action")
    rooms = _column_values(dataset, "room", "Room")
    multi_scenarios = _column_values(dataset, "multi_scenario", "MultiScenario")
    pair_item_ids = _column_values(dataset, "pair_item_id", "Pair Item ID")

    scenario_values = []
    normalized_risks = []
    risk_labels = []
    for risk, legacy_type, legacy_subtype, multi_scenario in zip(risks, legacy_types, legacy_subtypes, multi_scenarios):
        if force_scenario_code:
            scenario = _scenario_type_from_values(force_scenario_code)
        else:
            scenario = _scenario_type_from_values(
                legacy_subtype,
                legacy_type,
                split_name,
                multi_scenario=multi_scenario,
            )
        scenario_values.append(scenario)
        normalized_risk = _normalize_risk_value(risk, fallback_code=scenario, fallback_type=legacy_type)
        normalized_risks.append(normalized_risk)
        risk_labels.append(_risk_label_from_values(normalized_risk, legacy_type=legacy_type, scenario_code=scenario))

    public_columns = {
        "id": ids,
        "image_path": image_paths,
        "risk": normalized_risks,
        "type": risk_labels,
        "scenario_type": scenario_values,
        "risk_type": risk_types,
        "risk_subtype": risk_subtypes,
        "situation": situations,
        "hazard": [
            _first_nonempty(hazard, situation)
            for hazard, situation in zip(hazards, situations)
        ],
        "action": actions,
        "room": rooms,
        "multi_scenario": multi_scenarios,
        "pair_item_id": pair_item_ids,
    }

    temp_columns = {}
    for column in [col for col in PUBLIC_SCHEMA_COLUMNS if col != "image"]:
        temp_column = f"__embguard_v3_{column}"
        temp_columns[temp_column] = column
        if temp_column in dataset.column_names:
            dataset = dataset.remove_columns([temp_column])
        dataset = dataset.add_column(temp_column, public_columns[column])

    ordered_temp_columns = []
    for column in PUBLIC_SCHEMA_COLUMNS:
        if column == "image":
            if "image" in dataset.column_names:
                ordered_temp_columns.append("image")
        else:
            ordered_temp_columns.append(f"__embguard_v3_{column}")

    dataset = dataset.select_columns(ordered_temp_columns)
    for temp_column, public_column in temp_columns.items():
        dataset = dataset.rename_column(temp_column, public_column)

    ordered_columns = [column for column in PUBLIC_SCHEMA_COLUMNS if column in dataset.column_names]
    return dataset.select_columns(ordered_columns)


def _add_type_label_columns(dataset: Dataset, force_type_code: Optional[str] = None) -> Dataset:
    has_type = "Type" in dataset.column_names
    has_subtype = "Subtype" in dataset.column_names

    # Avoid dataset.map() here: rewriting large image columns can overflow Arrow
    # offsets for image-heavy datasets. Adding/replacing scalar columns is safer.
    if "Type Label" in dataset.column_names:
        dataset = dataset.remove_columns(["Type Label"])
    if "Subtype Label" in dataset.column_names:
        dataset = dataset.remove_columns(["Subtype Label"])

    if force_type_code:
        if "Type" in dataset.column_names:
            dataset = dataset.remove_columns(["Type"])
        dataset = dataset.add_column("Type", [force_type_code] * len(dataset))
        dataset = dataset.add_column("Type Label", [test_type_label(force_type_code)] * len(dataset))
    elif has_type:
        dataset = dataset.add_column("Type Label", [_label_for_value(value) for value in dataset["Type"]])

    if has_subtype:
        dataset = dataset.add_column("Subtype Label", [_label_for_value(value) for value in dataset["Subtype"]])

    return dataset


def migrate_embguardtest(source: DatasetDict) -> DatasetDict:
    migrated = DatasetDict()
    for code in TEST_TYPE_CODES:
        source_split = None
        for candidate in hf_split_candidates(code):
            if candidate in source:
                source_split = candidate
                break

        if not source_split:
            print(f"Warning: no source split found for {code}; tried {hf_split_candidates(code)}")
            continue

        target_split = test_type_slug(code)
        print(f"  {source_split} -> {target_split} ({test_type_label(code)})")
        migrated[target_split] = _add_type_label_columns(source[source_split], force_type_code=code)

    if not migrated:
        raise ValueError("No EMBGuardTest splits were found to migrate.")

    return migrated


def migrate_embguardtest_public(source: DatasetDict) -> DatasetDict:
    migrated = DatasetDict()
    for code in TEST_TYPE_CODES:
        source_split = None
        for candidate in hf_split_candidates(code):
            if candidate in source:
                source_split = candidate
                break

        if not source_split:
            print(f"Warning: no source split found for {code}; tried {hf_split_candidates(code)}")
            continue

        target_split = test_type_slug(code)
        print(f"  {source_split} -> {target_split}")
        migrated[target_split] = _normalize_public_schema(
            source[source_split],
            split_name=target_split,
            force_scenario_code=code,
        )

    if not migrated:
        raise ValueError("No EMBGuardTest splits were found to migrate.")

    return migrated


def migrate_label_columns_only(source: DatasetDict) -> DatasetDict:
    migrated = DatasetDict()
    for split_name, dataset in source.items():
        print(f"  {split_name} -> {split_name}")
        migrated[split_name] = _add_type_label_columns(dataset)
    return migrated


def _select_rows_by_type(dataset: Dataset, code: str) -> Dataset:
    indices = [
        index
        for index, value in enumerate(dataset["Type"])
        if normalize_test_type_code(value) == code
    ]
    return dataset.select(indices)


def _select_rows_by_scenario(dataset: Dataset, code: str) -> Dataset:
    source_values = []
    if "scenario_type" in dataset.column_names:
        source_values.append(dataset["scenario_type"])
    if "Type" in dataset.column_names:
        source_values.append(dataset["Type"])
    if "Subtype" in dataset.column_names:
        source_values.append(dataset["Subtype"])
    if "type" in dataset.column_names:
        source_values.append(dataset["type"])

    if not source_values:
        return dataset.select([])

    indices = []
    for index, values in enumerate(zip(*source_values)):
        if any(normalize_test_type_code(value) == code for value in values):
            indices.append(index)
    return dataset.select(indices)


def migrate_heldout_set(source: DatasetDict, split_naming: str) -> DatasetDict:
    migrated = DatasetDict()

    if split_naming in {"safety", "both"}:
        for split_name in ("safe", "unsafe"):
            if split_name in source:
                print(f"  {split_name} -> {split_name}")
                migrated[split_name] = _add_type_label_columns(source[split_name])

    if split_naming in {"type", "both"}:
        source_splits = [source[split_name] for split_name in ("safe", "unsafe") if split_name in source]
        if not source_splits:
            source_splits = [dataset for split_name, dataset in source.items() if split_name in {test_type_slug(code) for code in TEST_TYPE_CODES}]
        if not source_splits:
            raise ValueError("heldout_set source must contain safe/unsafe or type splits.")

        combined = concatenate_datasets(source_splits)
        combined = _add_type_label_columns(combined)
        for code in TEST_TYPE_CODES:
            target_split = test_type_slug(code)
            selected = _select_rows_by_type(combined, code)
            print(f"  Type={code} -> {target_split} ({test_type_label(code)}): {len(selected)} rows")
            migrated[target_split] = selected

    return migrated


def migrate_heldout_set_public(source: DatasetDict, split_naming: str) -> DatasetDict:
    migrated = DatasetDict()

    if split_naming in {"safety", "both"}:
        for split_name in ("safe", "unsafe"):
            if split_name in source:
                print(f"  {split_name} -> {split_name}")
                migrated[split_name] = _normalize_public_schema(source[split_name], split_name=split_name)

    if split_naming in {"type", "both"}:
        source_splits = [source[split_name] for split_name in ("safe", "unsafe") if split_name in source]
        if not source_splits:
            source_splits = [
                dataset
                for split_name, dataset in source.items()
                if split_name in {test_type_slug(code) for code in TEST_TYPE_CODES}
            ]
        if not source_splits:
            raise ValueError("heldout_set source must contain safe/unsafe or type splits.")

        combined = concatenate_datasets(source_splits)
        for code in TEST_TYPE_CODES:
            target_split = test_type_slug(code)
            selected = _select_rows_by_scenario(combined, code)
            print(f"  scenario_type={target_split}: {len(selected)} rows")
            migrated[target_split] = _normalize_public_schema(
                selected,
                split_name=target_split,
                force_scenario_code=code,
            )

    return migrated


def migrate_dataset(dataset_name: str, source: DatasetDict, heldout_split_naming: str, schema: str) -> DatasetDict:
    if schema == "v3":
        if dataset_name.startswith("EMBGuardTest"):
            return migrate_embguardtest_public(source)
        if dataset_name == "heldout_set":
            return migrate_heldout_set_public(source, heldout_split_naming)
        if dataset_name == "EMBHazard":
            migrated = DatasetDict()
            for split_name, dataset in source.items():
                print(f"  {split_name} -> {split_name}")
                migrated[split_name] = _normalize_public_schema(dataset, split_name=split_name)
            return migrated
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name.startswith("EMBGuardTest"):
        return migrate_embguardtest(source)
    if dataset_name == "heldout_set":
        return migrate_heldout_set(source, heldout_split_naming)
    if dataset_name == "EMBHazard":
        return migrate_label_columns_only(source)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def print_dataset_summary(dataset_name: str, dataset_dict: DatasetDict) -> None:
    print(f"\n{dataset_name} migrated summary:")
    for split_name, dataset in dataset_dict.items():
        print(f"  - {split_name}: {len(dataset)} rows")
        print(f"    columns: {dataset.column_names}")
        if len(dataset) > 0:
            sample = dataset[0]
            sample_bits = []
            for key in ["risk", "type", "scenario_type", "risk_type", "risk_subtype", "image_path", "Type", "Type Label", "Subtype", "Subtype Label"]:
                if key in sample:
                    sample_bits.append(f"{key}={sample[key]!r}")
            if sample_bits:
                print(f"    sample: {', '.join(sample_bits)}")


def parse_datasets(values: Iterable[str]) -> list[str]:
    return [canonical_dataset_name(value) for value in values]


def target_dataset_name(dataset_name: str, embguardtest_target_name: Optional[str]) -> str:
    if dataset_name.startswith("EMBGuardTest") and embguardtest_target_name:
        return embguardtest_target_name
    return dataset_name


def push_to_hub_with_retries(
    dataset: DatasetDict,
    repo_id: str,
    *,
    private: bool,
    token: Optional[str],
    commit_message: str,
    max_shard_size: Optional[str],
    retries: int,
) -> None:
    transient_statuses = {500, 502, 503, 504}
    for attempt in range(1, retries + 1):
        try:
            dataset.push_to_hub(
                repo_id,
                private=private,
                token=token,
                commit_message=commit_message,
                max_shard_size=max_shard_size,
            )
            return
        except HfHubHTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code not in transient_statuses or attempt == retries:
                raise
            delay = min(60, attempt * 15)
            print(
                f"push_to_hub failed with HTTP {status_code}; "
                f"retrying in {delay}s ({attempt}/{retries})"
            )
            time.sleep(delay)
        except RequestException as exc:
            if attempt == retries:
                raise
            delay = min(60, attempt * 15)
            print(
                f"push_to_hub failed with network error {exc.__class__.__name__}; "
                f"retrying in {delay}s ({attempt}/{retries})"
            )
            time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate EMBGuard HF datasets from Hub to Hub.")
    parser.add_argument("--source-org", "--org", dest="source_org", default="EMBGuard")
    parser.add_argument("--target-org", default=None, help="Target org. Defaults to --source-org.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["EMBGuardTest_v2", "heldout_set", "EMBHazard"],
        help="Datasets to migrate: EMBGuardTest, EMBGuardTest_v2, heldout_set, EMBHazard",
    )
    parser.add_argument("--source-revision", default=None, help="Optional source dataset revision.")
    parser.add_argument(
        "--embguardtest-target-name",
        default=None,
        help="Target repo name for EMBGuardTest only, e.g. EMBGuardTest_v2. Other datasets keep their names.",
    )
    parser.add_argument("--private", action="store_true", help="Make target datasets private.")
    parser.add_argument("--dry-run", action="store_true", help="Build transformed datasets without pushing.")
    parser.add_argument("--token", default=None, help="HF token. Defaults to HF_TOKEN.")
    parser.add_argument(
        "--heldout-split-naming",
        choices=["type", "safety", "both"],
        default="both",
        help="heldout_set split schema: type adds causal_risky/etc.; safety keeps safe/unsafe; both keeps both.",
    )
    parser.add_argument(
        "--schema",
        choices=["v3", "legacy-labels"],
        default="v3",
        help=(
            "Output schema. v3 normalizes columns to id/image/image_path/risk/type/"
            "scenario_type/risk_type/risk_subtype/etc.; legacy-labels keeps legacy "
            "Type columns and adds Type Label/Subtype Label."
        ),
    )
    parser.add_argument(
        "--commit-message",
        default="Normalize EMBGuard dataset public schema",
        help="Commit message for push_to_hub.",
    )
    parser.add_argument(
        "--max-shard-size",
        default=None,
        help="Optional maximum shard size passed to DatasetDict.push_to_hub, e.g. 1GB.",
    )
    parser.add_argument(
        "--push-retries",
        type=int,
        default=1,
        help="Number of push_to_hub attempts for transient Hugging Face 5xx errors.",
    )
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    target_org = args.target_org or args.source_org
    dataset_names = parse_datasets(args.datasets)

    if token and not args.dry_run:
        login(token=token)
    elif args.dry_run:
        print("Dry run enabled: push_to_hub will be skipped.")

    for dataset_name in dataset_names:
        source_repo = f"{args.source_org}/{dataset_name}"
        target_name = target_dataset_name(dataset_name, args.embguardtest_target_name)
        target_repo = f"{target_org}/{target_name}"
        source = _load_dataset_dict(source_repo, token=token, revision=args.source_revision)
        migrated = migrate_dataset(dataset_name, source, args.heldout_split_naming, args.schema)
        print_dataset_summary(target_name, migrated)

        if args.dry_run:
            print(f"Dry run: not pushing {target_repo}")
            continue

        print(f"\nPushing {dataset_name} to {target_repo}")
        push_to_hub_with_retries(
            migrated,
            target_repo,
            private=args.private,
            token=token,
            commit_message=args.commit_message,
            max_shard_size=args.max_shard_size,
            retries=args.push_retries,
        )
        print(f"Uploaded: https://huggingface.co/datasets/{target_repo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
