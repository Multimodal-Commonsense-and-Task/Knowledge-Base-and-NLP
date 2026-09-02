#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Tuple


def _get_project_root() -> str:
    """Get project root directory (EMBGuard)"""
    current_file = os.path.abspath(__file__)
    # json_to_dataset_csv.py is at: src/dataset_generation/utils/json_to_dataset_csv.py
    # Go up 4 levels to reach project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    return project_root


def _normalize_url_path(url: str, project_root: str) -> str:
    """Normalize URL path to be relative to project root"""
    if not url:
        return url
    
    # Convert to absolute path if relative
    if not os.path.isabs(url):
        return url
    
    # Make path relative to project root
    try:
        rel_path = os.path.relpath(url, project_root)
        # If path is outside project root, return as is
        if rel_path.startswith(".."):
            return url
        return rel_path
    except ValueError:
        # If paths are on different drives (Windows), return as is
        return url


HEADER = [
    "Category",
    "Subcategory", 
    "Type",
    "Subtype",
    "MultiScenario",
    "ID",
    "Situation",
    "Action",
    "Risk",
    "Risk Type",
    "Related Hazard",
    "Mitigate Action",
    "URL",
]


def _load_scenarios(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        return data["scenarios"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON format (expected list or {{scenarios: [...]}}): {path}")


def _split_plus(value: str, index: int) -> str:
    parts = [p.strip() for p in (value or "").split("+")]
    if len(parts) <= 1:
        return value or ""
    if index < len(parts):
        return parts[index]
    return parts[-1]


def _iter_dataset_rows(s: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    scenario_id = str(s.get("id") or "").strip()
    if not scenario_id:
        return

    is_dual = all(k in s for k in ["hazard1", "action1", "hazard2", "action2"])
    if is_dual:
        situation_1 = s.get("situation1") or s.get("situation") or ""
        situation_2 = s.get("situation2") or s.get("situation") or ""
        
        # Generate situations from hazards if not present
        if not situation_1:
            hazard1 = s.get("hazard1") or ""
            if hazard1:
                situation_1 = f"In a kitchen, {hazard1.lower()}"
        if not situation_2:
            hazard2 = s.get("hazard2") or ""
            if hazard2:
                situation_2 = f"In a kitchen, {hazard2.lower()}"
        
        base = {
            "type": s.get("type") or "unsafe",
            "url": s.get("url") or s.get("image_path") or "",
        }

        yield {
            **base,
            "id": f"{scenario_id}_v1",
            "risk_type": _split_plus(str(s.get("risk_type") or ""), 0),
            "mechanism": _split_plus(str(s.get("mechanism") or ""), 0),
            "hazard": s.get("hazard1") or "",
            "action": s.get("action1") or "",
            "situation": situation_1,
        }
        yield {
            **base,
            "id": f"{scenario_id}_v2",
            "risk_type": _split_plus(str(s.get("risk_type") or ""), 1),
            "mechanism": _split_plus(str(s.get("mechanism") or ""), 1),
            "hazard": s.get("hazard2") or "",
            "action": s.get("action2") or "",
            "situation": situation_2,
        }
        return

    # Generate situation from hazard if not present
    situation = s.get("situation") or ""
    if not situation:
        hazard = s.get("hazard") or ""
        if hazard:
            situation = f"In a kitchen, {hazard.lower()}"
    
    # Use image_path as url if url not present
    url = s.get("url") or s.get("image_path") or ""
    
    # Set type to "unsafe" if not present
    scenario_type = s.get("type") or "unsafe"
    
    yield {
        "id": scenario_id,
        "risk_type": s.get("risk_type") or "",
        "mechanism": s.get("mechanism") or "",
        "hazard": s.get("hazard") or "",
        "action": s.get("action") or "",
        "situation": situation,
        "type": scenario_type,
        "url": url,
    }


def _get_subtype_from_filename(filename: str) -> str:
    """Determine subtype based on filename"""
    filename = filename.lower()
    
    if "hazard_augmented" in filename:
        return "HR"  # Causal Risky
    elif "scene_augmented_safe" in filename:
        return "HNR"  # Decoupled Benign
    elif "scene_augmented" in filename:
        return "HR"  # Causal Risky
    elif "hazard_removed" in filename:
        return "NHR"  # Absent Benign
    else:
        return "HR"  # Default


def _is_safe_scenario(filename: str) -> bool:
    """Determine if scenario is safe based on filename"""
    filename = filename.lower()
    return "hazard_removed" in filename or "scene_augmented_safe" in filename


def _is_multi_scenario(filename: str) -> bool:
    """Determine if scenario is multi-scenario (hazard_augmented) based on filename"""
    filename = filename.lower()
    return "hazard_augmented" in filename


def _to_csv_row(d: Dict[str, Any], subtype: str = "HR", is_safe: bool = False, is_multi_scenario: bool = False) -> Dict[str, str]:
    category = str(d.get("risk_type") or "")
    risk_value = "X" if is_safe else "O"  # X for safe scenarios, O for risky scenarios
    type_value = "safe" if is_safe else str(d.get("type") or "unsafe")  # safe for safe scenarios, original type or unsafe for risky scenarios
    multi_scenario_value = "yes" if is_multi_scenario else "no"
    
    return {
        "Category": category,
        "Subcategory": str(d.get("mechanism") or ""),
        "Type": type_value,
        "Subtype": subtype,
        "MultiScenario": multi_scenario_value,
        "ID": str(d.get("id") or ""),
        "Situation": str(d.get("situation") or ""),
        "Action": str(d.get("action") or ""),
        "Risk": risk_value,
        "Risk Type": category,
        "Related Hazard": str(d.get("hazard") or ""),
        "Mitigate Action": "X",
        "URL": str(d.get("url") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert scenario JSON(s) to dataset CSV.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--require-url",
        action="store_true",
        help="Drop rows with empty URL.",
    )
    args = parser.parse_args()

    # Get project root for URL normalization
    project_root = _get_project_root()

    rows: List[Dict[str, str]] = []
    for path in args.inputs:
        scenarios = _load_scenarios(path)
        filename = os.path.basename(path)
        subtype = _get_subtype_from_filename(filename)
        is_safe = _is_safe_scenario(filename)
        is_multi = _is_multi_scenario(filename)
        
        for s in scenarios:
            for d in _iter_dataset_rows(s):
                # Normalize URL path to be relative to project root
                if "url" in d and d["url"]:
                    d["url"] = _normalize_url_path(d["url"], project_root)
                
                row = _to_csv_row(d, subtype, is_safe, is_multi)
                if args.require_url and not row["URL"]:
                    continue
                rows.append(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
