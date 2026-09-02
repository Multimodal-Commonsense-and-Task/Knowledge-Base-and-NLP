#!/usr/bin/env python3
"""
Merge Multiple Graph Files

Merges multiple graph JSON files (normalized, scene_augmented, hazard_removed, hazard_augmented)
into a single file with combined scenarios and updated metadata.
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Any


def load_graph_file(path: str) -> Dict[str, Any]:
    """Load a graph JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_multiple_scenarios(
    file_paths: List[str],
    labels: List[str]
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Merge scenarios from multiple files, adding source labels
    """
    merged_scenarios = []
    original_metadata = {}
    
    for file_path, label in zip(file_paths, labels):
        print(f"Processing {label}: {file_path}")
        
        data = load_graph_file(file_path)
        
        # Extract scenarios (handle both wrapped and unwrapped formats)
        scenarios = data.get('scenarios', data) if isinstance(data, dict) else data
        if not isinstance(scenarios, list):
            scenarios = [scenarios] if scenarios else []
        
        print(f"  - Found {len(scenarios)} scenarios")
        
        # Add scenarios with merge source label
        for scenario in scenarios:
            scenario_copy = scenario.copy()
            scenario_copy['merge_source'] = label
            merged_scenarios.append(scenario_copy)
        
        # Store original metadata
        original_metadata[label] = data.get('metadata', {}) if isinstance(data, dict) else {}
    
    return merged_scenarios, original_metadata


def build_merged_metadata(
    scenarios: List[Dict[str, Any]],
    original_metadata: Dict[str, Dict[str, Any]],
    labels: List[str]
) -> Dict[str, Any]:
    """
    Build metadata for merged scenarios
    """
    total_scenarios = len(scenarios)
    
    # Count by merge source
    source_counts = Counter(s.get('merge_source', 'unknown') for s in scenarios)
    
    # Count by risk type
    risk_type_counts = Counter(
        s.get('risk_type', 'unknown') for s in scenarios if s.get('risk_type')
    )
    
    # Count by mechanism
    mechanism_counts = Counter(
        s.get('mechanism', 'unknown') for s in scenarios if s.get('mechanism')
    )
    
    # Count by original source (original/augmented/etc)
    original_source_counts = Counter(
        s.get('source', 'unknown') for s in scenarios if s.get('source')
    )
    
    metadata = {
        'total_scenarios': total_scenarios,
        'merge_info': {
            'merged_files': {label: source_counts.get(label, 0) for label in labels},
            'original_metadata': original_metadata
        },
        'risk_type_counts': dict(risk_type_counts),
        'mechanism_counts': dict(mechanism_counts),
        'source_counts': dict(original_source_counts),
        'merge_breakdown': {
            'normalized': source_counts.get('normalized', 0),
            'scene_augmented': source_counts.get('scene_augmented', 0), 
            'hazard_removed': source_counts.get('hazard_removed', 0),
            'hazard_augmented': source_counts.get('hazard_augmented', 0)
        }
    }
    
    return metadata


def merge_multiple_graph_files(
    input_paths: List[str],
    labels: List[str],
    output_path: str
) -> None:
    """
    Merge multiple graph JSON files into one
    """
    if len(input_paths) != len(labels):
        raise ValueError("Number of input paths must match number of labels")
    
    print(f"Merging {len(input_paths)} graph files...")
    
    # Merge scenarios
    merged_scenarios, original_metadata = merge_multiple_scenarios(input_paths, labels)
    
    # Build metadata
    merged_metadata = build_merged_metadata(merged_scenarios, original_metadata, labels)
    
    # Create output
    output_data = {
        'metadata': merged_metadata,
        'scenarios': merged_scenarios
    }
    
    # Save merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerged {len(merged_scenarios)} scenarios to {output_path}")
    for label in labels:
        count = merged_metadata['merge_info']['merged_files'].get(label, 0)
        print(f"  - {label}: {count} scenarios")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple graph JSON files")
    parser.add_argument('--inputs', nargs='+', required=True, help='Input JSON files')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for each input file')
    parser.add_argument('--output', required=True, help='Output merged JSON file')
    
    args = parser.parse_args()
    
    # Check input files exist
    for input_path in args.inputs:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
    
    merge_multiple_graph_files(
        args.inputs,
        args.labels,
        args.output
    )


if __name__ == "__main__":
    main()

