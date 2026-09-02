#!/usr/bin/env python3
"""
Gather batch scenario information from raw batch files
Creates a unified dataset with batch job information for each scenario.

Output format:
{
    "metadata": {
        "total_scenarios": 137,
        "batch_jobs": ["batches/xxx", "batches/yyy"],
        "timestamps": [1234567890, 1234567891]
    },
    "scenarios": [
        {
            "id": "Fire_Risk_Open_Flame_Ignition_001",
            "req_id": "Fire_Risk_Open_Flame_Ignition_001-normalized",
            "batch_job": "batches/xxx",
            "batch_job_id": "xxx",
            "timestamp": 1234567890,
            "model_input": "prompt text...",
            "risk_type": "Fire_Risk",
            "mechanism": "Open_Flame_Ignition",
            ...
        },
        ...
    ]
}
"""

import json
import os
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_batch_scenario_files(raw_dir: str) -> List[Dict[str, Any]]:
    """Load all batch scenario files from raw directory"""
    all_scenarios = []
    
    # Find all batch_scenarios_*.json files
    pattern = os.path.join(raw_dir, "batch_scenarios_*.json")
    batch_files = glob.glob(pattern)
    
    print(f"Found {len(batch_files)} batch scenario files")
    
    for batch_file in batch_files:
        print(f"  Loading: {batch_file}")
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
                all_scenarios.extend(scenarios)
                print(f"    Added {len(scenarios)} scenarios")
        except Exception as e:
            print(f"    Error loading {batch_file}: {e}")
    
    return all_scenarios


def create_metadata(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create metadata summary"""
    batch_jobs = list(set(s.get('batch_job', '') for s in scenarios if s.get('batch_job')))
    timestamps = list(set(s.get('timestamp', 0) for s in scenarios if s.get('timestamp')))
    
    risk_types = {}
    mechanisms = {}
    
    for scenario in scenarios:
        risk_type = scenario.get('risk_type', 'Unknown')
        mechanism = scenario.get('mechanism', 'Unknown')
        
        risk_types[risk_type] = risk_types.get(risk_type, 0) + 1
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
    
    return {
        "total_scenarios": len(scenarios),
        "batch_jobs": sorted(batch_jobs),
        "timestamps": sorted(timestamps),
        "risk_type_counts": risk_types,
        "mechanism_counts": mechanisms
    }


def gather_batch_scenarios(raw_dir: str, output_file: str):
    """Main function to gather batch scenarios"""
    print(f"Gathering batch scenarios from: {raw_dir}")
    
    # Load all batch scenario files
    scenarios = load_batch_scenario_files(raw_dir)
    
    if not scenarios:
        print("No batch scenarios found!")
        return
    
    # Create metadata
    metadata = create_metadata(scenarios)
    
    # Create final output
    output_data = {
        "metadata": metadata,
        "scenarios": scenarios
    }
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nGathered batch scenarios saved to: {output_file}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Batch jobs: {len(metadata['batch_jobs'])}")
    
    # Print summary
    print(f"\nRisk Type distribution:")
    for risk_type, count in sorted(metadata['risk_type_counts'].items()):
        print(f"  {risk_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Gather batch scenario information')
    parser.add_argument('--raw-dir', required=True, help='Raw directory containing batch_scenarios_*.json files')
    parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"Error: Raw directory not found: {raw_dir}")
        return
    
    gather_batch_scenarios(str(raw_dir), args.output)


if __name__ == '__main__':
    main()
