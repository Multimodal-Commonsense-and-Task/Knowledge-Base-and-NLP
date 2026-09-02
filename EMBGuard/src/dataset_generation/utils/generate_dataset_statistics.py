#!/usr/bin/env python3
"""
Generate statistics for dataset CSV files.

This script reads a dataset CSV file and generates statistics based on:
- Category
- Subcategory
- Type
- Subtype
- MultiScenario

And their combinations.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any


def generate_statistics(csv_path: str, output_path: str) -> Dict[str, Any]:
    """
    Generate statistics from a dataset CSV file.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSON statistics file
        
    Returns:
        Dictionary containing statistics
    """
    # Read CSV and collect statistics
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    
    # Count by each dimension
    category_counts = Counter(row['Category'] for row in rows)
    subcategory_counts = Counter(row['Subcategory'] for row in rows)
    type_counts = Counter(row['Type'] for row in rows)
    subtype_counts = Counter(row['Subtype'] for row in rows)
    multiscenario_counts = Counter(row['MultiScenario'] for row in rows)
    
    # Count combinations
    category_subcategory = Counter((row['Category'], row['Subcategory']) for row in rows)
    category_type = Counter((row['Category'], row['Type']) for row in rows)
    category_subtype = Counter((row['Category'], row['Subtype']) for row in rows)
    subcategory_type = Counter((row['Subcategory'], row['Type']) for row in rows)
    type_subtype = Counter((row['Type'], row['Subtype']) for row in rows)
    multiscenario_subtype = Counter((row['MultiScenario'], row['Subtype']) for row in rows)
    
    # Create statistics dictionary
    stats = {
        'total_rows': total_rows,
        'by_category': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)),
        'by_subcategory': dict(sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True)),
        'by_type': dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)),
        'by_subtype': dict(sorted(subtype_counts.items(), key=lambda x: x[1], reverse=True)),
        'by_multiscenario': dict(sorted(multiscenario_counts.items(), key=lambda x: x[1], reverse=True)),
        'combinations': {
            'category_subcategory': {f"{k[0]} | {k[1]}": v for k, v in sorted(category_subcategory.items(), key=lambda x: x[1], reverse=True)},
            'category_type': {f"{k[0]} | {k[1]}": v for k, v in sorted(category_type.items(), key=lambda x: x[1], reverse=True)},
            'category_subtype': {f"{k[0]} | {k[1]}": v for k, v in sorted(category_subtype.items(), key=lambda x: x[1], reverse=True)},
            'subcategory_type': {f"{k[0]} | {k[1]}": v for k, v in sorted(subcategory_type.items(), key=lambda x: x[1], reverse=True)},
            'type_subtype': {f"{k[0]} | {k[1]}": v for k, v in sorted(type_subtype.items(), key=lambda x: x[1], reverse=True)},
            'multiscenario_subtype': {f"{k[0]} | {k[1]}": v for k, v in sorted(multiscenario_subtype.items(), key=lambda x: x[1], reverse=True)},
        }
    }
    
    # Save statistics
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate statistics for dataset CSV files.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON statistics file")
    args = parser.parse_args()
    
    try:
        stats = generate_statistics(args.csv, args.output)
        
        print(f"✓ Statistics saved: {args.output}")
        print(f"  Total rows: {stats['total_rows']}")
        print(f"  Categories: {len(stats['by_category'])}")
        print(f"  Subcategories: {len(stats['by_subcategory'])}")
        print(f"  Types: {len(stats['by_type'])}")
        print(f"  Subtypes: {len(stats['by_subtype'])}")
        print(f"  MultiScenario values: {len(stats['by_multiscenario'])}")
        
        return 0
    except Exception as e:
        print(f"Error generating statistics: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
