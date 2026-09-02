"""
Calculate correlation between two benchmark scores
Compares test set (EMBGuardTest) and heldout set scores across models
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.path import get_project_path
from utils.test_types import TEST_TYPE_CODES, extract_test_type_code_from_text, normalize_test_type_code


def load_evaluation_file(eval_file: Path) -> Dict:
    """Load evaluation JSON file"""
    with open(eval_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_model_name_from_path(path: Path) -> str:
    """Extract model name from path like results/EMBGuardTest/openai_gpt-4o/..."""
    # Get parent directory name (e.g., openai_gpt-4o)
    return path.parent.name


def parse_percentage_value(value_str: str) -> Optional[float]:
    """
    Parse percentage value from string like "56.8 (± 1.1)" or "56.8"
    Returns the mean value as a decimal (0.568 for 56.8%)
    Ignores CI part (everything after ± or parentheses)
    """
    if not value_str or value_str.strip() == "":
        return None
    
    # Remove whitespace
    value_str = str(value_str).strip()
    
    # Extract the first number (mean value) before any ±, (, or other non-numeric characters
    import re
    # Match number at the start (integer or decimal)
    # Stop at ±, (, or any non-numeric character (except decimal point)
    # Pattern: digits, optional decimal point, more digits, then stop at ± or (
    match = re.match(r'^([\d.]+)', value_str)
    if match:
        try:
            mean_pct = float(match.group(1))
            # Convert percentage to decimal
            return mean_pct / 100.0
        except (ValueError, TypeError):
            return None
    
    return None


def collect_scores_from_csv(
    csv_file: Path,
    metric: str = "potential_risk_accuracy",
) -> Dict[str, float]:
    """
    Collect scores from aggregated CSV file
    
    Args:
        csv_file: Path to CSV file (e.g., aggregated_results_overall_percentage.csv)
        metric: Metric column name (potential_risk, conditional_risk_type, conditional_hazard, etc.)
        
    Returns:
        Dictionary mapping model_name -> score
    """
    scores = {}
    
    if not csv_file.exists():
        print(f"Warning: CSV file not found: {csv_file}")
        return scores
    
    try:
        df = pd.read_csv(csv_file)
        
        # Map metric names to CSV column names
        metric_column_map = {
            "potential_risk_accuracy": "potential_risk",
            "conditional_risk_type_accuracy": "conditional_risk_type",
            "conditional_hazard_accuracy": "conditional_hazard",
            "risk_type_accuracy": "risk_type",
            "hazard_accuracy": "hazard",
        }
        
        column_name = metric_column_map.get(metric, metric.replace("_accuracy", ""))
        
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in CSV file")
            return scores
        
        if "model_name" not in df.columns:
            print(f"Warning: 'model_name' column not found in CSV file")
            return scores
        
        # Extract scores
        for _, row in df.iterrows():
            model_name = row["model_name"]
            value_str = row[column_name]
            
            # Parse percentage value
            score = parse_percentage_value(str(value_str))
            if score is not None:
                scores[model_name] = score
        
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        import traceback
        traceback.print_exc()
    
    return scores


def calculate_conditional_accuracies(data: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate conditional accuracies from evaluation data
    Returns (conditional_risk_type_accuracy, conditional_hazard_accuracy)
    """
    conditional_risk_type_acc = None
    conditional_hazard_acc = None
    
    if "evaluations" not in data:
        return conditional_risk_type_acc, conditional_hazard_acc
    
    # For risk_type: count only when potential_risk is correct AND risk_type was evaluated (not skipped)
    potential_risk_correct_risk_type_evaluated = 0
    conditional_risk_type_correct = 0
    
    # For hazard: count only when potential_risk is correct AND hazard was evaluated (not skipped)
    potential_risk_correct_hazard_evaluated = 0
    conditional_hazard_correct = 0
    
    for eval_item in data["evaluations"]:
        potential_risk_correct = eval_item.get("potential_risk", {}).get("correct", False)
        if potential_risk_correct:
            # Check risk_type: only if it was evaluated (not skipped)
            risk_type_info = eval_item.get("risk_type", {})
            risk_type_skipped = risk_type_info.get("skipped", False)
            risk_type_correct = risk_type_info.get("correct")
            
            if not risk_type_skipped and risk_type_correct is not None:
                # risk_type was evaluated (not skipped)
                potential_risk_correct_risk_type_evaluated += 1
                if risk_type_correct:
                    conditional_risk_type_correct += 1
            
            # Check hazard: only if it was evaluated (not skipped)
            hazard_info = eval_item.get("hazard", {})
            hazard_skipped = hazard_info.get("skipped", False)
            hazard_correct = hazard_info.get("correct")
            
            if not hazard_skipped and hazard_correct is not None:
                # hazard was evaluated (not skipped)
                potential_risk_correct_hazard_evaluated += 1
                if hazard_correct:
                    conditional_hazard_correct += 1
    
    if potential_risk_correct_risk_type_evaluated > 0:
        conditional_risk_type_acc = conditional_risk_type_correct / potential_risk_correct_risk_type_evaluated
    
    if potential_risk_correct_hazard_evaluated > 0:
        conditional_hazard_acc = conditional_hazard_correct / potential_risk_correct_hazard_evaluated
    
    return conditional_risk_type_acc, conditional_hazard_acc


def collect_test_set_scores(
    results_dir: Path,
    metric: str = "overall_accuracy",
    condition: Optional[str] = None,
) -> Dict[str, float]:
    """
    Collect test set scores for all models
    
    Args:
        results_dir: Path to results directory (e.g., results/EMBGuardTest)
        metric: Metric to extract (overall_accuracy, potential_risk_accuracy, conditional_risk_type_accuracy, conditional_hazard_accuracy)
        condition: Optional condition filter (e.g., "non-thinking", "thinking")
        
    Returns:
        Dictionary mapping model_name -> score
    """
    scores = {}
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return scores
    
    # Iterate through model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Find all evaluation files
        eval_files = list(model_dir.glob("*_evaluation.json"))
        
        if not eval_files:
            continue
        
        # Filter by condition if specified
        if condition:
            eval_files = [f for f in eval_files if condition in f.name]
        
        if not eval_files:
            continue
        
        # Aggregate scores across all EMBGuardTest types.
        total_items = 0
        weighted_sum = 0.0
        
        for eval_file in eval_files:
            try:
                data = load_evaluation_file(eval_file)
                stats = data.get("statistics", {})
                
                # Get metric value
                metric_value = None
                
                if metric == "conditional_risk_type_accuracy":
                    # Calculate conditional risk_type accuracy
                    conditional_risk_type_acc, _ = calculate_conditional_accuracies(data)
                    metric_value = conditional_risk_type_acc
                elif metric == "conditional_hazard_accuracy":
                    # Calculate conditional hazard accuracy
                    _, conditional_hazard_acc = calculate_conditional_accuracies(data)
                    metric_value = conditional_hazard_acc
                else:
                    # Standard metric from statistics
                    metric_value = stats.get(metric)
                
                if metric_value is None:
                    continue
                
                # Weight by total items
                total = stats.get("total", 0)
                if total == 0:
                    continue
                
                weighted_sum += metric_value * total
                total_items += total
                
            except Exception as e:
                print(f"Warning: Failed to load {eval_file}: {e}")
                continue
        
        if total_items > 0:
            # Calculate weighted average
            avg_score = weighted_sum / total_items
            scores[model_name] = avg_score
    
    return scores


def collect_heldout_set_scores(
    results_dir: Path,
    metric: str = "overall_accuracy",
    condition: Optional[str] = None,
    dataset_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Collect heldout set scores for all models
    
    Args:
        results_dir: Path to results directory (e.g., results/heldout_set)
        metric: Metric to extract (overall_accuracy, potential_risk_accuracy, conditional_risk_type_accuracy, conditional_hazard_accuracy)
        condition: Optional condition filter (e.g., "non-thinking", "thinking")
        dataset_type: Optional dataset type filter ("safe", "unsafe", or None for combined)
        
    Returns:
        Dictionary mapping model_name -> score
    """
    scores = {}
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return scores
    
    # Iterate through model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Find all evaluation files
        eval_files = list(model_dir.glob("*_evaluation.json"))
        
        if not eval_files:
            continue
        
        # Filter by condition if specified
        if condition:
            eval_files = [f for f in eval_files if condition in f.name]
        
        # Filter by dataset type if specified
        if dataset_type:
            eval_files = [f for f in eval_files if f"dataset_{dataset_type}" in f.name]
        
        if not eval_files:
            continue
        
        # Aggregate scores across datasets (safe, unsafe)
        total_items = 0
        weighted_sum = 0.0
        
        for eval_file in eval_files:
            try:
                data = load_evaluation_file(eval_file)
                stats = data.get("statistics", {})
                
                # Get metric value
                metric_value = None
                
                if metric == "conditional_risk_type_accuracy":
                    # Calculate conditional risk_type accuracy
                    conditional_risk_type_acc, _ = calculate_conditional_accuracies(data)
                    metric_value = conditional_risk_type_acc
                elif metric == "conditional_hazard_accuracy":
                    # Calculate conditional hazard accuracy
                    _, conditional_hazard_acc = calculate_conditional_accuracies(data)
                    metric_value = conditional_hazard_acc
                else:
                    # Standard metric from statistics
                    metric_value = stats.get(metric)
                
                if metric_value is None:
                    continue
                
                # Weight by total items
                total = stats.get("total", 0)
                if total == 0:
                    continue
                
                weighted_sum += metric_value * total
                total_items += total
                
            except Exception as e:
                print(f"Warning: Failed to load {eval_file}: {e}")
                continue
        
        if total_items > 0:
            # Calculate weighted average
            avg_score = weighted_sum / total_items
            scores[model_name] = avg_score
    
    return scores


def collect_test_set_scores_by_type(
    results_dir: Path,
    metric: str = "overall_accuracy",
    condition: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Collect test set scores for all models, grouped by EMBGuardTest type.
    
    Args:
        results_dir: Path to results directory (e.g., results/EMBGuardTest)
        metric: Metric to extract (overall_accuracy, potential_risk_accuracy, etc.)
        condition: Optional condition filter (e.g., "non-thinking", "thinking")
        
    Returns:
        Dictionary mapping type -> {model_name -> score}
    """
    scores_by_type = {test_type: {} for test_type in TEST_TYPE_CODES}
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return scores_by_type
    
    # Iterate through model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Find all evaluation files
        eval_files = list(model_dir.glob("*_evaluation.json"))
        
        if not eval_files:
            continue
        
        # Filter by condition if specified
        if condition:
            eval_files = [f for f in eval_files if condition in f.name]
        
        if not eval_files:
            continue
        
        # Group files by EMBGuardTest type code.
        files_by_type = {test_type: [] for test_type in TEST_TYPE_CODES}
        
        for eval_file in eval_files:
            # Extract type from filename (e.g., ..._HR_... or ..._test_dataset_HR_...)
            filename = eval_file.name
            test_type = extract_test_type_code_from_text(filename)
            if test_type in files_by_type:
                files_by_type[test_type].append(eval_file)
        
        # Calculate score for each type
        for test_type in TEST_TYPE_CODES:
            type_files = files_by_type[test_type]
            if not type_files:
                continue
            
            total_items = 0
            weighted_sum = 0.0
            
            for eval_file in type_files:
                try:
                    data = load_evaluation_file(eval_file)
                    stats = data.get("statistics", {})
                    
                    # Try to get metric from by_scenario_type first, then legacy by_type, then overall.
                    metric_value = None
                    by_scenario_type = {
                        normalize_test_type_code(type_key, default=type_key): type_stats
                        for type_key, type_stats in (stats.get("by_scenario_type") or stats.get("by_type", {})).items()
                    }
                    if test_type in by_scenario_type:
                        metric_value = by_scenario_type[test_type].get(metric)
                    
                    # Fallback to overall metric if not found in per-scenario statistics.
                    if metric_value is None:
                        metric_value = stats.get(metric)
                    
                    if metric_value is None:
                        continue
                    
                    # Weight by total items for this type
                    if test_type in by_scenario_type:
                        total = by_scenario_type[test_type].get("total", 0)
                    else:
                        total = stats.get("total", 0)
                    
                    if total == 0:
                        continue
                    
                    weighted_sum += metric_value * total
                    total_items += total
                    
                except Exception as e:
                    print(f"Warning: Failed to load {eval_file}: {e}")
                    continue
            
            if total_items > 0:
                avg_score = weighted_sum / total_items
                scores_by_type[test_type][model_name] = avg_score
    
    return scores_by_type


def calculate_correlation(
    scores1: Dict[str, float],
    scores2: Dict[str, float],
    method: str = "pearson",
) -> Tuple[float, float, pd.DataFrame]:
    """
    Calculate correlation between two sets of scores
    
    Args:
        scores1: First set of scores (model_name -> score)
        scores2: Second set of scores (model_name -> score)
        method: Correlation method ("pearson" or "spearman")
        
    Returns:
        Tuple of (correlation_coefficient, p_value, dataframe)
    """
    # Find common models
    common_models = set(scores1.keys()) & set(scores2.keys())
    
    if len(common_models) < 2:
        print(f"Warning: Only {len(common_models)} common models found. Need at least 2 for correlation.")
        return None, None, pd.DataFrame()
    
    # Create lists of scores for common models
    x_scores = [scores1[model] for model in sorted(common_models)]
    y_scores = [scores2[model] for model in sorted(common_models)]
    
    # Calculate correlation
    if method.lower() == "pearson":
        corr, p_value = pearsonr(x_scores, y_scores)
    elif method.lower() == "spearman":
        corr, p_value = spearmanr(x_scores, y_scores)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        "model": sorted(common_models),
        "score1": x_scores,
        "score2": y_scores,
    })
    
    return corr, p_value, df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate correlation between test set and heldout set scores"
    )
    parser.add_argument(
        "--test-results-dir",
        type=str,
        default=None,
        help="Path to test set results directory (default: results/EMBGuardTest). If --test-csv is specified, this is ignored."
    )
    parser.add_argument(
        "--heldout-results-dir",
        type=str,
        default=None,
        help="Path to heldout set results directory (default: results/heldout_set). If --heldout-csv is specified, this is ignored."
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to test set aggregated CSV file (e.g., results/EMBGuardTest/aggregated_results_overall_percentage.csv). If specified, uses CSV instead of scanning results directory."
    )
    parser.add_argument(
        "--heldout-csv",
        type=str,
        default=None,
        help="Path to heldout set aggregated CSV file (e.g., results/heldout_set/aggregated_results_overall_percentage.csv). If specified, uses CSV instead of scanning results directory."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["overall_accuracy", "potential_risk_accuracy", "risk_type_accuracy", "hazard_accuracy", 
                 "conditional_risk_type_accuracy", "conditional_hazard_accuracy"],
        help="Metric to use for correlation. If not specified, calculates for all three: potential_risk, conditional_risk_type, conditional_hazard"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Condition filter (e.g., 'non-thinking', 'thinking'). If not specified, uses all conditions."
    )
    parser.add_argument(
        "--heldout-dataset-type",
        type=str,
        default=None,
        choices=["safe", "unsafe", None],
        help="Heldout dataset type filter ('safe', 'unsafe', or None for combined)"
    )
    parser.add_argument(
        "--correlation-method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method (default: pearson)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save correlation results (CSV format). If not specified, only prints to console."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plot"
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Path to save plot (default: {output_file}.png or correlation_plot.png)"
    )
    
    args = parser.parse_args()
    
    # Get project path
    project_path = get_project_path()
    
    # Resolve paths
    if args.test_csv:
        test_csv_file = project_path / args.test_csv
        test_results_dir = None
    else:
        test_csv_file = None
        test_results_dir = project_path / (args.test_results_dir or "results/EMBGuardTest")
    
    if args.heldout_csv:
        heldout_csv_file = project_path / args.heldout_csv
        heldout_results_dir = None
    else:
        heldout_csv_file = None
        heldout_results_dir = project_path / (args.heldout_results_dir or "results/heldout_set")
    
    # Determine which metrics to calculate
    if args.metric:
        metrics_to_calculate = [args.metric]
    else:
        # Default: calculate for all three metrics
        metrics_to_calculate = [
            "potential_risk_accuracy",
            "conditional_risk_type_accuracy",
            "conditional_hazard_accuracy"
        ]
    
    # Calculate both Pearson and Spearman correlations
    correlation_methods = ["pearson", "spearman"]
    
    # Calculate correlation for each metric and each method
    all_results = {}
    
    for metric in metrics_to_calculate:
        all_results[metric] = {}
        
        for corr_method in correlation_methods:
            print("\n" + "="*60)
            print(f"Correlation Analysis: {metric} ({corr_method.title()})")
            print("="*60)
            if test_csv_file:
                print(f"Test set CSV: {test_csv_file}")
            else:
                print(f"Test set results: {test_results_dir}")
            if heldout_csv_file:
                print(f"Heldout set CSV: {heldout_csv_file}")
            else:
                print(f"Heldout set results: {heldout_results_dir}")
            if args.condition:
                print(f"Condition filter: {args.condition}")
            if args.heldout_dataset_type:
                print(f"Heldout dataset type: {args.heldout_dataset_type}")
            print(f"Correlation method: {corr_method}")
            print("="*60)
            print()
            
            # Collect scores
            print("Collecting test set scores...")
            if test_csv_file:
                test_scores = collect_scores_from_csv(
                    test_csv_file,
                    metric=metric,
                )
                print(f"Found {len(test_scores)} models in test set (from CSV)")
            else:
                test_scores = collect_test_set_scores(
                    test_results_dir,
                    metric=metric,
                    condition=args.condition,
                )
                print(f"Found {len(test_scores)} models in test set")
            
            print("\nCollecting heldout set scores...")
            if heldout_csv_file:
                heldout_scores = collect_scores_from_csv(
                    heldout_csv_file,
                    metric=metric,
                )
                print(f"Found {len(heldout_scores)} models in heldout set (from CSV)")
            else:
                heldout_scores = collect_heldout_set_scores(
                    heldout_results_dir,
                    metric=metric,
                    condition=args.condition,
                    dataset_type=args.heldout_dataset_type,
                )
                print(f"Found {len(heldout_scores)} models in heldout set")
        
            # Calculate correlation
            print(f"\nCalculating {corr_method} correlation...")
            corr, p_value, df = calculate_correlation(
                test_scores,
                heldout_scores,
                method=corr_method,
            )
            
            if corr is None:
                print(f"Error: Could not calculate {corr_method} correlation for {metric} (insufficient data)")
                continue
            
            # Store results
            all_results[metric][corr_method] = {
                "corr": corr,
                "p_value": p_value,
                "df": df,
            }
            
            # Print results
            print("\n" + "="*60)
            print(f"Results for {metric} ({corr_method.title()} correlation)")
            print("="*60)
            print(f"Number of models: {len(df)}")
            print(f"Correlation ({corr_method}): {corr:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
            print()
            
            # Print model scores (only for first method to avoid duplication)
            if corr_method == correlation_methods[0]:
                print("Model Scores (Overall):")
                print("-"*60)
                print(f"{'Model':<40} {'Test Set':<15} {'Heldout Set':<15}")
                print("-"*60)
                for _, row in df.iterrows():
                    print(f"{row['model']:<40} {row['score1']:<15.4f} {row['score2']:<15.4f}")
                print()
    
    # Print summary of all metrics
    print("\n" + "="*60)
    print("Summary: All Metrics")
    print("="*60)
    print(f"{'Metric':<35} {'Method':<10} {'Correlation':<15} {'P-value':<15} {'Significance':<15}")
    print("-"*60)
    for metric, method_results in all_results.items():
        metric_display = metric.replace("_accuracy", "").replace("_", " ").title()
        for corr_method, result in method_results.items():
            corr = result["corr"]
            p_value = result["p_value"]
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"{metric_display:<35} {corr_method.title():<10} {corr:<15.4f} {p_value:<15.4f} {sig:<15}")
    
    # Save to file if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results for each metric and each method
        for metric, method_results in all_results.items():
            metric_suffix = metric.replace("_accuracy", "").replace("_", "-")
            for corr_method, result in method_results.items():
                method_output_path = output_path.parent / f"{output_path.stem}_{metric_suffix}_{corr_method}.csv"
                result['df'].to_csv(method_output_path, index=False)
                print(f"Results for {metric} ({corr_method}) saved to: {method_output_path}")
        
        # Save summary
        summary_path = output_path.with_suffix('.summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Correlation Analysis Summary\n")
            f.write("="*60 + "\n")
            if test_csv_file:
                f.write(f"Test set CSV: {test_csv_file}\n")
            else:
                f.write(f"Test set results: {test_results_dir}\n")
            if heldout_csv_file:
                f.write(f"Heldout set CSV: {heldout_csv_file}\n")
            else:
                f.write(f"Heldout set results: {heldout_results_dir}\n")
            f.write(f"Metrics: {', '.join(metrics_to_calculate)}\n")
            f.write(f"Correlation methods: {', '.join(correlation_methods)}\n")
            if args.condition:
                f.write(f"Condition filter: {args.condition}\n")
            if args.heldout_dataset_type:
                f.write(f"Heldout dataset type: {args.heldout_dataset_type}\n")
            f.write("\n")
            
            for metric, method_results in all_results.items():
                f.write(f"\n{metric}:\n")
                for corr_method, result in method_results.items():
                    f.write(f"  {corr_method.title()} correlation:\n")
                    f.write(f"    Number of models: {len(result['df'])}\n")
                    f.write(f"    Correlation: {result['corr']:.4f}\n")
                    f.write(f"    P-value: {result['p_value']:.4f}\n")
                    f.write(f"    Significance: {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}\n")
        
        print(f"\nSummary saved to: {summary_path}")
    
    # Generate visualization if requested
    if args.plot or args.plot_file:
        # Create plots for each metric and each method
        for metric, method_results in all_results.items():
            for corr_method, result in method_results.items():
                df = result['df']
                corr = result['corr']
                p_value = result['p_value']
                
                plot_file = args.plot_file
                if plot_file is None:
                    if args.output_file:
                        metric_suffix = metric.replace("_accuracy", "").replace("_", "-")
                        plot_file = str(Path(args.output_file).parent / f"{Path(args.output_file).stem}_{metric_suffix}_{corr_method}.png")
                    else:
                        metric_suffix = metric.replace("_accuracy", "").replace("_", "-")
                        plot_file = f"correlation_plot_{metric_suffix}_{corr_method}.png"
                else:
                    # If plot_file is specified, add metric and method suffix
                    metric_suffix = metric.replace("_accuracy", "").replace("_", "-")
                    plot_file = str(Path(plot_file).parent / f"{Path(plot_file).stem}_{metric_suffix}_{corr_method}.png")
                
                plot_path = Path(plot_file)
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create scatter plot
                plt.figure(figsize=(10, 8))
                sns.set_style("whitegrid")
                
                # Plot scatter
                plt.scatter(df['score1'], df['score2'], s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
                
                # Add regression line
                z = np.polyfit(df['score1'], df['score2'], 1)
                p = np.poly1d(z)
                plt.plot(df['score1'], p(df['score1']), "r--", alpha=0.8, linewidth=2, label=f'Linear fit (r={corr:.3f})')
                
                # Add model labels
                for _, row in df.iterrows():
                    # Shorten model names for readability
                    model_name = row['model']
                    # Remove provider prefix if present
                    if '_' in model_name:
                        parts = model_name.split('_', 1)
                        if len(parts) > 1:
                            model_name = parts[1]
                    plt.annotate(model_name, (row['score1'], row['score2']), 
                                xytext=(5, 5), textcoords='offset points', 
                                fontsize=8, alpha=0.7)
                
                # Labels and title
                metric_display = metric.replace("_accuracy", "").replace("_", " ").title()
                plt.xlabel(f'Test Set ({metric_display})', fontsize=12, fontweight='bold')
                plt.ylabel(f'Heldout Set ({metric_display})', fontsize=12, fontweight='bold')
                
                # Build title with correlation and p-value
                title = f'Correlation: {corr_method.title()} r = {corr:.3f}'
                if p_value < 0.001:
                    title += ' (***'
                elif p_value < 0.01:
                    title += ' (**'
                elif p_value < 0.05:
                    title += ' (*'
                else:
                    title += ' (ns'
                
                # Add p-value to title
                if p_value < 0.001:
                    title += f', p < 0.001)'
                elif p_value < 0.01:
                    title += f', p = {p_value:.3f})'
                elif p_value < 0.05:
                    title += f', p = {p_value:.3f})'
                else:
                    title += f', p = {p_value:.3f})'
                
                plt.title(title, fontsize=14, fontweight='bold', pad=20)
                
                # Add grid
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                # Set axis limits to 0-1 (must be set after axis('equal') or instead of it)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                
                # Set equal aspect ratio manually (1:1) while maintaining 0-1 limits
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                
                # Tight layout
                plt.tight_layout()
                
                # Save plot
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot for {metric} ({corr_method}) saved to: {plot_path}")
                plt.close()
        
        # Optionally show plot
        # plt.show()


if __name__ == "__main__":
    main()
