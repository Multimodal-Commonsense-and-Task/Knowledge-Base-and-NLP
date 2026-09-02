#!/usr/bin/env python3
"""
Aggregate evaluation results across multiple runs
Calculates averages and 95% confidence intervals for each model and test type
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy import stats
import csv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.test_types import (
    TEST_TYPE_CODE_TO_LABEL,
    TEST_TYPE_CODES,
    extract_test_type_code_from_text,
    normalize_test_type_code,
    test_type_label,
    test_type_slug,
)


TEST_TYPE_LABELS = TEST_TYPE_CODE_TO_LABEL


def load_evaluation_file(json_path: Path) -> Optional[Dict]:
    """Load evaluation JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def calculate_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval using t-distribution
    
    Args:
        data: List of values
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound, margin)
    """
    if len(data) < 2:
        val = data[0] if data else 0.0
        return (val, val, 0.0)
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    std_err = stats.sem(data_array)  # Standard error of the mean
    
    # Calculate t-value for confidence interval
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, len(data_array) - 1)
    
    margin = t_value * std_err
    return (mean - margin, mean + margin, margin)


def aggregate_results(
    results_dir: Path,
    output_file: Optional[Path] = None,
    run_prefix: str = "Run_"
) -> Dict:
    """
    Aggregate evaluation results across all runs
    
    Args:
        results_dir: Directory containing Run_1, Run_2, ... folders
        output_file: Output text file path (auto-generated if None)
        run_prefix: Prefix for run folders (default: "Run_")
    
    Returns:
        Dictionary with aggregated statistics
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    # Find all run directories
    run_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith(run_prefix)])
    
    if not run_dirs:
        raise ValueError(f"No run directories found in {results_dir}")
    
    # Structure: model_name -> test_type -> metric -> [values across runs]
    model_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Structure for individual run statistics: run_dir -> model_name -> test_type -> metrics
    run_statistics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Collect data from all runs
    for run_dir in run_dirs:
        run_num = run_dir.name.replace(run_prefix, "")
        print(f"Processing {run_dir.name}...")
        
        # Find all model directories in this run
        for model_dir in run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            # First pass: collect all test type data for this model in this run
            # Structure: test_type -> metric -> (value, count)
            run_type_data = defaultdict(lambda: defaultdict(lambda: {"value": None, "count": 0}))
            
            # Find all evaluation JSON files
            eval_files = list(model_dir.glob("*_evaluation.json"))
            
            for eval_file in eval_files:
                # Load evaluation data first
                data = load_evaluation_file(eval_file)
                if not data or "statistics" not in data:
                    continue
                
                stats = data["statistics"]
                filename = eval_file.stem
                
                # Check if this is heldout_set (safe/unsafe structure)
                is_heldout = "heldout_set" in str(run_dir) or "safe" in filename.lower() or "unsafe" in filename.lower()
                
                # Calculate conditional accuracies (risk_type and hazard accuracy when potential_risk is correct)
                # This is calculated from the evaluations array, considering all items in the file
                # Only count cases where risk_type/hazard were actually evaluated (not skipped)
                conditional_risk_type_acc = None
                conditional_hazard_acc = None
                if "evaluations" in data:
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
                
                if is_heldout:
                    # For heldout_set, file itself represents a type (safe or unsafe)
                    # Check "unsafe" first because "safe" is contained in "unsafe"
                    if "unsafe" in filename.lower():
                        test_type = "unsafe"
                    elif "safe" in filename.lower():
                        test_type = "safe"
                    else:
                        continue
                    
                    # For heldout_set, use top-level statistics directly
                    file_total = stats.get("total", 0)
                    type_metrics = {
                        "potential_risk_accuracy": stats.get("potential_risk_accuracy"),
                        "risk_type_accuracy": stats.get("risk_type_accuracy"),
                        "hazard_accuracy": stats.get("hazard_accuracy"),
                        "overall_accuracy": stats.get("overall_accuracy"),
                        "conditional_risk_type_accuracy": conditional_risk_type_acc,
                        "conditional_hazard_accuracy": conditional_hazard_acc,
                    }
                    
                    # Store for individual run statistics
                    for metric, value in type_metrics.items():
                        if value is not None:
                            if metric not in run_statistics[run_dir][model_name][test_type]:
                                run_statistics[run_dir][model_name][test_type][metric] = value
                            if run_type_data[test_type][metric]["value"] is None:
                                run_type_data[test_type][metric]["value"] = value
                                run_type_data[test_type][metric]["count"] = file_total
                    
                    # Store for aggregation (across runs)
                    for metric, value in type_metrics.items():
                        if value is not None:
                            model_data[model_name][test_type][metric].append(value)
                
                else:
                    # For test_set (EMBGuardTest), extract test type from filename
                    test_type = None
                    test_type = extract_test_type_code_from_text(filename)
                    
                    if not test_type:
                        # Try to get from statistics by_scenario_type, then legacy by_type.
                        by_scenario_type = stats.get("by_scenario_type") or stats.get("by_type", {})
                        if by_scenario_type:
                            test_type = normalize_test_type_code(
                                list(by_scenario_type.keys())[0],
                                default=list(by_scenario_type.keys())[0],
                            )
                    
                    if not test_type:
                        continue
                    
                    # Store for specific test type (individual run statistics)
                    by_scenario_type = {
                        normalize_test_type_code(type_key, default=type_key): type_stats
                        for type_key, type_stats in (stats.get("by_scenario_type") or stats.get("by_type", {})).items()
                    }
                    if test_type in by_scenario_type:
                        type_stats = by_scenario_type[test_type]
                        type_total = type_stats.get("total", 0)  # Get data count for this test type
                        type_metrics = {
                            "potential_risk_accuracy": type_stats.get("potential_risk_accuracy"),
                            "risk_type_accuracy": type_stats.get("risk_type_accuracy"),
                            "hazard_accuracy": type_stats.get("hazard_accuracy"),
                            "overall_accuracy": type_stats.get("overall_accuracy"),
                            "conditional_risk_type_accuracy": conditional_risk_type_acc,
                            "conditional_hazard_accuracy": conditional_hazard_acc,
                        }
                        # Store for individual run statistics (test type specific)
                        # Only store if not already set (avoid overwriting)
                        for metric, value in type_metrics.items():
                            if value is not None:
                                if metric not in run_statistics[run_dir][model_name][test_type]:
                                    run_statistics[run_dir][model_name][test_type][metric] = value
                                if run_type_data[test_type][metric]["value"] is None:
                                    run_type_data[test_type][metric]["value"] = value
                                    run_type_data[test_type][metric]["count"] = type_total
                        
                        # Store for aggregation (across runs)
                        for metric, value in type_metrics.items():
                            if value is not None:
                                model_data[model_name][test_type][metric].append(value)
            
            # Calculate overall from all scenario types for this run using weighted average
            # Collect all scenario_type values with their counts
            overall_metrics = defaultdict(lambda: {"weighted_sum": 0.0, "total_count": 0})
            for test_type, metrics in run_type_data.items():
                for metric, data_info in metrics.items():
                    value = data_info["value"]
                    count = data_info["count"]
                    if value is not None and not (isinstance(value, float) and np.isnan(value)) and count > 0:
                        overall_metrics[metric]["weighted_sum"] += value * count
                        overall_metrics[metric]["total_count"] += count
            
            # Store overall for individual run statistics (weighted average of all scenario types)
            for metric, metric_data in overall_metrics.items():
                total_count = metric_data["total_count"]
                if total_count > 0:
                    # Calculate weighted average: sum(value * count) / sum(count)
                    weighted_avg = metric_data["weighted_sum"] / total_count
                    run_statistics[run_dir][model_name]["overall"][metric] = weighted_avg
                    
                    # Also store for aggregation across runs (only once per run)
                    model_data[model_name]["overall"][metric].append(weighted_avg)
    
    # Calculate statistics
    aggregated = {}
    for model_name, type_data in model_data.items():
        aggregated[model_name] = {}
        
        for test_type, metrics in type_data.items():
            aggregated[model_name][test_type] = {}
            
            for metric, values in metrics.items():
                if not values:
                    continue
                
                # Filter out None and NaN values
                values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if not values:
                    continue
                
                mean = np.mean(values)
                ci_lower, ci_upper, ci_margin = calculate_ci(values)
                
                aggregated[model_name][test_type][metric] = {
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_margin": ci_margin,
                    "n_runs": len(values),
                    "std": np.std(values) if len(values) > 1 else 0.0,
                }
    
    # Generate output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("EMBGuard Evaluation Results Summary")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Sort models for consistent output
    sorted_models = sorted(aggregated.keys())
    
    for model_name in sorted_models:
        output_lines.append("=" * 80)
        output_lines.append(f"Model: {model_name}")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        model_data = aggregated[model_name]
        
        # Overall results
        if "overall" in model_data:
            output_lines.append("Overall (All Types Combined):")
            output_lines.append("-" * 80)
            overall = model_data["overall"]
            
            for metric in ["overall_accuracy", "potential_risk_accuracy", 
                          "risk_type_accuracy", "hazard_accuracy",
                          "conditional_risk_type_accuracy", "conditional_hazard_accuracy"]:
                if metric in overall:
                    stat = overall[metric]
                    metric_name = metric.replace('_', ' ').title()
                    if metric.startswith("conditional"):
                        metric_name = metric_name.replace("Conditional ", "Conditional ").replace("Accuracy", "Accuracy (when potential_risk correct)")
                    output_lines.append(
                        f"  {metric_name}: "
                        f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f} "
                        f"(95% CI, n={stat['n_runs']}, std={stat['std']:.4f})"
                    )
            output_lines.append("")
        
        # Results by scenario_type
        test_types = list(TEST_TYPE_CODES)
        for test_type in test_types:
            if test_type not in model_data:
                continue
            
            output_lines.append(f"Scenario Type: {test_type_label(test_type)}")
            output_lines.append("-" * 80)
            type_data = model_data[test_type]
            
            for metric in ["overall_accuracy", "potential_risk_accuracy", 
                          "risk_type_accuracy", "hazard_accuracy",
                          "conditional_risk_type_accuracy", "conditional_hazard_accuracy"]:
                if metric in type_data:
                    stat = type_data[metric]
                    metric_name = metric.replace('_', ' ').title()
                    if metric.startswith("conditional"):
                        metric_name = metric_name.replace("Conditional ", "Conditional ").replace("Accuracy", "Accuracy (when potential_risk correct)")
                    output_lines.append(
                        f"  {metric_name}: "
                        f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f} "
                        f"(95% CI, n={stat['n_runs']}, std={stat['std']:.4f})"
                    )
            output_lines.append("")
    
    output_text = "\n".join(output_lines)
    
    # Save aggregated results to file
    if output_file is None:
        output_file = results_dir / "aggregated_results.txt"
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"\nAggregated results saved to: {output_file}")
    
    # Generate CSV file for easy copy-paste
    csv_file = output_file.parent / "aggregated_results_overall.csv"
    csv_lines = []
    csv_lines.append("model_name,potential_risk,risk_type,hazard,conditional_risk_type,conditional_hazard")
    
    for model_name in sorted_models:
        model_data = aggregated[model_name]
        if "overall" not in model_data:
            continue
        
        overall = model_data["overall"]
        
        # Get values with CI format: mean ± margin
        potential_risk = ""
        risk_type = ""
        hazard = ""
        conditional_risk_type = ""
        conditional_hazard = ""
        
        if "potential_risk_accuracy" in overall:
            stat = overall["potential_risk_accuracy"]
            potential_risk = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
        
        if "risk_type_accuracy" in overall:
            stat = overall["risk_type_accuracy"]
            risk_type = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
        
        if "hazard_accuracy" in overall:
            stat = overall["hazard_accuracy"]
            hazard = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
        
        if "conditional_risk_type_accuracy" in overall:
            stat = overall["conditional_risk_type_accuracy"]
            conditional_risk_type = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
        
        if "conditional_hazard_accuracy" in overall:
            stat = overall["conditional_hazard_accuracy"]
            conditional_hazard = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
        
        csv_lines.append(f"{model_name},{potential_risk},{risk_type},{hazard},{conditional_risk_type},{conditional_hazard}")
    
    csv_text = "\n".join(csv_lines)
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(csv_text)
    
    print(f"CSV results saved to: {csv_file}")
    
    # Generate percentage version CSV (overall_percentage.csv)
    percentage_csv_file = output_file.parent / "aggregated_results_overall_percentage.csv"
    percentage_csv_lines = []
    percentage_csv_lines.append("model_name,potential_risk,risk_type,hazard,conditional_risk_type,conditional_hazard")
    
    for model_name in sorted_models:
        model_data = aggregated[model_name]
        if "overall" not in model_data:
            continue
        
        overall = model_data["overall"]
        
        # Get values with CI format: mean ± margin (as percentage, 1 decimal place)
        potential_risk = ""
        risk_type = ""
        hazard = ""
        conditional_risk_type = ""
        conditional_hazard = ""
        
        if "potential_risk_accuracy" in overall:
            stat = overall["potential_risk_accuracy"]
            mean_pct = stat['mean'] * 100
            margin_pct = stat['ci_margin'] * 100
            potential_risk = f"{mean_pct:.1f} (± {margin_pct:.1f})"
        
        if "risk_type_accuracy" in overall:
            stat = overall["risk_type_accuracy"]
            mean_pct = stat['mean'] * 100
            margin_pct = stat['ci_margin'] * 100
            risk_type = f"{mean_pct:.1f} (± {margin_pct:.1f})"
        
        if "hazard_accuracy" in overall:
            stat = overall["hazard_accuracy"]
            mean_pct = stat['mean'] * 100
            margin_pct = stat['ci_margin'] * 100
            hazard = f"{mean_pct:.1f} (± {margin_pct:.1f})"
        
        if "conditional_risk_type_accuracy" in overall:
            stat = overall["conditional_risk_type_accuracy"]
            mean_pct = stat['mean'] * 100
            margin_pct = stat['ci_margin'] * 100
            conditional_risk_type = f"{mean_pct:.1f} (± {margin_pct:.1f})"
        
        if "conditional_hazard_accuracy" in overall:
            stat = overall["conditional_hazard_accuracy"]
            mean_pct = stat['mean'] * 100
            margin_pct = stat['ci_margin'] * 100
            conditional_hazard = f"{mean_pct:.1f} (± {margin_pct:.1f})"
        
        percentage_csv_lines.append(f"{model_name},{potential_risk},{risk_type},{hazard},{conditional_risk_type},{conditional_hazard}")
    
    percentage_csv_text = "\n".join(percentage_csv_lines)
    with open(percentage_csv_file, 'w', encoding='utf-8') as f:
        f.write(percentage_csv_text)
    
    print(f"Percentage CSV results saved to: {percentage_csv_file}")
    
    # Generate scenario_type-based CSV for EMBGuardTest conditions.
    type_csv_file = output_file.parent / "aggregated_results_by_scenario_type.csv"
    type_csv_lines = []
    type_csv_lines.append("model_name,scenario_type,potential_risk,risk_type,hazard,conditional_risk_type,conditional_hazard")
    
    test_types = list(TEST_TYPE_CODES)
    for model_name in sorted_models:
        model_data = aggregated[model_name]
        for test_type in test_types:
            if test_type not in model_data:
                continue
            
            type_data = model_data[test_type]
            
            # Get values with CI format: mean ± margin
            potential_risk = ""
            risk_type = ""
            hazard = ""
            conditional_risk_type = ""
            conditional_hazard = ""
            
            if "potential_risk_accuracy" in type_data:
                stat = type_data["potential_risk_accuracy"]
                potential_risk = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
            
            if "risk_type_accuracy" in type_data:
                stat = type_data["risk_type_accuracy"]
                risk_type = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
            
            if "hazard_accuracy" in type_data:
                stat = type_data["hazard_accuracy"]
                hazard = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
            
            # Only set conditional accuracies if risk_type and hazard exist
            # For benign scenario types where there are no risky cases,
            # conditional accuracies should also be empty
            if risk_type or hazard:
                if "conditional_risk_type_accuracy" in type_data:
                    stat = type_data["conditional_risk_type_accuracy"]
                    conditional_risk_type = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
                
                if "conditional_hazard_accuracy" in type_data:
                    stat = type_data["conditional_hazard_accuracy"]
                    conditional_hazard = f"{stat['mean']:.4f} ± {stat['ci_margin']:.4f}"
            
            # For benign scenario types where there are no risky cases,
            # risk_type, hazard, conditional_risk_type, and conditional_hazard will be empty,
            # but we still show potential_risk
            scenario_type = test_type_slug(test_type)
            type_csv_lines.append(f"{model_name},{scenario_type},{potential_risk},{risk_type},{hazard},{conditional_risk_type},{conditional_hazard}")
    
    type_csv_text = "\n".join(type_csv_lines)
    with open(type_csv_file, 'w', encoding='utf-8') as f:
        f.write(type_csv_text)
    
    print(f"Type-based CSV results saved to: {type_csv_file}")
    
    # Generate risk_type-based CSV (Fire_Risk, Electric_Risk, etc.)
    # Need to analyze evaluations to get risk_type breakdown
    risk_type_csv_file = output_file.parent / "aggregated_results_by_risk_type.csv"
    risk_type_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> risk_type -> metric -> [values]
    
    # Collect risk_type data from all runs
    for run_dir in run_dirs:
        for model_dir in run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            for eval_file in model_dir.glob("*_evaluation.json"):
                data = load_evaluation_file(eval_file)
                if not data or "evaluations" not in data:
                    continue
                
                # Group evaluations by risk_type
                risk_type_groups = defaultdict(lambda: {
                    "correct": 0, 
                    "total": 0, 
                    "potential_risk_correct": 0, 
                    "hazard_correct": 0,
                    "risk_type_correct": 0
                })
                
                for eval_item in data["evaluations"]:
                    if "risk_type" not in eval_item:
                        continue
                    
                    risk_type_info = eval_item["risk_type"]
                    expected_risk_type = risk_type_info.get("expected", "")
                    
                    if not expected_risk_type or expected_risk_type.lower() == "none":
                        continue
                    
                    risk_type_groups[expected_risk_type]["total"] += 1
                    
                    if eval_item.get("overall_correct", False):
                        risk_type_groups[expected_risk_type]["correct"] += 1
                    
                    if eval_item.get("potential_risk", {}).get("correct", False):
                        risk_type_groups[expected_risk_type]["potential_risk_correct"] += 1
                    
                    if eval_item.get("hazard", {}).get("correct", False):
                        risk_type_groups[expected_risk_type]["hazard_correct"] += 1
                    
                    if risk_type_info.get("correct", False):
                        risk_type_groups[expected_risk_type]["risk_type_correct"] += 1
                
                # Calculate accuracies for each risk_type
                for risk_type, counts in risk_type_groups.items():
                    if counts["total"] > 0:
                        overall_acc = counts["correct"] / counts["total"]
                        potential_risk_acc = counts["potential_risk_correct"] / counts["total"]
                        hazard_acc = counts["hazard_correct"] / counts["total"]
                        risk_type_acc = counts["risk_type_correct"] / counts["total"]
                        
                        risk_type_data[model_name][risk_type]["overall_accuracy"].append(overall_acc)
                        risk_type_data[model_name][risk_type]["potential_risk_accuracy"].append(potential_risk_acc)
                        risk_type_data[model_name][risk_type]["risk_type_accuracy"].append(risk_type_acc)
                        risk_type_data[model_name][risk_type]["hazard_accuracy"].append(hazard_acc)
    
    # Generate risk_type CSV
    risk_type_csv_lines = []
    risk_type_csv_lines.append("model_name,risk_type,potential_risk,risk_type_acc,hazard")
    
    # Get all unique risk types
    all_risk_types = set()
    for model_data in risk_type_data.values():
        all_risk_types.update(model_data.keys())
    all_risk_types = sorted(all_risk_types)
    
    for model_name in sorted_models:
        if model_name not in risk_type_data:
            continue
        
        model_risk_types = risk_type_data[model_name]
        for risk_type in all_risk_types:
            if risk_type not in model_risk_types:
                continue
            
            metrics = model_risk_types[risk_type]
            
            # Calculate mean and CI for each metric
            potential_risk = ""
            risk_type_acc = ""
            hazard = ""
            
            if "potential_risk_accuracy" in metrics and metrics["potential_risk_accuracy"]:
                values = [v for v in metrics["potential_risk_accuracy"] if v is not None]
                if values:
                    mean = np.mean(values)
                    _, _, margin = calculate_ci(values)
                    potential_risk = f"{mean:.4f} ± {margin:.4f}"
            
            if "risk_type_accuracy" in metrics and metrics["risk_type_accuracy"]:
                values = [v for v in metrics["risk_type_accuracy"] if v is not None]
                if values:
                    mean = np.mean(values)
                    _, _, margin = calculate_ci(values)
                    risk_type_acc = f"{mean:.4f} ± {margin:.4f}"
            
            if "hazard_accuracy" in metrics and metrics["hazard_accuracy"]:
                values = [v for v in metrics["hazard_accuracy"] if v is not None]
                if values:
                    mean = np.mean(values)
                    _, _, margin = calculate_ci(values)
                    hazard = f"{mean:.4f} ± {margin:.4f}"
            
            risk_type_csv_lines.append(f"{model_name},{risk_type},{potential_risk},{risk_type_acc},{hazard}")
    
    risk_type_csv_text = "\n".join(risk_type_csv_lines)
    with open(risk_type_csv_file, 'w', encoding='utf-8') as f:
        f.write(risk_type_csv_text)
    
    print(f"Risk-type-based CSV results saved to: {risk_type_csv_file}")
    
    # Generate and save individual run statistics
    print("\nGenerating individual run statistics...")
    for run_dir, models in run_statistics.items():
        for model_name, test_types in models.items():
            model_dir = run_dir / model_name
            if not model_dir.exists():
                continue
            
            output_lines = []
            output_lines.append("=" * 80)
            output_lines.append(f"Single Inference Statistics")
            output_lines.append(f"Run: {run_dir.name}")
            output_lines.append(f"Model: {model_name}")
            output_lines.append("=" * 80)
            output_lines.append("")
            
            # Overall results
            if "overall" in test_types:
                output_lines.append("Overall (All Types Combined):")
                output_lines.append("-" * 80)
                overall = test_types["overall"]
                
                for metric in ["overall_accuracy", "potential_risk_accuracy", 
                              "risk_type_accuracy", "hazard_accuracy"]:
                    if metric in overall:
                        value = overall[metric]
                        output_lines.append(
                            f"  {metric.replace('_', ' ').title()}: {value:.4f}"
                        )
                output_lines.append("")
            
            # Results by test type
            # Check if this is heldout_set or test_set
            is_heldout = "heldout_set" in str(model_dir)
            
            if is_heldout:
                # For heldout_set, show safe and unsafe
                test_type_order = ["safe", "unsafe"]
                type_label = "Dataset"
            else:
                # For test_set, show EMBGuard scenario types in a stable order.
                test_type_order = list(TEST_TYPE_CODES)
                type_label = "Scenario Type"
            
            for test_type in test_type_order:
                if test_type not in test_types:
                    continue
                
                display_type = test_type_label(test_type)
                output_lines.append(f"{type_label}: {display_type}")
                output_lines.append("-" * 80)
                type_data = test_types[test_type]
                
                for metric in ["overall_accuracy", "potential_risk_accuracy", 
                              "risk_type_accuracy", "hazard_accuracy",
                              "conditional_risk_type_accuracy", "conditional_hazard_accuracy"]:
                    if metric in type_data:
                        value = type_data[metric]
                        metric_name = metric.replace('_', ' ').title()
                        if metric.startswith("conditional"):
                            metric_name = metric_name.replace("Conditional ", "Conditional ").replace("Accuracy", "Accuracy (when potential_risk correct)")
                        output_lines.append(
                            f"  {metric_name}: {value:.4f}"
                        )
                output_lines.append("")
            
            # Save to model directory
            single_stats_file = model_dir / "single_inference_statistics.txt"
            with open(single_stats_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_lines))
            
            print(f"  Saved: {single_stats_file.relative_to(results_dir)}")
    
    return aggregated


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across multiple runs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/EMBGuardTest",
        help="Directory containing Run_1, Run_2, ... folders (default: results/EMBGuardTest)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output text file path (default: {results_dir}/aggregated_results.txt)"
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="Run_",
        help="Prefix for run directories (default: Run_)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = project_root / args.results_dir
    output_file = project_root / args.output_file if args.output_file else None
    
    try:
        aggregated = aggregate_results(
            results_dir=results_dir,
            output_file=output_file,
            run_prefix=args.run_prefix
        )
        
        print(f"\nAggregated results for {len(aggregated)} models")
        print("Summary:")
        for model_name, data in aggregated.items():
            if "overall" in data and "overall_accuracy" in data["overall"]:
                mean = data["overall"]["overall_accuracy"]["mean"]
                ci = data["overall"]["overall_accuracy"]
                print(f"  {model_name}: {mean:.4f} ± {ci['ci_margin']:.4f} (95% CI)")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
