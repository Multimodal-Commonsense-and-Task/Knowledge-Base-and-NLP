#!/usr/bin/env python3
"""
Visualize trend lines for potential_risk across EMBGuardTest scenario types:
Causal Risky, Selective Risky, Decoupled Benign, and Absent Benign.
Each model gets its own line showing how potential_risk changes across scenario types.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.test_types import TEST_TYPE_CODE_TO_LABEL, normalize_test_type_code


def parse_value(value_str: str) -> Optional[float]:
    """
    Parse value from string like "0.9277 ± 0.0145" or "56.8 (± 1.1)" or "0.9277"
    Returns the mean value as a decimal
    - If value is already in decimal format (0.9277), returns as is
    - If value is in percentage format (56.8), converts to decimal (0.568)
    Ignores CI part (everything after ± or parentheses)
    """
    if not value_str or value_str.strip() == "":
        return None
    
    # Remove whitespace
    value_str = str(value_str).strip()
    
    # Extract the first number (mean value) before any ±, (, or other non-numeric characters
    import re
    match = re.match(r'^([\d.]+)', value_str)
    if match:
        try:
            mean_val = float(match.group(1))
            # Check if value is likely a percentage (> 1) or already decimal (<= 1)
            # If > 1, assume it's percentage and convert to decimal
            # If <= 1, assume it's already decimal
            if mean_val > 1.0:
                # Likely percentage format, convert to decimal
                return mean_val / 100.0
            else:
                # Already in decimal format
                return mean_val
        except (ValueError, TypeError):
            return None
    
    return None


def load_scenario_type_data(csv_file: Path) -> pd.DataFrame:
    """
    Load scenario_type-based CSV file and parse potential_risk values
    
    Args:
        csv_file: Path to CSV file (e.g., aggregated_results_by_scenario_type.csv)
        
    Returns:
        DataFrame with parsed data
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)

    if 'scenario_type' not in df.columns and 'test_type' in df.columns:
        df['scenario_type'] = df['test_type']

    if 'model_name' not in df.columns or 'scenario_type' not in df.columns or 'potential_risk' not in df.columns:
        raise ValueError(f"CSV file must have columns: model_name, scenario_type, potential_risk. Found: {df.columns.tolist()}")

    df['scenario_type_code'] = df['scenario_type'].apply(lambda value: normalize_test_type_code(value, default=value))

    # Parse potential_risk values
    df['potential_risk_value'] = df['potential_risk'].apply(parse_value)
    
    # Filter out rows with None values
    df = df[df['potential_risk_value'].notna()]
    
    return df


def plot_type_trends(
    csv_file: Path,
    output_file: Path,
    models: Optional[List[str]] = None,
    model_colors: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot trend lines for potential_risk across scenario types
    
    Args:
        csv_file: Path to scenario_type-based CSV file
        output_file: Path to save the plot
        models: Optional list of model names to plot (if None, plots all models)
        title: Optional custom title
    """
    # Load data
    df = load_scenario_type_data(csv_file)
    
    # Define scenario_type order using canonical codes for stable plotting.
    scenario_type_order = ['HR', 'MHR', 'HNR', 'NHR']
    
    # Map scenario types to display labels
    scenario_type_labels = TEST_TYPE_CODE_TO_LABEL
    
    # ============================================
    # Model selection, colors, and markers (edit here)
    # ============================================
    # All available models with their colors and markers
    # Uncomment the models you want to plot
    ALL_MODELS = {
        'openai_gpt-4o': {'color': 'red', 'marker': 'o'},
        'openai_gpt-4o-mini': {'color': 'orange', 'marker': 'o'},
        'openai_gpt-5.1': {'color': '#359994', 'marker': '^'},  # Red triangle
        'gemini_gemini-2.5-flash': {'color': 'blue', 'marker': 'o'},
        'gemini_gemini-2.5-pro': {'color': '#27615f', 'marker': '^'},  # Blue triangle
        'vllm_EMBGuard_EMBGuard-2B': {'color': '#8961c8', 'marker': 'o'},  # Green circle
        'vllm_EMBGuard_EMBGuard-4B': {'color': '#006edf', 'marker': 'o'},  # Darker green circle
        'vllm_Qwen_Qwen3-VL-2B-Instruct': {'color': '#AC90D9', 'marker': 's'},  # Lighter green square (2B)
        'vllm_Qwen_Qwen3-VL-4B-Instruct': {'color': '#4D9AE9', 'marker': 's'},  # Light green square (4B)
        'vllm_OpenGVLab_InternVL3_5-1B-HF': {'color': 'brown', 'marker': 'o'},
        'vllm_OpenGVLab_InternVL3_5-2B-HF': {'color': 'saddlebrown', 'marker': 'o'},
        'openrouter_qwen_qwen3-vl-8b-instruct': {'color': 'pink', 'marker': 'o'},
        'openrouter_qwen_qwen3-vl-32b-instruct': {'color': 'deeppink', 'marker': 'o'},
        'openrouter_qwen_qwen3-vl-30b-a3b-instruct': {'color': 'hotpink', 'marker': 'o'},
        'openrouter_qwen_qwen3-vl-235b-a22b-instruct': {'color': 'magenta', 'marker': 'o'},
        'openrouter_google_gemma-3-4b-it': {'color': 'cyan', 'marker': 'o'},
        'openrouter_google_gemma-3-12b-it': {'color': 'teal', 'marker': 'o'},
        'openrouter_google_gemma-3-27b-it': {'color': 'darkcyan', 'marker': 'o'},
    }
    
    # Select models to plot (uncomment the ones you want)
    # You can also specify a number to plot first N models
    SELECTED_MODELS = [
        # 'openai_gpt-4o',
        # 'openai_gpt-4o-mini',
        'openai_gpt-5.1',
        # 'gemini_gemini-2.5-flash',
        'gemini_gemini-2.5-pro',
        'vllm_EMBGuard_EMBGuard-2B',
        'vllm_EMBGuard_EMBGuard-4B',
        'vllm_Qwen_Qwen3-VL-4B-Instruct',
        'vllm_Qwen_Qwen3-VL-2B-Instruct',
        # 'vllm_OpenGVLab_InternVL3_5-1B-HF',
        # 'vllm_OpenGVLab_InternVL3_5-2B-HF',
        # 'openrouter_qwen_qwen3-vl-8b-instruct',
        # 'openrouter_qwen_qwen3-vl-32b-instruct',
        # 'openrouter_qwen_qwen3-vl-30b-a3b-instruct',
        # 'openrouter_qwen_qwen3-vl-235b-a22b-instruct',
        # 'openrouter_google_gemma-3-4b-it',
        # 'openrouter_google_gemma-3-12b-it',
        # 'openrouter_google_gemma-3-27b-it',
    ]
    
    # Build model config dict from selected models
    model_configs = {}
    if not model_colors:
        for model in SELECTED_MODELS:
            if model in ALL_MODELS:
                model_configs[model] = ALL_MODELS[model]
    else:
        # If model_colors provided via command line, convert to config format
        for model, color in model_colors.items():
            model_configs[model] = {'color': color, 'marker': 'o'}  # Default marker
    
    # Filter by models if specified (command line takes precedence)
    if models:
        df = df[df['model_name'].isin(models)]
        # Update model_configs to only include specified models
        model_configs = {k: v for k, v in model_configs.items() if k in models}
    elif SELECTED_MODELS:
        # Use SELECTED_MODELS from code
        df = df[df['model_name'].isin(SELECTED_MODELS)]
        # Filter model_configs to only include selected models
        model_configs = {k: v for k, v in model_configs.items() if k in SELECTED_MODELS}
    
    # Get unique models
    unique_models = sorted(df['model_name'].unique())
    
    if len(unique_models) == 0:
        raise ValueError("No models found in the data")
    
    print(f"Found {len(unique_models)} models: {unique_models}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_style("whitegrid")
    
    # Add background colors for scenario types
    # Causal Risky and Selective Risky: red background (connected)
    # Decoupled Benign and Absent Benign: green background (connected)
    scenario_type_positions = {scenario_type: idx for idx, scenario_type in enumerate(scenario_type_order)}
    
    # Causal Risky and Selective Risky: connected red background
    if 'HR' in scenario_type_positions and 'MHR' in scenario_type_positions:
        hr_pos = scenario_type_positions['HR']
        mhr_pos = scenario_type_positions['MHR']
        # Connect Causal Risky and Selective Risky with one continuous background.
        ax.axvspan(hr_pos - 0.4, mhr_pos + 0.4, alpha=0.2, color='red', zorder=0)
    
    # Decoupled Benign and Absent Benign: connected green background
    if 'HNR' in scenario_type_positions and 'NHR' in scenario_type_positions:
        hnr_pos = scenario_type_positions['HNR']
        nhr_pos = scenario_type_positions['NHR']
        # Connect Decoupled Benign and Absent Benign with one continuous background.
        ax.axvspan(hnr_pos - 0.4, nhr_pos + 0.4, alpha=0.2, color='green', zorder=0)
    
    # Define colors and markers for models
    # Use model_configs if available, otherwise use default
    if model_configs:
        # Extract colors and markers from configs
        colors_dict = {model: config['color'] for model, config in model_configs.items()}
        markers_dict = {model: config['marker'] for model, config in model_configs.items()}
        # For models not in configs, use default
        default_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
        colors = [colors_dict.get(model, default_colors[idx]) for idx, model in enumerate(unique_models)]
        markers = [markers_dict.get(model, 'o') for model in unique_models]
    else:
        # Use default colormap and markers
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
        markers = ['o'] * len(unique_models)
    
    # Plot line for each model
    for idx, model_name in enumerate(unique_models):
        model_data = df[df['model_name'] == model_name]
        
        # Extract values in the correct order
        values = []
        for scenario_type in scenario_type_order:
            scenario_data = model_data[model_data['scenario_type_code'] == scenario_type]
            if len(scenario_data) > 0:
                value = scenario_data['potential_risk_value'].iloc[0]
                values.append(value)
            else:
                values.append(None)
        
        # Filter out None values for plotting
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        valid_scenario_types = [scenario_type_order[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]
        
        if len(valid_values) > 0:
            # Shorten model name for legend
            display_name = model_name
            if '_' in model_name:
                parts = model_name.split('_', 1)
                if len(parts) > 1:
                    display_name = parts[1]
            
            # Convert scenario_type names to positions for plotting
            valid_positions = [scenario_type_positions[t] for t in valid_scenario_types]
            
            # Get color and marker for this model
            if model_configs and model_name in model_configs:
                model_color = model_configs[model_name]['color']
                model_marker = model_configs[model_name]['marker']
            else:
                model_color = colors[idx]
                model_marker = markers[idx]
            
            # Determine line width: thicker for EMBGuard models
            if 'EMBGuard' in model_name:
                line_width = 6
            elif 'Qwen' in model_name:
                line_width = 2
            else:
                line_width = 3
            
            # Determine line style: dashed for Qwen models
            if 'Qwen' in model_name:
                line_style = '--'
            else:
                line_style = '-'
            
            ax.plot(valid_positions, valid_values, 
                    marker=model_marker, markersize=8, linewidth=line_width, 
                    linestyle=line_style, color=model_color, label=display_name, alpha=0.8, zorder=2)
    
    # Labels and title
    # ax.set_xlabel('Scenario Type', fontsize=16, fontweight='bold')
    ax.set_ylabel('Potential Risk', fontsize=14, fontweight='bold')
    
    # if title:
    #     ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    # else:
    #     ax.set_title('Potential Risk Accuracy by Scenario Type', fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(scenario_type_order)))
    display_labels = [scenario_type_labels.get(tt, tt) for tt in scenario_type_order]
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y', zorder=1)
    
    # Set y-axis limits (0 to 1 for accuracy)
    ax.set_ylim(0, 1)
    
    # Add legend
    # ax.legend(loc='best', fontsize=10, ncol=2)
    
    # Reduce padding/margins for tighter layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
    
    # Save plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trend lines for potential_risk across scenario types"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to scenario_type-based CSV file (e.g., results/EMBGuardTest/aggregated_results_by_scenario_type.csv)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output plot"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of model names to plot (if not specified, plots all models)"
    )
    parser.add_argument(
        "--model-colors",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of model:color pairs (e.g., 'model1:red' 'model2:blue'). Colors can be named colors or hex codes."
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the plot"
    )
    
    args = parser.parse_args()
    
    # Parse model colors if provided
    model_colors = None
    if args.model_colors:
        model_colors = {}
        for pair in args.model_colors:
            if ':' not in pair:
                raise ValueError(f"Invalid model:color pair: {pair}. Expected format: 'model_name:color'")
            model_name, color = pair.split(':', 1)
            model_colors[model_name] = color.strip()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    
    csv_file = project_root / args.csv_file
    output_file = project_root / args.output_file
    
    # Plot trends
    plot_type_trends(
        csv_file=csv_file,
        output_file=output_file,
        models=args.models,
        model_colors=model_colors,
        title=args.title,
    )


if __name__ == "__main__":
    main()
