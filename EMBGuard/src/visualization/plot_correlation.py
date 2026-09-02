#!/usr/bin/env python3
"""
Visualize correlation results for multiple metrics in a single figure
Takes CSV files for potential_risk, conditional_risk_type, and conditional_hazard
and plots them together with different colors
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_correlation_data(csv_file: Path) -> Tuple[pd.DataFrame, float, float]:
    """
    Load correlation data from CSV file
    
    Args:
        csv_file: Path to CSV file with columns: model, score1, score2
        
    Returns:
        Tuple of (dataframe, correlation, p_value)
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    if 'model' not in df.columns or 'score1' not in df.columns or 'score2' not in df.columns:
        raise ValueError(f"CSV file must have columns: model, score1, score2. Found: {df.columns.tolist()}")
    
    # Calculate correlation
    corr, p_value = pearsonr(df['score1'], df['score2'])
    
    return df, corr, p_value


def plot_correlations(
    potential_risk_csv: Path,
    conditional_risk_type_csv: Path,
    conditional_hazard_csv: Path,
    output_file: Path,
    correlation_method: str = "pearson",
    title: Optional[str] = None,
) -> None:
    """
    Plot correlations for three metrics in a single figure
    
    Args:
        potential_risk_csv: Path to potential_risk correlation CSV
        conditional_risk_type_csv: Path to conditional_risk_type correlation CSV
        conditional_hazard_csv: Path to conditional_hazard correlation CSV
        output_file: Path to save the plot
        correlation_method: Correlation method name (for display)
        title: Optional custom title
    """
    # Load data
    try:
        df_pr, corr_pr, p_pr = load_correlation_data(potential_risk_csv)
        df_crt, corr_crt, p_crt = load_correlation_data(conditional_risk_type_csv)
        df_ch, corr_ch, p_ch = load_correlation_data(conditional_hazard_csv)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Create figure with tighter margins
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    # Set tight margins from the start
    plt.rcParams['figure.autolayout'] = False
    
    # Define colors for each metric
    colors = {
        'potential_risk': '#DE5151',  # Blue
        'conditional_risk_type': '#FF853E',  # Orange
        'conditional_hazard': '#F9DD01',  # Green
    }
    
    # Plot scatter points and regression lines for each metric
    metrics_data = [
        (df_pr, corr_pr, p_pr, 'Potential Risk', colors['potential_risk']),
        (df_crt, corr_crt, p_crt, 'Risk Type', colors['conditional_risk_type']),
        (df_ch, corr_ch, p_ch, 'Hazard', colors['conditional_hazard']),
    ]
    
    for df, corr, p_value, label, color in metrics_data:
        plt.scatter(
            df['score2'], df['score1'],
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5,
            color=color,
            # label=label
        )
        
        z = np.polyfit(df['score2'], df['score1'], 1)
        p = np.poly1d(z)
        
        # Format p-value for display
        if p_value < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = f'p = {p_value:.3f}'
        
        plt.plot(df['score2'], p(df['score2']), "-", alpha=0.9, linewidth=3, 
                color=color, label=f'{label} (r={corr:.3f}, {p_str})')
    
    plt.xlabel('Heldout Set', fontsize=20, fontweight='bold')
    plt.ylabel('EMBGuardTest', fontsize=20, fontweight='bold')
    
    # if title:
    #     plt.title(title, fontsize=14, fontweight='bold', pad=20)
    # else:
    #     plt.title(f'Correlation Analysis ({correlation_method.title()})', 
    #              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=15)
    
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.7)
    
    ax = plt.gca()
    ax.set_aspect(0.65, adjustable='box')
    
    # Reduce padding/margins for tighter layout
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Plot saved to: {output_file}\n")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize correlation results for multiple metrics in a single figure"
    )
    parser.add_argument(
        "--potential-risk-csv",
        type=str,
        required=True,
        help="Path to potential_risk correlation CSV file"
    )
    parser.add_argument(
        "--conditional-risk-type-csv",
        type=str,
        required=True,
        help="Path to conditional_risk_type correlation CSV file"
    )
    parser.add_argument(
        "--conditional-hazard-csv",
        type=str,
        required=True,
        help="Path to conditional_hazard correlation CSV file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output plot"
    )
    parser.add_argument(
        "--correlation-method",
        type=str,
        default="pearson",
        help="Correlation method name (for display, default: pearson)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the plot"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    
    potential_risk_csv = project_root / args.potential_risk_csv
    conditional_risk_type_csv = project_root / args.conditional_risk_type_csv
    conditional_hazard_csv = project_root / args.conditional_hazard_csv
    output_file = project_root / args.output_file
    
    # Plot correlations
    plot_correlations(
        potential_risk_csv=potential_risk_csv,
        conditional_risk_type_csv=conditional_risk_type_csv,
        conditional_hazard_csv=conditional_hazard_csv,
        output_file=output_file,
        correlation_method=args.correlation_method,
        title=args.title,
    )


if __name__ == "__main__":
    main()
