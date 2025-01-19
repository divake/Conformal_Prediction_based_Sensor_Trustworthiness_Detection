import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

def create_plot_dirs(base_dir: str = 'plots') -> Dict[str, Path]:
    """Create all needed plot directories."""
    base_dir = Path(base_dir)
    subdirs = ['metrics', 'roc', 'set_sizes', 'abstention']
    
    dirs = {}
    for subdir in subdirs:
        path = base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        dirs[subdir] = path
    
    return dirs

def plot_metrics_vs_severity(
    severities: List[int],
    coverages: List[float],
    set_sizes: List[float],
    abstention_rates: List[float],
    save_dir: str = 'plots/metrics'
) -> None:
    """Plot key metrics against severity levels for occlusion corruption."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 5))
    
    # Coverage plot
    plt.subplot(1, 3, 1)
    plt.plot(severities, coverages, 'o-', label='Occlusion', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    plt.xlabel('Severity')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Set size plot
    plt.subplot(1, 3, 2)
    plt.plot(severities, set_sizes, 'o-', label='Occlusion', linewidth=2)
    plt.xlabel('Severity')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Abstention rate plot
    plt.subplot(1, 3, 3)
    plt.plot(severities, abstention_rates, 'o-', label='Occlusion', linewidth=2)
    plt.xlabel('Severity')
    plt.ylabel('Abstention Rate')
    plt.title('Abstention Rate vs Severity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(
    results: Dict[int, Dict],  # {severity: results_dict}
    save_dir: str = 'plots/roc'
) -> None:
    """Plot ROC curves for each severity level."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 8))
    
    # Use different shades of blue for different severities
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(results)))
    
    for severity, color in zip(sorted(results.keys()), colors):
        res = results[severity]['abstention_results']
        thresholds = sorted(res.keys())
        fpr_rates = [res[t]['fpr'] for t in thresholds]
        tpr_rates = [res[t]['tpr'] for t in thresholds]
        auc = results[severity]['auc']
        
        plt.plot(fpr_rates, tpr_rates, color=color, linewidth=2,
                label=f'Severity {severity} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Severity Level (Occlusion)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_set_size_distribution(
    set_sizes_by_severity: Dict[int, np.ndarray],
    save_dir: str = 'plots/set_sizes'
) -> None:
    """Plot set size distributions for each severity level."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(set_sizes_by_severity)))
    
    for severity, color in zip(sorted(set_sizes_by_severity.keys()), colors):
        set_sizes = set_sizes_by_severity[severity]
        unique_sizes, counts = np.unique(set_sizes, return_counts=True)
        percentages = (counts / len(set_sizes)) * 100
        
        plt.plot(unique_sizes, percentages, 'o-', color=color, linewidth=2,
                label=f'Severity {severity}')
    
    plt.xlabel('Set Size')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Set Size Distribution by Severity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'set_size_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_nonconformity_analysis(
    results: Dict,
    severity: int,
    save_dir: str = 'plots/nonconformity_abstention'
) -> None:
    """Plot analysis results with linear scale thresholds"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = sorted(list(results.keys()))
    tpr_rates = [results[t]['tpr'] for t in thresholds]
    fpr_rates = [results[t]['fpr'] for t in thresholds]
    abstention_rates = [results[t]['abstention_rate'] for t in thresholds]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Nonconformity-based Abstention Analysis (Severity {severity})', fontsize=14)
    
    # ROC curve
    axs[0, 0].plot(fpr_rates, tpr_rates, 'o-')
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axs[0, 0].set_xlabel('False Positive Rate (FPR)')
    axs[0, 0].set_ylabel('True Positive Rate (TPR)')
    axs[0, 0].set_title('ROC Curve')
    axs[0, 0].grid(True)
    
    # Abstention rate
    axs[0, 1].plot(thresholds, abstention_rates, 'o-')
    axs[0, 1].set_xlabel('Nonconformity Threshold')
    axs[0, 1].set_ylabel('Abstention Rate')
    axs[0, 1].set_title('Abstention Rate vs Threshold')
    axs[0, 1].grid(True)
    
    # TPR and FPR vs threshold
    axs[1, 0].plot(thresholds, tpr_rates, 'o-', label='TPR')
    axs[1, 0].plot(thresholds, fpr_rates, 'o-', label='FPR')
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylabel('Rate')
    axs[1, 0].set_title('TPR and FPR vs Threshold')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # TPR-FPR difference
    diff_rates = [tpr - fpr for tpr, fpr in zip(tpr_rates, fpr_rates)]
    axs[1, 1].plot(thresholds, diff_rates, 'o-')
    axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('TPR - FPR')
    axs[1, 1].set_title('TPR-FPR Difference')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'nonconformity_abstention_analysis_severity_{severity}.png')
    plt.close()
