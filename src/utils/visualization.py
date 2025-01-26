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
    severities: List[List[int]],
    coverages: List[List[float]],
    set_sizes: List[List[float]],
    abstention_rates: List[List[float]],
    labels: List[str],
    save_dir: str = 'plots/metrics'
) -> None:
    """Plot key metrics against severity levels for multiple corruptions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 5))
    
    # Colors for each corruption type
    colors = {
        'occlusion': 'b',
        'rain': 'r',
        'fog': 'g',
        'blur': 'm',
        'snow': 'c'
    }
    
    # Coverage plot
    plt.subplot(1, 3, 1)
    for i, (sev, cov, label) in enumerate(zip(severities, coverages, labels)):
        plt.plot(sev, cov, f'o-', color=colors[label.lower()], label=label.capitalize(), linewidth=2)
    plt.axhline(y=0.9, color='k', linestyle='--', label='Target (90%)')
    plt.xlabel('Severity')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Set size plot
    plt.subplot(1, 3, 2)
    for i, (sev, size, label) in enumerate(zip(severities, set_sizes, labels)):
        plt.plot(sev, size, f'o-', color=colors[label.lower()], label=label.capitalize(), linewidth=2)
    plt.xlabel('Severity')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Abstention rate plot
    plt.subplot(1, 3, 3)
    for i, (sev, rate, label) in enumerate(zip(severities, abstention_rates, labels)):
        plt.plot(sev, rate, f'o-', color=colors[label.lower()], label=label.capitalize(), linewidth=2)
    plt.xlabel('Severity')
    plt.ylabel('Abstention Rate')
    plt.title('Abstention Rate vs Severity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    save_dir: str = 'plots/roc'
) -> None:
    """Plot ROC curves for each severity level and corruption type."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 8))
    
    colors = {
        'occlusion': plt.cm.Blues(np.linspace(0.3, 1, 5)),
        'rain': plt.cm.Reds(np.linspace(0.3, 1, 5)),
        'fog': plt.cm.Greens(np.linspace(0.3, 1, 5)),
        'blur': plt.cm.Purples(np.linspace(0.3, 1, 5)),
        'snow': plt.cm.YlGn(np.linspace(0.3, 1, 5))
    }
    
    for corruption_name, results in results_by_corruption.items():
        for severity, color in zip(sorted(results.keys()), colors[corruption_name]):
            res = results[severity]['abstention_results']
            thresholds = sorted(res.keys())
            fpr_rates = [res[t]['fpr'] for t in thresholds]
            tpr_rates = [res[t]['tpr'] for t in thresholds]
            auc = results[severity]['auc']
            
            label = f'{corruption_name.capitalize()} Severity {severity} (AUC = {auc:.3f})'
            plt.plot(fpr_rates, tpr_rates, color=color, linewidth=2, label=label)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Severity Level and Corruption Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_set_size_distribution(
    set_sizes_by_corruption: Dict[str, Dict[int, np.ndarray]],
    save_dir: str = 'plots/set_sizes'
) -> None:
    """Plot set size distributions for each severity level and corruption type."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    
    colors = {
        'occlusion': plt.cm.Blues(np.linspace(0.3, 1, 5)),
        'rain': plt.cm.Reds(np.linspace(0.3, 1, 5)),
        'fog': plt.cm.Greens(np.linspace(0.3, 1, 5)),
        'blur': plt.cm.Purples(np.linspace(0.3, 1, 5)),
        'snow': plt.cm.YlGn(np.linspace(0.3, 1, 5))
    }
    
    for corruption_name, set_sizes_by_severity in set_sizes_by_corruption.items():
        for severity, color in zip(sorted(set_sizes_by_severity.keys()), colors[corruption_name]):
            set_sizes = set_sizes_by_severity[severity]
            unique_sizes, counts = np.unique(set_sizes, return_counts=True)
            percentages = (counts / len(set_sizes)) * 100
            
            plt.plot(unique_sizes, percentages, 'o-', color=color, linewidth=2,
                    label=f'{corruption_name.capitalize()} Severity {severity}')
    
    plt.xlabel('Set Size')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Set Size Distribution by Severity and Corruption Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    
    # Colors for each corruption type
    colors = {
        'occlusion': 'b',
        'rain': 'r',
        'fog': 'g',
        'blur': 'm',
        'snow': 'c'
    }
    
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Nonconformity-based Abstention Analysis (Severity {severity})', fontsize=14)
    
    # Plot for each corruption type
    for corruption_name, corruption_results in results.items():
        thresholds = sorted(list(corruption_results.keys()))
        tpr_rates = [corruption_results[t]['tpr'] for t in thresholds]
        fpr_rates = [corruption_results[t]['fpr'] for t in thresholds]
        abstention_rates = [corruption_results[t]['abstention_rate'] for t in thresholds]
        
        # ROC curve
        axs[0, 0].plot(fpr_rates, tpr_rates, f'{colors[corruption_name]}-', 
                      label=corruption_name.capitalize(), linewidth=2)
        
        # Abstention rate
        axs[0, 1].plot(thresholds, abstention_rates, f'{colors[corruption_name]}-',
                      label=corruption_name.capitalize(), linewidth=2)
        
        # TPR and FPR vs threshold
        axs[1, 0].plot(thresholds, tpr_rates, f'{colors[corruption_name]}-',
                      label=f'TPR ({corruption_name.capitalize()})', linewidth=2)
        axs[1, 0].plot(thresholds, fpr_rates, f'{colors[corruption_name]}--',
                      label=f'FPR ({corruption_name.capitalize()})', linewidth=2)
        
        # TPR-FPR difference
        diff_rates = [tpr - fpr for tpr, fpr in zip(tpr_rates, fpr_rates)]
        axs[1, 1].plot(thresholds, diff_rates, f'{colors[corruption_name]}-',
                      label=corruption_name.capitalize(), linewidth=2)
    
    # ROC curve settings
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axs[0, 0].set_xlabel('False Positive Rate (FPR)')
    axs[0, 0].set_ylabel('True Positive Rate (TPR)')
    axs[0, 0].set_title('ROC Curve')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Abstention rate settings
    axs[0, 1].set_xlabel('Nonconformity Threshold')
    axs[0, 1].set_ylabel('Abstention Rate')
    axs[0, 1].set_title('Abstention Rate vs Threshold')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # TPR and FPR vs threshold settings
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylabel('Rate')
    axs[1, 0].set_title('TPR and FPR vs Threshold')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # TPR-FPR difference settings
    axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('TPR - FPR')
    axs[1, 1].set_title('TPR-FPR Difference')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f'nonconformity_abstention_analysis_severity_{severity}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_paper_plots(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    uncertainty_metrics_by_corruption: Dict[str, Dict[int, Dict]],
    set_sizes_by_corruption: Dict[str, Dict[int, np.ndarray]],
    save_dir: str = 'plots/result_paper'
) -> None:
    """Create publication-quality plots for IJCNN paper."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set general plotting style for publication
    plt.style.use('seaborn-paper')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
    })
    
    # Color scheme for different corruptions
    colors = {
        'occlusion': '#2ecc71',    # Green
        'rain': '#3498db',         # Blue
        'fog': '#e74c3c',          # Red
        'blur': '#9b59b6',         # Purple
        'snow': '#f1c40f'          # Yellow
    }
    
    # 1. Unified Abstention Analysis
    create_unified_abstention_plot(results_by_corruption, colors, save_dir)
    
    # 2. Severity Impact Summary
    create_severity_impact_plot(results_by_corruption, colors, save_dir)
    
    # 3. Confidence and Set Size Distributions
    create_confidence_setsize_plot(
        uncertainty_metrics_by_corruption,
        set_sizes_by_corruption,
        colors,
        save_dir
    )
    
    # 4. Key Metrics and Threshold Analysis
    create_threshold_analysis_plot(results_by_corruption, colors, save_dir)

def create_unified_abstention_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create unified abstention analysis plot for LiDAR data."""
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1.2])
    
    # ROC Curves
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    severity = 3  # Focus on medium severity for comparison
    
    for corruption_name, results in results_by_corruption.items():
        res = results[severity]['abstention_results']
        thresholds = sorted(res.keys())
        fpr_rates = [res[t]['fpr'] for t in thresholds]
        tpr_rates = [res[t]['tpr'] for t in thresholds]
        abstention_rates = [res[t]['abstention_rate'] for t in thresholds]
        
        # ROC curve
        ax1.plot(fpr_rates, tpr_rates, color=colors[corruption_name], 
                label=f'{corruption_name.capitalize()}', linewidth=2)
        
        # TPR/FPR vs Threshold
        ax2.plot(thresholds, tpr_rates, color=colors[corruption_name], 
                label=f'{corruption_name.capitalize()} (TPR)', linewidth=2)
        ax2.plot(thresholds, fpr_rates, color=colors[corruption_name], 
                linestyle='--', alpha=0.5, label=f'{corruption_name.capitalize()} (FPR)')
        
        # Abstention Rate
        ax3.plot(thresholds, abstention_rates, color=colors[corruption_name], 
                label=f'{corruption_name.capitalize()}', linewidth=2)
    
    # Style ROC plot
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('(a) ROC Curves')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Style TPR/FPR plot
    ax2.set_xlabel('Nonconformity Threshold')
    ax2.set_ylabel('Rate')
    ax2.set_title('(b) TPR/FPR vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1))
    
    # Style Abstention plot
    ax3.set_xlabel('Nonconformity Threshold')
    ax3.set_ylabel('Abstention Rate')
    ax3.set_title('(c) Abstention Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'lidar_unified_abstention_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_severity_impact_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create severity impact summary plot for LiDAR data."""
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    severities = sorted(list(next(iter(results_by_corruption.values())).keys()))
    
    for corruption_name, results in results_by_corruption.items():
        coverages = [results[s]['coverage'] for s in severities]
        set_sizes = [results[s]['avg_set_size'] for s in severities]
        abstention_rates = [np.mean([r['abstention_rate'] for r in results[s]['abstention_results'].values()]) 
                          for s in severities]
        
        ax1.plot(severities, coverages, 'o-', color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
        ax2.plot(severities, set_sizes, 'o-', color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
        ax3.plot(severities, abstention_rates, 'o-', color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
    
    # Style coverage plot
    ax1.axhline(y=0.9, color='k', linestyle='--', label='Target (90%)')
    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Coverage')
    ax1.set_title('(a) Coverage vs Severity')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Style set size plot
    ax2.set_xlabel('Severity Level')
    ax2.set_ylabel('Average Set Size')
    ax2.set_title('(b) Set Size vs Severity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Style abstention plot
    ax3.set_xlabel('Severity Level')
    ax3.set_ylabel('Average Abstention Rate')
    ax3.set_title('(c) Abstention Rate vs Severity')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'lidar_severity_impact_summary.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_confidence_setsize_plot(
    uncertainty_metrics: Dict[str, Dict[int, Dict]],
    set_sizes: Dict[str, Dict[int, np.ndarray]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create confidence and set size distribution plot for LiDAR data."""
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    severities = sorted(list(next(iter(uncertainty_metrics.values())).keys()))
    
    for corruption_name in uncertainty_metrics.keys():
        # Extract metrics
        entropies = [np.mean(uncertainty_metrics[corruption_name][s]['normalized_entropy']) 
                    for s in severities]
        confidences = [np.mean(uncertainty_metrics[corruption_name][s]['confidence']) 
                      for s in severities]
        
        # Plot entropy and confidence
        ax1.plot(severities, entropies, 'o-', color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
        ax2.plot(severities, confidences, 'o-', color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
        
        # Plot set size distributions for severity 3
        set_size_dist = set_sizes[corruption_name][3]
        unique_sizes, counts = np.unique(set_size_dist, return_counts=True)
        percentages = (counts / len(set_size_dist)) * 100
        ax3.plot(unique_sizes, percentages, 'o-', color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
    
    # Style entropy plot
    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Normalized Entropy')
    ax1.set_title('(a) Predictive Uncertainty')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Style confidence plot
    ax2.set_xlabel('Severity Level')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('(b) Model Confidence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Style set size distribution plot
    ax3.set_xlabel('Set Size')
    ax3.set_ylabel('Percentage of Samples (%)')
    ax3.set_title('(c) Set Size Distribution (Severity 3)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'lidar_confidence_setsize_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_threshold_analysis_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create threshold analysis plot for LiDAR data."""
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    severity = 3  # Focus on medium severity
    
    for corruption_name, results in results_by_corruption.items():
        res = results[severity]['abstention_results']
        thresholds = sorted(res.keys())
        
        # Calculate metrics
        tpr_minus_fpr = [res[t]['tpr'] - res[t]['fpr'] for t in thresholds]
        coverage_vs_threshold = [1 - res[t]['abstention_rate'] for t in thresholds]
        efficiency = [(res[t]['tpr'] * (1 - res[t]['fpr'])) for t in thresholds]
        
        # Plot metrics
        ax1.plot(thresholds, tpr_minus_fpr, color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
        ax2.plot(thresholds, coverage_vs_threshold, color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
        ax3.plot(thresholds, efficiency, color=colors[corruption_name], 
                label=corruption_name.capitalize(), linewidth=2)
    
    # Style TPR-FPR plot
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Nonconformity Threshold')
    ax1.set_ylabel('TPR - FPR')
    ax1.set_title('(a) Discrimination Ability')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Style coverage plot
    ax2.set_xlabel('Nonconformity Threshold')
    ax2.set_ylabel('Effective Coverage')
    ax2.set_title('(b) Coverage vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Style efficiency plot
    ax3.set_xlabel('Nonconformity Threshold')
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('(c) Abstention Efficiency')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'lidar_threshold_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
