import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def create_plot_dirs(base_dir: str = 'plots_imagenet') -> Dict[str, Path]:
    """Create all needed plot directories."""
    base_dir = Path(base_dir)
    subdirs = ['metrics', 'roc', 'set_sizes', 'abstention', 'paper']
    
    dirs = {}
    for subdir in subdirs:
        path = base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        dirs[subdir] = path
    
    return dirs

def plot_metrics_vs_severity(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    save_dir: str = 'plots_imagenet/metrics'
) -> None:
    """Plot key metrics against severity levels."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 5))
    
    # Extract data
    colors = {'base': '#2ecc71', 'rain': '#3498db'}
    
    # Coverage plot
    plt.subplot(1, 3, 1)
    for corruption_name, results in results_by_corruption.items():
        severities = sorted(results.keys())
        coverages = [results[s]['coverage'] for s in severities]
        if corruption_name == 'base':
            plt.axhline(y=coverages[0], color=colors[corruption_name], linestyle='--',
                       label=f'{corruption_name.capitalize()} Coverage')
        else:
            plt.plot(severities, coverages, 'o-', 
                    label=f'{corruption_name.capitalize()} Coverage', 
                    color=colors[corruption_name], linewidth=2)
    
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    plt.xlabel('Severity')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Set size plot
    plt.subplot(1, 3, 2)
    for corruption_name, results in results_by_corruption.items():
        severities = sorted(results.keys())
        set_sizes = [results[s]['avg_set_size'] for s in severities]
        if corruption_name == 'base':
            plt.axhline(y=set_sizes[0], color=colors[corruption_name], linestyle='--',
                       label=f'{corruption_name.capitalize()}')
        else:
            plt.plot(severities, set_sizes, 'o-', 
                    label=corruption_name.capitalize(), 
                    color=colors[corruption_name], linewidth=2)
    
    plt.xlabel('Severity')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Abstention rate plot
    plt.subplot(1, 3, 3)
    for corruption_name, results in results_by_corruption.items():
        severities = sorted(results.keys())
        abstention_rates = [np.mean([r['abstention_rate'] for r in results[s]['abstention_results'].values()]) 
                          for s in severities]
        if corruption_name == 'base':
            plt.axhline(y=abstention_rates[0], color=colors[corruption_name], linestyle='--',
                       label=f'{corruption_name.capitalize()}')
        else:
            plt.plot(severities, abstention_rates, 'o-', 
                    label=corruption_name.capitalize(), 
                    color=colors[corruption_name], linewidth=2)
    
    plt.xlabel('Severity')
    plt.ylabel('Abstention Rate')
    plt.title('Abstention Rate vs Severity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(
    results: Dict[str, Dict[int, Dict]],
    save_dir: str = 'plots_imagenet/roc'
) -> None:
    """Plot ROC curves for each severity level."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 8))
    
    # Use different color maps for each corruption
    colors = {'base': '#2ecc71', 'rain': '#3498db'}
    
    # Plot curves for each corruption type
    for corruption_name, corruption_results in results.items():
        if corruption_name == 'base':
            continue
        for severity in sorted(corruption_results.keys()):
            res = corruption_results[severity]['abstention_results']
            thresholds = sorted(res.keys())
            fpr_rates = [res[t]['fpr'] for t in thresholds]
            tpr_rates = [res[t]['tpr'] for t in thresholds]
            auc = corruption_results[severity]['auc']
            
            plt.plot(fpr_rates, tpr_rates, linewidth=2,
                    label=f'{corruption_name.capitalize()} Severity {severity} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Severity Level')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_set_size_distribution(
    set_sizes_dict: Dict[str, Dict[int, np.ndarray]],
    save_dir: str = 'plots_imagenet/set_sizes'
) -> None:
    """Plot set size distributions for each severity level."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    
    colors = {'base': '#2ecc71', 'rain': '#3498db'}
    
    # Plot distributions
    for corruption_name, severity_dict in set_sizes_dict.items():
        if corruption_name == 'base':
            continue
        for severity in sorted(severity_dict.keys()):
            set_sizes = severity_dict[severity]
            unique_sizes, counts = np.unique(set_sizes, return_counts=True)
            percentages = (counts / len(set_sizes)) * 100
            
            plt.plot(unique_sizes, percentages, 'o-', 
                    color=colors[corruption_name], linewidth=2,
                    label=f'{corruption_name.capitalize()} Severity {severity}')
    
    plt.xlabel('Prediction Set Size')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Set Size Distribution by Severity Level')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'set_size_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_abstention_analysis(
    thresholds: np.ndarray,
    results_base: Dict,
    save_dir: str = 'plots_imagenet/abstention'
) -> None:
    """Plot abstention analysis metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Abstention Analysis', fontsize=14)
    
    # Extract metrics
    tpr = [results_base[t]['tpr'] for t in thresholds]
    fpr = [results_base[t]['fpr'] for t in thresholds]
    abs_rate = [results_base[t]['abstention_rate'] for t in thresholds]
    
    # ROC curve
    axs[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label='Base')
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axs[0, 0].set_xlabel('False Positive Rate')
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[0, 0].set_title('ROC Curve')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Abstention rate vs threshold
    axs[0, 1].plot(thresholds, abs_rate, 'b-', linewidth=2, label='Base')
    axs[0, 1].set_xlabel('Threshold')
    axs[0, 1].set_ylabel('Abstention Rate')
    axs[0, 1].set_title('Abstention Rate vs Threshold')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # TPR/FPR vs threshold
    axs[1, 0].plot(thresholds, tpr, 'b-', linewidth=2, label='TPR')
    axs[1, 0].plot(thresholds, fpr, 'b--', linewidth=2, label='FPR')
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylabel('Rate')
    axs[1, 0].set_title('TPR and FPR vs Threshold')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # TPR-FPR difference
    diff = np.array(tpr) - np.array(fpr)
    axs[1, 1].plot(thresholds, diff, 'b-', linewidth=2, label='TPR-FPR')
    axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('TPR - FPR')
    axs[1, 1].set_title('TPR-FPR Difference')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'abstention_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distributions(
    uncertainty_metrics: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: str = 'plots_imagenet/metrics'
) -> None:
    """Plot confidence and entropy distributions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 10))
    
    # Plot confidence distributions
    plt.subplot(2, 2, 1)
    for corruption_name, metrics in uncertainty_metrics.items():
        for severity, metric_dict in metrics.items():
            confidence = metric_dict['confidence']
            plt.hist(confidence, bins=50, alpha=0.5, 
                    label=f'{corruption_name.capitalize()} (Severity {severity})',
                    density=True, color=colors[corruption_name])
    
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # Plot entropy distributions
    plt.subplot(2, 2, 2)
    for corruption_name, metrics in uncertainty_metrics.items():
        for severity, metric_dict in metrics.items():
            entropy = metric_dict['entropy']
            plt.hist(entropy, bins=50, alpha=0.5,
                    label=f'{corruption_name.capitalize()} (Severity {severity})',
                    density=True, color=colors[corruption_name])
    
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Entropy Distribution')
    plt.legend()
    
    # Plot margin distributions
    plt.subplot(2, 2, 3)
    for corruption_name, metrics in uncertainty_metrics.items():
        for severity, metric_dict in metrics.items():
            margin = metric_dict['margin']
            plt.hist(margin, bins=50, alpha=0.5,
                    label=f'{corruption_name.capitalize()} (Severity {severity})',
                    density=True, color=colors[corruption_name])
    
    plt.xlabel('Margin')
    plt.ylabel('Density')
    plt.title('Margin Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_severity_impact(
    softmax_scores_by_corruption: Dict[str, Dict[int, np.ndarray]],
    conformal_qhat: float,
    k_reg: int,
    lam_reg: float,
    save_dir: str = 'plots_imagenet/metrics'
) -> None:
    """Analyze how severity affects conformal scores."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    
    colors = {'base': '#2ecc71', 'rain': '#3498db'}
    
    for corruption_name, severity_dict in softmax_scores_by_corruption.items():
        for severity, scores in severity_dict.items():
            reg_vec = np.array([0]*k_reg + [lam_reg]*(scores.shape[1]-k_reg))[None,:]
            sort_idx = np.argsort(scores, axis=1)[:,::-1]
            sorted_probs = np.take_along_axis(scores, sort_idx, axis=1)
            sorted_reg = sorted_probs + reg_vec
            cumsum_reg = sorted_reg.cumsum(axis=1)
            
            mean_scores = np.mean(cumsum_reg, axis=0)
            std_scores = np.std(cumsum_reg, axis=0)
            
            plt.plot(range(1, len(mean_scores) + 1), mean_scores,
                    label=f'{corruption_name.capitalize()} (Severity {severity})',
                    color=colors[corruption_name])
            plt.fill_between(range(1, len(mean_scores) + 1),
                           mean_scores - std_scores,
                           mean_scores + std_scores,
                           alpha=0.2, color=colors[corruption_name])
    
    plt.axhline(y=conformal_qhat, color='r', linestyle='--',
                label='Conformal Threshold')
    plt.xlabel('Number of Classes')
    plt.ylabel('Cumulative Score')
    plt.title('Average Conformal Scores by Severity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'severity_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_paper_plots(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    uncertainty_metrics_by_corruption: Dict[str, Dict[int, Dict]],
    set_sizes_by_corruption: Dict[str, Dict[int, np.ndarray]],
    save_dir: str = 'plots_imagenet/paper'
) -> None:
    """Create publication-quality plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set publication style
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
    
    colors = {'base': '#2ecc71', 'rain': '#3498db'}
    
    # Create unified plots
    create_unified_abstention_plot(results_by_corruption, colors, save_dir)
    create_severity_impact_plot(results_by_corruption, colors, save_dir)
    create_confidence_setsize_plot(uncertainty_metrics_by_corruption, 
                                 set_sizes_by_corruption, colors, save_dir)
    create_threshold_analysis_plot(results_by_corruption, colors, save_dir)

def create_unified_abstention_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create unified abstention analysis plot."""
    plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1.2])
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    severity = 3  # Medium severity
    
    for corruption_name, results in results_by_corruption.items():
        if corruption_name == 'base':
            continue
        res = results[severity]['abstention_results']
        thresholds = sorted(res.keys())
        fpr_rates = [res[t]['fpr'] for t in thresholds]
        tpr_rates = [res[t]['tpr'] for t in thresholds]
        abstention_rates = [res[t]['abstention_rate'] for t in thresholds]
        
        # ROC curve
        ax1.plot(fpr_rates, tpr_rates, color=colors[corruption_name],
                label=corruption_name.capitalize(), linewidth=2)
        
        # TPR/FPR vs Threshold
        ax2.plot(thresholds, tpr_rates, color=colors[corruption_name],
                label=f'{corruption_name.capitalize()} (TPR)', linewidth=2)
        ax2.plot(thresholds, fpr_rates, color=colors[corruption_name],
                linestyle='--', alpha=0.5, label=f'{corruption_name.capitalize()} (FPR)')
        
        # Abstention Rate
        ax3.plot(thresholds, abstention_rates, color=colors[corruption_name],
                label=corruption_name.capitalize(), linewidth=2)
    
    # Style plots
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('(a) ROC Curves')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Nonconformity Threshold')
    ax2.set_ylabel('Rate')
    ax2.set_title('(b) TPR/FPR vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Nonconformity Threshold')
    ax3.set_ylabel('Abstention Rate')
    ax3.set_title('(c) Abstention Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'unified_abstention_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_severity_impact_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create severity impact summary plot."""
    plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    severities = sorted(list(next(iter(results_by_corruption.values())).keys()))
    
    for corruption_name, results in results_by_corruption.items():
        if corruption_name == 'base':
            continue
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
    
    # Style plots
    ax1.axhline(y=0.9, color='k', linestyle='--', label='Target (90%)')
    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Coverage')
    ax1.set_title('(a) Coverage vs Severity')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Severity Level')
    ax2.set_ylabel('Average Set Size')
    ax2.set_title('(b) Set Size vs Severity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Severity Level')
    ax3.set_ylabel('Average Abstention Rate')
    ax3.set_title('(c) Abstention Rate vs Severity')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'severity_impact_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confidence_setsize_plot(
    uncertainty_metrics: Dict[str, Dict[int, Dict]],
    set_sizes: Dict[str, Dict[int, np.ndarray]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create confidence and set size distribution plot."""
    plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    severities = sorted(list(next(iter(uncertainty_metrics.values())).keys()))
    
    for corruption_name in uncertainty_metrics.keys():
        if corruption_name == 'base':
            continue
        # Extract metrics
        entropies = [np.mean(uncertainty_metrics[corruption_name][s]['entropy'])
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
    
    # Style plots
    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Entropy')
    ax1.set_title('(a) Predictive Uncertainty')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Severity Level')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('(b) Model Confidence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Set Size')
    ax3.set_ylabel('Percentage of Samples (%)')
    ax3.set_title('(c) Set Size Distribution (Severity 3)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confidence_setsize_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_threshold_analysis_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create threshold analysis plot."""
    plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    severity = 3  # Medium severity
    
    for corruption_name, results in results_by_corruption.items():
        if corruption_name == 'base':
            continue
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
    
    # Style plots
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Nonconformity Threshold')
    ax1.set_ylabel('TPR - FPR')
    ax1.set_title('(a) Discrimination Ability')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Nonconformity Threshold')
    ax2.set_ylabel('Effective Coverage')
    ax2.set_title('(b) Coverage vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Nonconformity Threshold')
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('(c) Abstention Efficiency')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close() 