# src_vision/utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
import os

def create_plot_dirs(base_dir: str = 'plots_vision') -> Dict[str, Path]:
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
    severities: List[List[int]],  # [[fog_severities], [snow_severities]]
    coverages: List[List[float]], # [[fog_coverages], [snow_coverages]]
    set_sizes: List[List[float]], # [[fog_set_sizes], [snow_set_sizes]]
    abstention_rates: List[List[float]], # [[fog_rates], [snow_rates]]
    labels: List[str] = ['Fog', 'Snow'],
    save_dir: str = 'plots_vision/metrics'
) -> None:
    """Plot key metrics against severity levels for both corruptions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 5))
    
    # Coverage plot
    plt.subplot(1, 3, 1)
    for i, (sev, cov, label) in enumerate(zip(severities, coverages, labels)):
        plt.plot(sev, cov, 'o-', label=f'{label} Coverage', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    plt.xlabel('Severity')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Set size plot
    plt.subplot(1, 3, 2)
    for i, (sev, sizes, label) in enumerate(zip(severities, set_sizes, labels)):
        plt.plot(sev, sizes, 'o-', label=f'{label}', linewidth=2)
    plt.xlabel('Severity')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Abstention rate plot
    plt.subplot(1, 3, 3)
    for i, (sev, rates, label) in enumerate(zip(severities, abstention_rates, labels)):
        plt.plot(sev, rates, 'o-', label=f'{label}', linewidth=2)
    plt.xlabel('Severity')
    plt.ylabel('Abstention Rate')
    plt.title('Abstention Rate vs Severity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(
    results: Dict[str, Dict[int, Dict]],  # {'fog': results_fog, 'snow': results_snow, 'rain': results_rain}
    save_dir: str = 'plots_vision/roc'
) -> None:
    """Plot ROC curves for each severity level and corruption type."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 8))
    
    # Use different color maps for each corruption
    colors_fog = plt.cm.Blues(np.linspace(0.3, 1, 5))
    colors_snow = plt.cm.Oranges(np.linspace(0.3, 1, 5))
    colors_rain = plt.cm.Greens(np.linspace(0.3, 1, 5))  # Add rain colors
    colors_motionblur = plt.cm.Purples(np.linspace(0.3, 1, 5))  # Add motion blur colors
    
    # Plot curves for each corruption type
    for corruption, colors in [('fog', colors_fog), ('snow', colors_snow), ('rain', colors_rain), ('motionblur', colors_motionblur)]:
        for severity, color in zip(sorted(results[corruption].keys()), colors):
            res = results[corruption][severity]['abstention_results']
            thresholds = sorted(res.keys())
            fpr_rates = [res[t]['fpr'] for t in thresholds]
            tpr_rates = [res[t]['tpr'] for t in thresholds]
            auc = results[corruption][severity]['auc']
            
            plt.plot(fpr_rates, tpr_rates, color=color, linewidth=2,
                    label=f'{corruption.capitalize()} Severity {severity} (AUC = {auc:.3f})')
    
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
    set_sizes_dict: Dict[str, Dict[int, np.ndarray]],  # {'fog': {...}, 'snow': {...}}
    save_dir: str = 'plots_vision/set_sizes'
) -> None:
    """Plot set size distributions for each severity level and corruption type."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    
    # Different color schemes for fog and snow
    colors_fog = plt.cm.Blues(np.linspace(0.3, 1, 5))
    colors_snow = plt.cm.Oranges(np.linspace(0.3, 1, 5))
    
    # Plot fog distributions
    for severity, color in zip(sorted(set_sizes_dict['fog'].keys()), colors_fog):
        set_sizes = set_sizes_dict['fog'][severity]
        unique_sizes, counts = np.unique(set_sizes, return_counts=True)
        percentages = (counts / len(set_sizes)) * 100
        
        plt.plot(unique_sizes, percentages, 'o-', color=color, linewidth=2,
                label=f'Fog Severity {severity}')
    
    # Plot snow distributions
    for severity, color in zip(sorted(set_sizes_dict['snow'].keys()), colors_snow):
        set_sizes = set_sizes_dict['snow'][severity]
        unique_sizes, counts = np.unique(set_sizes, return_counts=True)
        percentages = (counts / len(set_sizes)) * 100
        
        plt.plot(unique_sizes, percentages, 'o-', color=color, linewidth=2,
                label=f'Snow Severity {severity}')
    
    plt.xlabel('Prediction Set Size')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Set Size Distribution by Severity Level and Corruption Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'set_size_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_abstention_analysis(
    thresholds: np.ndarray,
    results_fog: Dict,
    results_snow: Dict,
    results_rain: Dict,  # Add rain results
    results_motionblur: Dict,  # Add motion blur
    severity: int,
    save_dir: str = 'plots_vision/abstention'
) -> None:
    """Plot abstention analysis metrics for all corruptions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Abstention Analysis (Severity {severity})', fontsize=14)
    
    # Extract metrics for all corruptions
    tpr_fog = [results_fog[t]['tpr'] for t in thresholds]
    fpr_fog = [results_fog[t]['fpr'] for t in thresholds]
    abs_fog = [results_fog[t]['abstention_rate'] for t in thresholds]
    
    tpr_snow = [results_snow[t]['tpr'] for t in thresholds]
    fpr_snow = [results_snow[t]['fpr'] for t in thresholds]
    abs_snow = [results_snow[t]['abstention_rate'] for t in thresholds]
    
    tpr_rain = [results_rain[t]['tpr'] for t in thresholds]
    fpr_rain = [results_rain[t]['fpr'] for t in thresholds]
    abs_rain = [results_rain[t]['abstention_rate'] for t in thresholds]
    
    tpr_motionblur = [results_motionblur[t]['tpr'] for t in thresholds]
    fpr_motionblur = [results_motionblur[t]['fpr'] for t in thresholds]
    abs_motionblur = [results_motionblur[t]['abstention_rate'] for t in thresholds]
    
    # ROC curve
    axs[0, 0].plot(fpr_fog, tpr_fog, 'b-', linewidth=2, label='Fog')
    axs[0, 0].plot(fpr_snow, tpr_snow, 'orange', linewidth=2, label='Snow')
    axs[0, 0].plot(fpr_rain, tpr_rain, 'g-', linewidth=2, label='Rain')
    axs[0, 0].plot(fpr_motionblur, tpr_motionblur, 'purple', linewidth=2, label='Motion Blur')
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axs[0, 0].set_xlabel('False Positive Rate')
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[0, 0].set_title('ROC Curve')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Abstention rate vs threshold
    axs[0, 1].plot(thresholds, abs_fog, 'b-', linewidth=2, label='Fog')
    axs[0, 1].plot(thresholds, abs_snow, 'orange', linewidth=2, label='Snow')
    axs[0, 1].plot(thresholds, abs_rain, 'g-', linewidth=2, label='Rain')
    axs[0, 1].plot(thresholds, abs_motionblur, 'purple', linewidth=2, label='Motion Blur')
    axs[0, 1].set_xlabel('Threshold')
    axs[0, 1].set_ylabel('Abstention Rate')
    axs[0, 1].set_title('Abstention Rate vs Threshold')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # TPR/FPR vs threshold
    axs[1, 0].plot(thresholds, tpr_fog, 'b-', linewidth=2, label='TPR (Fog)')
    axs[1, 0].plot(thresholds, fpr_fog, 'b--', linewidth=2, label='FPR (Fog)')
    axs[1, 0].plot(thresholds, tpr_snow, 'orange', linewidth=2, label='TPR (Snow)')
    axs[1, 0].plot(thresholds, fpr_snow, 'orange', linestyle='--', linewidth=2, label='FPR (Snow)')
    axs[1, 0].plot(thresholds, tpr_rain, 'g-', linewidth=2, label='TPR (Rain)')
    axs[1, 0].plot(thresholds, fpr_rain, 'g--', linewidth=2, label='FPR (Rain)')
    axs[1, 0].plot(thresholds, tpr_motionblur, 'purple', linewidth=2, label='TPR (Motion Blur)')
    axs[1, 0].plot(thresholds, fpr_motionblur, 'purple', linestyle='--', linewidth=2, label='FPR (Motion Blur)')
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylabel('Rate')
    axs[1, 0].set_title('TPR and FPR vs Threshold')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # TPR-FPR difference
    diff_fog = np.array(tpr_fog) - np.array(fpr_fog)
    diff_snow = np.array(tpr_snow) - np.array(fpr_snow)
    diff_rain = np.array(tpr_rain) - np.array(fpr_rain)
    diff_motionblur = np.array(tpr_motionblur) - np.array(fpr_motionblur)
    axs[1, 1].plot(thresholds, diff_fog, 'b-', linewidth=2, label='Fog')
    axs[1, 1].plot(thresholds, diff_snow, 'orange', linewidth=2, label='Snow')
    axs[1, 1].plot(thresholds, diff_rain, 'g-', linewidth=2, label='Rain')
    axs[1, 1].plot(thresholds, diff_motionblur, 'purple', linewidth=2, label='Motion Blur')
    axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('TPR - FPR')
    axs[1, 1].set_title('TPR-FPR Difference')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'abstention_analysis_severity_{severity}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distributions(
    softmax_scores_by_severity: Dict[int, np.ndarray],
    set_sizes_by_severity: Dict[int, np.ndarray],  # Add this parameter
    save_dir: str
) -> None:
    """
    Plot distributions of model confidence across severities.
    
    Args:
        softmax_scores_by_severity: Dict mapping severity to softmax scores array
        set_sizes_by_severity: Dict mapping severity to set sizes array
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    # Plot top-1 probability distributions
    plt.subplot(2, 2, 1)
    for severity, scores in softmax_scores_by_severity.items():
        top1_probs = np.max(scores, axis=1)
        plt.hist(top1_probs, bins=50, alpha=0.5, label=f'Severity {severity}', density=True)
    plt.xlabel('Top-1 Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Top-1 Probabilities')
    plt.legend()
    
    # Plot entropy distributions
    plt.subplot(2, 2, 2)
    for severity, scores in softmax_scores_by_severity.items():
        entropy = -np.sum(scores * np.log(scores + 1e-10), axis=1)
        plt.hist(entropy, bins=50, alpha=0.5, label=f'Severity {severity}', density=True)
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Entropy')
    plt.legend()
    
    # Plot cumulative probability curves
    plt.subplot(2, 2, 3)
    for severity, scores in softmax_scores_by_severity.items():
        sorted_probs = np.sort(scores, axis=1)[:, ::-1]
        mean_cumsum = np.mean(np.cumsum(sorted_probs, axis=1), axis=0)
        plt.plot(range(1, len(mean_cumsum) + 1), mean_cumsum, 
                label=f'Severity {severity}', marker='o', markersize=3)
    plt.xlabel('Number of Classes')
    plt.ylabel('Cumulative Probability')
    plt.title('Average Cumulative Probability Curves')
    plt.legend()
    
    # Plot set size distributions
    plt.subplot(2, 2, 4)
    for severity, scores in set_sizes_by_severity.items():
        max_score = int(np.max(scores))
        plt.hist(scores, bins=range(0, max_score + 2), 
                alpha=0.5, label=f'Severity {severity}', density=True)
    plt.xlabel('Set Size')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Set Sizes')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_distributions.png'))
    plt.close()

def analyze_severity_impact(
    softmax_scores_by_severity: Dict[int, np.ndarray],
    conformal_qhat: float,
    k_reg: int,
    lam_reg: float,
    save_dir: str
) -> None:
    """
    Analyze how severity affects conformal scores and thresholds.
    
    Args:
        softmax_scores_by_severity: Dict mapping severity to softmax scores array
        conformal_qhat: The conformal threshold
        k_reg: Number of top classes without regularization
        lam_reg: Regularization strength
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot average conformal scores by severity
    for severity, scores in softmax_scores_by_severity.items():
        reg_vec = np.array([0]*k_reg + [lam_reg]*(scores.shape[1]-k_reg))[None,:]
        sort_idx = np.argsort(scores, axis=1)[:,::-1]
        sorted_probs = np.take_along_axis(scores, sort_idx, axis=1)
        sorted_reg = sorted_probs + reg_vec
        cumsum_reg = sorted_reg.cumsum(axis=1)
        
        mean_scores = np.mean(cumsum_reg, axis=0)
        std_scores = np.std(cumsum_reg, axis=0)
        
        plt.plot(range(1, len(mean_scores) + 1), mean_scores, 
                label=f'Severity {severity}')
        plt.fill_between(range(1, len(mean_scores) + 1), 
                        mean_scores - std_scores, 
                        mean_scores + std_scores, 
                        alpha=0.2)
    
    plt.axhline(y=conformal_qhat, color='r', linestyle='--', 
                label='Conformal Threshold')
    plt.xlabel('Number of Classes')
    plt.ylabel('Cumulative Score')
    plt.title('Average Conformal Scores by Severity')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'severity_impact.png'))
    plt.close()

def create_paper_plots(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    uncertainty_metrics_by_corruption: Dict[str, Dict[int, Dict]],
    set_sizes_by_corruption: Dict[str, Dict[int, np.ndarray]],
    save_dir: str = 'plots_vision/result_paper'
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
        'fog': '#2ecc71',    # Green
        'snow': '#3498db',   # Blue
        'rain': '#e74c3c',   # Red
        'blur': '#9b59b6'    # Purple
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
    """Create unified abstention analysis plot."""
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
    plt.savefig(save_dir / 'unified_abstention_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_severity_impact_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create severity impact summary plot."""
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
    plt.savefig(save_dir / 'severity_impact_summary.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_confidence_setsize_plot(
    uncertainty_metrics: Dict[str, Dict[int, Dict]],
    set_sizes: Dict[str, Dict[int, np.ndarray]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create confidence and set size distribution plot."""
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
    plt.savefig(save_dir / 'confidence_setsize_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_threshold_analysis_plot(
    results_by_corruption: Dict[str, Dict[int, Dict]],
    colors: Dict[str, str],
    save_dir: Path
) -> None:
    """Create threshold analysis plot."""
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
    plt.savefig(save_dir / 'threshold_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()