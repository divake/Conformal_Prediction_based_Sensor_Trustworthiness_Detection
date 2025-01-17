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
    severities: List[int],
    coverages: List[float],
    set_sizes: List[float],
    abstention_rates: List[float],
    save_dir: str = 'plots_vision/metrics'
) -> None:
    """Plot key metrics against severity levels."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 5))
    
    # Coverage plot
    plt.subplot(1, 3, 1)
    plt.plot(severities, coverages, 'o-', label='Empirical Coverage', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    plt.xlabel('Fog Severity')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Severity')
    plt.grid(True)
    plt.legend()
    
    # Set size plot
    plt.subplot(1, 3, 2)
    plt.plot(severities, set_sizes, 'o-', linewidth=2)
    plt.xlabel('Fog Severity')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Severity')
    plt.grid(True)
    
    # Abstention rate plot
    plt.subplot(1, 3, 3)
    plt.plot(severities, abstention_rates, 'o-', linewidth=2)
    plt.xlabel('Fog Severity')
    plt.ylabel('Abstention Rate')
    plt.title('Abstention Rate vs Severity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(
    results_by_severity: Dict[int, Dict],
    save_dir: str = 'plots_vision/roc'
) -> None:
    """Plot ROC curves for each severity level."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(8, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_by_severity)))
    
    for (severity, severity_results), color in zip(sorted(results_by_severity.items()), colors):
        # Get abstention results for this severity
        abstention_results = severity_results['abstention_results']
        thresholds = sorted(abstention_results.keys())
        
        # Extract TPR and FPR values
        fpr_rates = [abstention_results[t]['fpr'] for t in thresholds]
        tpr_rates = [abstention_results[t]['tpr'] for t in thresholds]
        auc = severity_results['auc']
        
        plt.plot(fpr_rates, tpr_rates, color=color, linewidth=2,
                label=f'Severity {severity} (AUC = {auc:.3f})')
    
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
    set_sizes_by_severity: Dict[int, np.ndarray],
    save_dir: str = 'plots_vision/set_sizes'
) -> None:
    """Plot set size distributions for each severity level."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(set_sizes_by_severity)))
    
    for severity, color in zip(sorted(set_sizes_by_severity.keys()), colors):
        set_sizes = set_sizes_by_severity[severity]
        unique_sizes, counts = np.unique(set_sizes, return_counts=True)
        percentages = (counts / len(set_sizes)) * 100
        
        plt.plot(unique_sizes, percentages, 'o-', color=color, linewidth=2,
                label=f'Severity {severity}')
    
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
    tpr_rates: np.ndarray,
    fpr_rates: np.ndarray,
    abstention_rates: np.ndarray,
    severity: int,
    save_dir: str = 'plots_vision/abstention'
) -> None:
    """Plot abstention analysis metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Abstention Analysis (Severity {severity})', fontsize=14)
    
    # ROC curve
    axs[0, 0].plot(fpr_rates, tpr_rates, 'o-', linewidth=2)
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axs[0, 0].set_xlabel('False Positive Rate')
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[0, 0].set_title('ROC Curve')
    axs[0, 0].grid(True)
    
    # Abstention rate vs threshold
    axs[0, 1].plot(thresholds, abstention_rates, 'o-', linewidth=2)
    axs[0, 1].set_xlabel('Threshold')
    axs[0, 1].set_ylabel('Abstention Rate')
    axs[0, 1].set_title('Abstention Rate vs Threshold')
    axs[0, 1].grid(True)
    
    # TPR/FPR vs threshold
    axs[1, 0].plot(thresholds, tpr_rates, 'o-', linewidth=2, label='TPR')
    axs[1, 0].plot(thresholds, fpr_rates, 'o-', linewidth=2, label='FPR')
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylabel('Rate')
    axs[1, 0].set_title('TPR and FPR vs Threshold')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # TPR-FPR difference
    diff_rates = tpr_rates - fpr_rates
    axs[1, 1].plot(thresholds, diff_rates, 'o-', linewidth=2)
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