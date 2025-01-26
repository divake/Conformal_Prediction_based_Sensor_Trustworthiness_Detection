import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def create_plot_dirs(base_dir: str) -> Dict[str, str]:
    """
    Create directory structure for plots.
    
    Args:
        base_dir: Base directory for plots
        
    Returns:
        Dictionary mapping plot types to their directories
    """
    plot_types = ['metrics', 'roc', 'set_sizes', 'abstention', 'confidence', 'paper', 'corruptions']
    plot_dirs = {}
    
    for plot_type in plot_types:
        plot_dir = os.path.join(base_dir, plot_type)
        os.makedirs(plot_dir, exist_ok=True)
        plot_dirs[plot_type] = plot_dir
    
    return plot_dirs

def plot_metrics_vs_severity(
    severities: List[List[int]],
    coverages: List[List[float]],
    set_sizes: List[List[float]],
    abstention_rates: List[List[float]],
    labels: List[str],
    save_dir: str
) -> None:
    """
    Plot metrics against corruption severity.
    
    Args:
        severities: List of severity levels for each corruption
        coverages: List of coverage values for each corruption
        set_sizes: List of set sizes for each corruption
        abstention_rates: List of abstention rates for each corruption
        labels: Names of corruptions
        save_dir: Directory to save plots
    """
    metrics = {
        'Coverage': coverages,
        'Average Set Size': set_sizes,
        'Abstention Rate': abstention_rates
    }
    
    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=(10, 6))
        
        for i, (values, label) in enumerate(zip(metric_values, labels)):
            plt.plot(severities[i], values, marker='o', label=label)
        
        plt.xlabel('Corruption Severity')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs. Corruption Severity')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, f'{metric_name.lower().replace(" ", "_")}_vs_severity.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_roc_curves(
    results: Dict[str, Dict[int, Dict]],
    save_dir: str,
    title: str = "ROC Curves"
) -> None:
    """Plot ROC curves for each corruption type and severity."""
    # Plot for each severity
    for severity in range(1, 6):
        plt.figure(figsize=(10, 10))
        has_data = False
        
        for corruption_name, corruption_results in results.items():
            if severity in corruption_results:
                result = corruption_results[severity]
                if 'abstention_results' in result:
                    thresholds = sorted(result['abstention_results'].keys())
                    tpr = [result['abstention_results'][t]['tpr'] for t in thresholds]
                    fpr = [result['abstention_results'][t]['fpr'] for t in thresholds]
                    auc = result.get('auc', 0.0)
                    
                    plt.plot(fpr, tpr, 
                            label=f'{corruption_name.capitalize()} (AUC={auc:.3f})')
                    has_data = True
        
        if has_data:
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{title} (Severity {severity})')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(save_dir, f'roc_curves_severity_{severity}.png'),
                       bbox_inches='tight', dpi=300)
        plt.close()

def plot_set_size_distribution(
    set_sizes_by_corruption: Dict[str, Dict[int, np.ndarray]],
    save_dir: str,
    title: str = "Distribution of Prediction Set Sizes"
) -> None:
    """
    Plot distribution of prediction set sizes.
    
    Args:
        set_sizes_by_corruption: Dictionary mapping corruption names to their set sizes
        save_dir: Directory to save plots
        title: Plot title
    """
    # Plot for each severity
    for severity in range(1, 6):
        plt.figure(figsize=(12, 6))
        has_data = False
        
        for corruption_name, set_sizes in set_sizes_by_corruption.items():
            if severity in set_sizes:
                sizes = set_sizes[severity]
                # Add explicit label for the plot
                sns.kdeplot(sizes, label=f'{corruption_name.capitalize()} Distribution')
                plt.axvline(np.mean(sizes), linestyle='--', alpha=0.5, 
                          label=f'{corruption_name.capitalize()} Mean')
                has_data = True
        
        if has_data:
            plt.xlabel('Set Size')
            plt.ylabel('Density')
            plt.title(f'{title} (Severity {severity})')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(save_dir, f'set_size_distribution_severity_{severity}.png'),
                       bbox_inches='tight', dpi=300)
        plt.close()

def plot_abstention_analysis(
    thresholds: np.ndarray,
    results_base: Dict,
    save_dir: str = None,
    title: str = "Abstention Analysis"
) -> None:
    """
    Plot abstention analysis metrics for base case.
    
    Args:
        thresholds: Array of abstention thresholds
        results_base: Dictionary of base results
        save_dir: Directory to save plots
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot base results with explicit labels
    tpr = [results_base[t]['tpr'] for t in thresholds]
    fpr = [results_base[t]['fpr'] for t in thresholds]
    abstention_rates = [results_base[t]['abstention_rate'] for t in thresholds]
    
    plt.plot(thresholds, tpr, label='True Positive Rate', linestyle='-', color='blue')
    plt.plot(thresholds, fpr, label='False Positive Rate', linestyle='--', color='red')
    plt.plot(thresholds, abstention_rates, label='Abstention Rate', linestyle=':', color='green')
    
    plt.xlabel('Abstention Threshold')
    plt.ylabel('Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'abstention_analysis_base.png'),
                    bbox_inches='tight', dpi=300)
    plt.close()

def plot_confidence_distributions(
    softmax_scores: np.ndarray,
    labels: np.ndarray,
    prediction_sets: np.ndarray,
    save_dir: str,
    title: str = "Confidence Score Distributions"
) -> None:
    """
    Plot distributions of confidence scores.
    
    Args:
        softmax_scores: Model's softmax probabilities
        labels: True labels
        prediction_sets: Prediction sets
        save_dir: Directory to save plots
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Get confidence scores
    confidence_scores = np.max(softmax_scores, axis=1)
    
    # Split by correctness
    correct_mask = prediction_sets[np.arange(len(labels)), labels]
    correct_conf = confidence_scores[correct_mask]
    incorrect_conf = confidence_scores[~correct_mask]
    
    # Plot distributions with explicit labels
    if len(correct_conf) > 0:
        sns.kdeplot(correct_conf, label='Correct Predictions', color='green')
    if len(incorrect_conf) > 0:
        sns.kdeplot(incorrect_conf, label='Incorrect Predictions', color='red')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'confidence_distributions.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_paper_plots(
    results_by_corruption: Dict[str, Dict],
    uncertainty_metrics_by_corruption: Dict[str, Dict],
    set_sizes_by_corruption: Dict[str, Dict],
    save_dir: Optional[str] = None
) -> None:
    """
    Create publication-quality plots for the paper.
    """
    if save_dir is None:
        save_dir = os.path.join('plots_imagenet', 'paper')
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style for paper plots
    plt.style.use('seaborn')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })
    
    # Plot ROC curves for each severity
    for severity in range(1, 6):
        plt.figure(figsize=(10, 8))
        has_data = False
        
        for corruption_name, results in results_by_corruption.items():
            if severity in results and 'abstention_results' in results[severity]:
                result = results[severity]
                thresholds = sorted(result['abstention_results'].keys())
                tpr = [result['abstention_results'][t]['tpr'] for t in thresholds]
                fpr = [result['abstention_results'][t]['fpr'] for t in thresholds]
                auc = result.get('auc', 0.0)
                
                plt.plot(fpr, tpr, label=f'{corruption_name.capitalize()} (AUC={auc:.3f})')
                has_data = True
        
        if has_data:
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves (Severity {severity})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'roc_curves_severity_{severity}.png'),
                       bbox_inches='tight', dpi=300)
        plt.close()
    
    # Plot set size distributions for each severity
    for severity in range(1, 6):
        plt.figure(figsize=(12, 6))
        has_data = False
        
        for corruption_name, set_sizes in set_sizes_by_corruption.items():
            if severity in set_sizes:
                sizes = set_sizes[severity]
                sns.kdeplot(sizes, label=f'{corruption_name.capitalize()} Distribution')
                plt.axvline(np.mean(sizes), linestyle='--', alpha=0.5,
                          label=f'{corruption_name.capitalize()} Mean')
                has_data = True
        
        if has_data:
            plt.xlabel('Set Size')
            plt.ylabel('Density')
            plt.title(f'Set Size Distribution (Severity {severity})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'set_size_distribution_severity_{severity}.png'),
                       bbox_inches='tight', dpi=300)
        plt.close()
    
    # Plot uncertainty metrics for each corruption type and severity
    for corruption_name, metrics in uncertainty_metrics_by_corruption.items():
        for severity in metrics:
            plt.figure(figsize=(12, 6))
            metric_data = metrics[severity]
            
            # Plot entropy distribution
            if 'entropy' in metric_data:
                sns.kdeplot(metric_data['entropy'], label='Entropy')
            
            # Plot confidence distribution
            if 'confidence' in metric_data:
                sns.kdeplot(metric_data['confidence'], label='Confidence')
            
            plt.xlabel('Metric Value')
            plt.ylabel('Density')
            plt.title(f'Uncertainty Metrics - {corruption_name.capitalize()} (Severity {severity})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'uncertainty_metrics_{corruption_name}_severity_{severity}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
    
    # Reset style
    plt.style.use('default') 