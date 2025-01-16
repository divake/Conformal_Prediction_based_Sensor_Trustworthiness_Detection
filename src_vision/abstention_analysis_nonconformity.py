# src_vision/abstention_analysis_nonconformity.py

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import timm
from scipy import stats
from data.cifar100 import setup_cifar100
from data.corruptions import CorruptedCIFAR100Dataset, FogCorruption
from utils.visualization import plot_metrics_vs_severity, plot_roc_curves, plot_set_size_distribution, create_plot_dirs, plot_abstention_analysis
from utils.model_utils import get_model_predictions
from utils.logging_utils import setup_logging


def compute_conformal_scores(softmax_scores: np.ndarray, labels: np.ndarray, k_reg: int = 5, lam_reg: float = 0.01) -> np.ndarray:
    """
    Compute adaptive conformal scores following RAPS approach.
    
    Args:
        softmax_scores: softmax probabilities [n_samples, n_classes]
        labels: true labels [n_samples]
        k_reg: number of top classes without regularization
        lam_reg: regularization strength
    """
    n_samples, n_classes = softmax_scores.shape
    
    # Create regularization vector
    reg_vec = np.array([0]*k_reg + [lam_reg]*(n_classes-k_reg))[None,:]
    
    # Sort probabilities and add regularization
    sort_idx = np.argsort(softmax_scores, axis=1)[:,::-1]
    sorted_probs = np.take_along_axis(softmax_scores, sort_idx, axis=1)
    sorted_reg = sorted_probs + reg_vec
    
    # Find positions of true labels
    true_label_pos = np.where(sort_idx == labels[:,None])[1]
    
    # Compute scores with randomization
    rand_terms = np.random.rand(n_samples) * sorted_reg[np.arange(n_samples), true_label_pos]
    scores = sorted_reg.cumsum(axis=1)[np.arange(n_samples), true_label_pos] - rand_terms
    
    return scores

def find_prediction_set_threshold(cal_scores: np.ndarray, alpha: float = 0.1) -> float:
    """
    Find conformal threshold with proper finite sample correction.
    """
    n = len(cal_scores)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    return qhat

def get_prediction_sets(
    softmax_scores: np.ndarray, 
    qhat: float,
    k_reg: int = 5,
    lam_reg: float = 0.01,
    rand: bool = True
) -> np.ndarray:
    """
    Generate prediction sets using conformal threshold with RAPS approach.
    """
    n_samples, n_classes = softmax_scores.shape
    reg_vec = np.array([0]*k_reg + [lam_reg]*(n_classes-k_reg))[None,:]
    
    # Sort and regularize
    sort_idx = np.argsort(softmax_scores, axis=1)[:,::-1]
    sorted_probs = np.take_along_axis(softmax_scores, sort_idx, axis=1)
    sorted_reg = sorted_probs + reg_vec
    
    # Compute cumulative sums
    cumsum_reg = sorted_reg.cumsum(axis=1)
    
    if rand:
        # With randomization
        rand_terms = np.random.rand(n_samples, 1) * sorted_reg
        indicators = (cumsum_reg - rand_terms) <= qhat
    else:
        # Without randomization
        indicators = cumsum_reg - sorted_reg <= qhat
    
    # Map back to original class order
    prediction_sets = np.take_along_axis(indicators, sort_idx.argsort(axis=1), axis=1)
    
    return prediction_sets

def evaluate_sets(prediction_sets: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate prediction set performance."""
    coverage = np.mean(prediction_sets[np.arange(len(labels)), labels])
    set_sizes = prediction_sets.sum(axis=1)
    return {
        'coverage': coverage,
        'avg_set_size': np.mean(set_sizes),
        'std_set_size': np.std(set_sizes),
        'max_set_size': np.max(set_sizes),
        'min_set_size': np.min(set_sizes)
    }


def compute_nonconformity_scores(
    softmax_scores: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """Compute nonconformity scores for abstention."""
    true_class_probs = softmax_scores[np.arange(len(labels)), labels]
    return -np.log(true_class_probs + 1e-7)

def analyze_abstention(
    nonconf_scores: np.ndarray,
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
    candidate_thresholds: np.ndarray
) -> Tuple[float, Dict]:
    """
    Analyze abstention performance and find optimal threshold (second qhat).
    Returns optimal threshold and performance metrics.
    """
    true_labels_in_set = prediction_sets[np.arange(len(true_labels)), true_labels]
    should_abstain = ~true_labels_in_set
    
    results = {}
    max_diff = -float('inf')
    optimal_threshold = None
    
    # Analyze each candidate threshold
    for threshold in candidate_thresholds:
        abstained = nonconf_scores > threshold
        
        tp = np.sum(abstained & should_abstain)
        fp = np.sum(abstained & ~should_abstain)
        tn = np.sum(~abstained & ~should_abstain)
        fn = np.sum(~abstained & should_abstain)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Store results
        results[threshold] = {
            'tpr': tpr,
            'fpr': fpr,
            'abstention_rate': np.mean(abstained)
        }
        
        # Track best threshold based on TPR-FPR difference
        diff = tpr - fpr
        if diff > max_diff:
            max_diff = diff
            optimal_threshold = threshold
    
    return optimal_threshold, results

def calculate_auc(results: Dict) -> float:
    """Calculate AUC for abstention performance."""
    thresholds = sorted(results.keys())
    tpr = [results[t]['tpr'] for t in thresholds]
    fpr = [results[t]['fpr'] for t in thresholds]
    
    # Sort by FPR for proper AUC calculation
    points = sorted(zip(fpr, tpr))
    fpr = [p[0] for p in points]
    tpr = [p[1] for p in points]
    
    return np.trapz(tpr, fpr)

def main():
    # Setup
    np.random.seed(42)  # For reproducible randomization in conformal prediction
    torch.manual_seed(42)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging('vision_conformal')
    
    # Load model and data
    model = timm.create_model('vit_base_patch16_224_in21k', 
                            pretrained=False,
                            num_classes=100,
                            img_size=96).to(device)
    model.load_state_dict(torch.load('/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/vit_phase2_best.pth'))
    
    _, cal_loader, test_loader, _, _, test_dataset = setup_cifar100()
    
    # Conformal prediction parameters
    k_reg = 5  # No regularization for top-5 classes
    lam_reg = 0.01  # Regularization strength
    alpha = 0.1  # Target 90% coverage
    
    # Get calibration predictions
    cal_probs, cal_labels = get_model_predictions(model, cal_loader, device)
    
    # Calibration phase
    logger.info("Starting calibration phase...")
    cal_scores = compute_conformal_scores(cal_probs, cal_labels, k_reg, lam_reg)
    conformal_qhat = find_prediction_set_threshold(cal_scores, alpha=alpha)
    
    # Validate calibration set performance
    cal_sets = get_prediction_sets(cal_probs, conformal_qhat, k_reg, lam_reg)
    cal_metrics = evaluate_sets(cal_sets, cal_labels)
    logger.info(f"Calibration set metrics:")
    logger.info(f"Coverage: {cal_metrics['coverage']:.4f}")
    logger.info(f"Average set size: {cal_metrics['avg_set_size']:.4f}")
    
    # Analysis parameters
    severity_levels = [1, 2, 3, 4, 5]
    abstention_thresholds = np.linspace(0, 5, 50)
    
    # Results storage
    results_by_severity = {}
    severities, coverages, set_sizes, abstention_rates = [], [], [], []
    set_sizes_by_severity = {}
    
    for severity in severity_levels:
        logger.info(f"\nAnalyzing severity {severity}")
        
        # Create corrupted dataset and get predictions
        corrupted_dataset = CorruptedCIFAR100Dataset(
            test_dataset, FogCorruption, severity=severity
        )
        corrupted_loader = DataLoader(
            corrupted_dataset, batch_size=128, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_probs, test_labels = get_model_predictions(model, corrupted_loader, device)
        
        # Generate prediction sets
        prediction_sets = get_prediction_sets(test_probs, conformal_qhat, k_reg, lam_reg)
        metrics = evaluate_sets(prediction_sets, test_labels)
        
        # Store set sizes for distribution plot
        set_sizes_by_severity[severity] = prediction_sets.sum(axis=1)
        
        # Compute nonconformity scores and analyze abstention
        nonconf_scores = compute_nonconformity_scores(test_probs, test_labels)
        abstention_qhat, abstention_results = analyze_abstention(
            nonconf_scores, prediction_sets, test_labels, abstention_thresholds
        )
        
        # Calculate AUC
        auc = calculate_auc(abstention_results)
        
        # Store metrics for plotting
        severities.append(severity)
        coverages.append(metrics['coverage'])
        set_sizes.append(metrics['avg_set_size'])
        mean_abstention = np.mean([res['abstention_rate'] 
                                for res in abstention_results.values()])
        abstention_rates.append(mean_abstention)
        
        # Store results
        results_by_severity[severity] = {
            **metrics,  # Include all metrics from evaluate_sets
            'abstention_qhat': abstention_qhat,
            'abstention_results': abstention_results,
            'auc': auc,
            'prediction_sets': prediction_sets
        }
        
        # Log results
        logger.info(f"Coverage: {metrics['coverage']:.4f}")
        logger.info(f"Average set size: {metrics['avg_set_size']:.4f}")
        logger.info(f"Set size std: {metrics['std_set_size']:.4f}")
        logger.info(f"AUC: {auc:.4f}")
    
    # Create plot directories and generate visualizations
    plot_dirs = create_plot_dirs('plots_vision')
    
    plot_metrics_vs_severity(
        severities=severities,
        coverages=coverages,
        set_sizes=set_sizes,
        abstention_rates=abstention_rates,
        save_dir=plot_dirs['metrics']
    )
    
    plot_roc_curves(results_by_severity, save_dir=plot_dirs['roc'])
    plot_set_size_distribution(set_sizes_by_severity, save_dir=plot_dirs['set_sizes'])
    
    # Generate abstention analysis plots
    for severity in severity_levels:
        results = results_by_severity[severity]['abstention_results']
        thresholds = sorted(results.keys())
        tpr_rates = [results[t]['tpr'] for t in thresholds]
        fpr_rates = [results[t]['fpr'] for t in thresholds]
        current_abstention_rates = [results[t]['abstention_rate'] for t in thresholds]
        
        plot_abstention_analysis(
            thresholds=np.array(thresholds),
            tpr_rates=np.array(tpr_rates),
            fpr_rates=np.array(fpr_rates),
            abstention_rates=np.array(current_abstention_rates),
            severity=severity,
            save_dir=plot_dirs['abstention']
        )
    
    logger.info("\nAnalysis completed. Plots saved in plots_vision directory.")

if __name__ == '__main__':
    main()