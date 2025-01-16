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


def find_prediction_set_threshold(
    cal_scores: np.ndarray,
    alpha: float = 0.1  # For 90% coverage
) -> float:
    """Find conformal threshold (first qhat) for prediction sets."""
    n = len(cal_scores)
    qhat = np.quantile(cal_scores, 1 - alpha, method='higher')
    return qhat

def compute_conformal_scores(
    softmax_scores: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """Compute conformal scores for adaptive prediction sets."""
    n_samples = len(labels)
    scores = []
    
    for i in range(n_samples):
        # Sort probabilities in descending order
        sorted_probs = np.sort(softmax_scores[i])[::-1]
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Find true class position
        true_class_rank = np.where(
            np.argsort(softmax_scores[i])[::-1] == labels[i]
        )[0][0]
        
        # Score is cumulative probability up to true class
        score = cumsum_probs[true_class_rank]
        scores.append(score)
    
    return np.array(scores)

def get_prediction_sets(
    softmax_scores: np.ndarray,
    qhat: float
) -> np.ndarray:
    """Generate prediction sets using conformal threshold."""
    n_samples, n_classes = softmax_scores.shape
    prediction_sets = np.zeros((n_samples, n_classes), dtype=bool)
    
    for i in range(n_samples):
        sorted_idx = np.argsort(softmax_scores[i])[::-1]
        cumsum = np.cumsum(softmax_scores[i][sorted_idx])
        set_size = np.searchsorted(cumsum, qhat) + 1
        prediction_sets[i, sorted_idx[:set_size]] = True
    
    return prediction_sets

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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging('vision_conformal')
    
    # Load model and data
    model = timm.create_model('vit_base_patch16_224_in21k', 
                            pretrained=False,
                            num_classes=100,
                            img_size=96).to(device)
    model.load_state_dict(torch.load('/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/vit_phase2_best.pth'))
    
    _, cal_loader, test_loader, _, _, test_dataset = setup_cifar100()
    
    # Get calibration predictions
    cal_probs, cal_labels = get_model_predictions(model, cal_loader, device)
    
    # Compute calibration scores and find initial qhat
    cal_scores = compute_conformal_scores(cal_probs, cal_labels)
    conformal_qhat = find_prediction_set_threshold(cal_scores)
    
    # Analysis parameters
    severity_levels = [1, 2, 3, 4, 5]
    abstention_thresholds = np.linspace(0, 5, 50)  # Range for log probabilities
    
    # Results storage
    results_by_severity = {}
    severities = []
    coverages = []
    set_sizes = []
    abstention_rates = []
    set_sizes_by_severity = {}
    
    for severity in severity_levels:
        logger.info(f"Analyzing severity {severity}")
        
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
        prediction_sets = get_prediction_sets(test_probs, conformal_qhat)
        coverage = np.mean(prediction_sets[np.arange(len(test_labels)), test_labels])
        avg_set_size = np.mean(prediction_sets.sum(axis=1))
        
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
        coverages.append(coverage)
        set_sizes.append(avg_set_size)
        # Calculate mean abstention rate across thresholds
        mean_abstention = np.mean([res['abstention_rate'] 
                                 for res in abstention_results.values()])
        abstention_rates.append(mean_abstention)
        
        # Store results
        results_by_severity[severity] = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'abstention_qhat': abstention_qhat,
            'abstention_results': abstention_results,
            'auc': auc,
            'prediction_sets': prediction_sets  # Store for possible later use
        }
        
        # Log results
        logger.info(f"Coverage: {coverage:.4f}")
        logger.info(f"Average set size: {avg_set_size:.4f}")
        logger.info(f"AUC: {auc:.4f}")
    
    # Create plot directories
    plot_dirs = create_plot_dirs('plots_vision')
    
    # Generate visualization plots
    plot_metrics_vs_severity(
        severities=severities,
        coverages=coverages,
        set_sizes=set_sizes,
        abstention_rates=abstention_rates,
        save_dir=plot_dirs['metrics']
    )
    
    plot_roc_curves(results_by_severity, save_dir=plot_dirs['roc'])
    plot_set_size_distribution(set_sizes_by_severity, save_dir=plot_dirs['set_sizes'])
    
    # Generate abstention analysis plots for each severity
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
    
    logger.info("Analysis completed. Plots saved in plots_vision directory.")

if __name__ == '__main__':
    main()