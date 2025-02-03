import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import timm
from scipy import stats
from data_loader import get_imagenet_dataset
from data.corruptions import FogCorruption, CorruptedImageNetDataset
import os
import time
from datetime import datetime, timedelta
from utils.visualization import (
    plot_roc_curves,
    plot_set_size_distribution,
    create_plot_dirs,
    plot_abstention_analysis,
    plot_confidence_distributions,
    create_paper_plots,
    plot_metrics_vs_severity,
    analyze_severity_impact
)


def compute_conformal_scores(
    softmax_scores: np.ndarray, 
    labels: np.ndarray, 
    k_reg: int = 5, 
    lam_reg: float = 0.02
) -> np.ndarray:
    """
    Compute adaptive conformal scores.
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
    lam_reg: float = 0.02,
    rand: bool = True
) -> np.ndarray:
    """
    Generate prediction sets with adaptive regularization based on predictive uncertainty.
    """
    n_samples, n_classes = softmax_scores.shape
    
    # Compute uncertainty metrics for each sample
    entropy = -np.sum(softmax_scores * np.log(softmax_scores + 1e-7), axis=1)
    max_probs = np.max(softmax_scores, axis=1)
    
    # Normalize entropy to [0,1] range
    normalized_entropy = entropy / np.log(n_classes)
    
    # Compute adaptive regularization based on uncertainty
    # High entropy/low confidence -> less regularization
    adaptive_factor = np.clip(1.0 - normalized_entropy, 0.01, 1.0)
    
    # Create sample-specific regularization
    base_reg = 1e-6
    effective_lam = lam_reg * adaptive_factor[:, None]  # Make it broadcastable
    
    # Create regularization vector per sample
    reg_vec = np.concatenate([
        np.zeros((n_samples, k_reg)),
        np.broadcast_to(effective_lam, (n_samples, n_classes - k_reg))
    ], axis=1)
    
    # Sort and regularize
    sort_idx = np.argsort(softmax_scores, axis=1)[:,::-1]
    sorted_probs = np.take_along_axis(softmax_scores, sort_idx, axis=1)
    sorted_reg = sorted_probs + reg_vec
    
    # Compute cumulative sums
    cumsum_reg = sorted_reg.cumsum(axis=1)
    
    if rand:
        rand_terms = np.random.rand(n_samples, 1) * sorted_reg
        indicators = (cumsum_reg - rand_terms) <= qhat
    else:
        indicators = cumsum_reg - sorted_reg <= qhat
    
    # Map back to original class order
    prediction_sets = np.take_along_axis(indicators, sort_idx.argsort(axis=1), axis=1)
    
    return prediction_sets

def compute_uncertainty_metrics(softmax_scores: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute various uncertainty metrics from softmax scores."""
    entropy = -np.sum(softmax_scores * np.log(softmax_scores + 1e-7), axis=1)
    max_probs = np.max(softmax_scores, axis=1)
    margin = np.sort(softmax_scores, axis=1)[:, -1] - np.sort(softmax_scores, axis=1)[:, -2]
    
    return {
        'entropy': entropy,
        'confidence': max_probs,
        'margin': margin,
        'normalized_entropy': entropy / np.log(softmax_scores.shape[1])
    }

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
    """Compute nonconformity scores with adjusted sensitivity."""
    true_class_probs = softmax_scores[np.arange(len(labels)), labels]
    # Add temperature scaling to make scores less extreme
    temperature = 2.0  # Increase this to make abstention less aggressive
    return -np.log(true_class_probs + 1e-7) / temperature

def analyze_abstention(
    nonconf_scores: np.ndarray,
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
    candidate_thresholds: np.ndarray
) -> Tuple[float, Dict]:
    """
    Analyze abstention performance and find optimal threshold.
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

def get_model_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions efficiently using batch processing."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_probs), np.concatenate(all_labels)

def main():
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    # Setup
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure logging
    logger = logging.getLogger('imagenet_conformal')
    logger.setLevel(logging.INFO)
    
    # Add console handler if not already added
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info("Starting ImageNet conformal prediction analysis...")
    logger.info(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model from checkpoint
    CHECKPOINT_PATH = '/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/vit_imagenet.pth'
    logger.info(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    
    # Initialize model with efficient settings
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.to(device)  # Move to GPU before loading state dict
    
    # Load state dict efficiently
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Enable inference optimizations
    with torch.amp.autocast('cuda'):  # Enable automatic mixed precision
        model = torch.jit.optimize_for_inference(torch.jit.script(model))
    
    logger.info("Model loaded successfully and optimized for inference")
    
    # Load and split dataset
    dataset, cal_loader, test_loader = get_imagenet_dataset(
        samples_per_class=50,  # Use 50 samples per class for balanced evaluation
        split_ratio=0.5        # 20% for calibration, 80% for test
    )
    
    # Conformal prediction parameters
    k_reg = 5  # No regularization for top-5 classes
    lam_reg = 0.02  # Regularization strength
    alpha = 0.1  # Target 90% coverage
    
    # Analysis parameters
    low_range = np.linspace(0, 0.1, 50)
    mid_range = np.linspace(0.1, 1, 50)
    high_range = np.exp(np.linspace(0, np.log(5), 50))
    abstention_thresholds = np.unique(np.concatenate([low_range, mid_range, high_range]))
    
    # Get calibration predictions
    logger.info("Getting calibration predictions...")
    cal_probs, cal_labels = get_model_predictions(model, cal_loader, device)
    
    # Calibration phase
    logger.info("Starting calibration phase...")
    cal_scores = compute_conformal_scores(cal_probs, cal_labels, k_reg, lam_reg)
    conformal_qhat = find_prediction_set_threshold(cal_scores, alpha=alpha)
    logger.info(f"Conformal qhat: {conformal_qhat:.4f}")
    
    # Validate calibration set performance
    cal_sets = get_prediction_sets(cal_probs, conformal_qhat, k_reg, lam_reg)
    cal_metrics = evaluate_sets(cal_sets, cal_labels)
    logger.info(f"Calibration set metrics:")
    logger.info(f"Coverage: {cal_metrics['coverage']:.4f}")
    logger.info(f"Average set size: {cal_metrics['avg_set_size']:.4f}")
    
    # Compute abstention threshold on calibration data
    cal_nonconf_scores = compute_nonconformity_scores(cal_probs, cal_labels)
    abstention_qhat, _ = analyze_abstention(
        cal_nonconf_scores, 
        cal_sets,
        cal_labels,
        abstention_thresholds
    )
    logger.info(f"Abstention qhat: {abstention_qhat:.4f}")
    
    # Initialize results storage
    results_by_corruption = {
        'base': {},
        'rain': {}
    }
    set_sizes_by_corruption = {
        'base': {},
        'rain': {}
    }
    uncertainty_metrics_by_corruption = {
        'base': {},
        'rain': {}
    }
    
    # Analyze base test set
    logger.info("\nAnalyzing base test set...")
    test_probs, test_labels = get_model_predictions(model, test_loader, device)
    prediction_sets = get_prediction_sets(test_probs, conformal_qhat, k_reg, lam_reg)
    metrics = evaluate_sets(prediction_sets, test_labels)
    
    # Compute abstention metrics
    nonconf_scores = compute_nonconformity_scores(test_probs, test_labels)
    true_labels_in_set = prediction_sets[np.arange(len(test_labels)), test_labels]
    should_abstain = ~true_labels_in_set
    
    # Calculate abstention results
    abstention_results = {}
    for threshold in abstention_thresholds:
        abstained = nonconf_scores > threshold
        tp = np.sum(abstained & should_abstain)
        fp = np.sum(abstained & ~should_abstain)
        tn = np.sum(~abstained & ~should_abstain)
        fn = np.sum(~abstained & should_abstain)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        abstention_results[threshold] = {
            'tpr': tpr,
            'fpr': fpr,
            'abstention_rate': np.mean(abstained)
        }
    
    # Calculate AUC
    auc = calculate_auc(abstention_results)
    
    # Store results
    results_by_corruption['base'][1] = {
        **metrics,
        'abstention_qhat': abstention_qhat,
        'abstention_results': abstention_results,
        'auc': auc,
        'prediction_sets': prediction_sets,
        'softmax_scores': test_probs,  # Store softmax scores
        'uncertainty_metrics': compute_uncertainty_metrics(test_probs)  # Store uncertainty metrics
    }
    set_sizes_by_corruption['base'][1] = prediction_sets.sum(axis=1)
    uncertainty_metrics_by_corruption['base'][1] = compute_uncertainty_metrics(test_probs)
    
    # Analyze rain corruption
    logger.info("\nAnalyzing rain corruption...")
    for severity in range(1, 6):
        logger.info(f"Processing severity {severity}")
        
        # Create corrupted dataset with optimized wrapper
        corrupted_dataset = CorruptedImageNetDataset(
            dataset.test_dataset,
            FogCorruption,
            severity=severity
        )
        corrupted_loader = DataLoader(
            corrupted_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Get predictions for corrupted data
        corrupt_probs, corrupt_labels = get_model_predictions(model, corrupted_loader, device)
        
        # Get prediction sets
        corrupt_sets = get_prediction_sets(corrupt_probs, conformal_qhat, k_reg, lam_reg)
        corrupt_metrics = evaluate_sets(corrupt_sets, corrupt_labels)
        
        # Compute abstention metrics for corrupted data
        corrupt_nonconf_scores = compute_nonconformity_scores(corrupt_probs, corrupt_labels)
        corrupt_true_labels_in_set = corrupt_sets[np.arange(len(corrupt_labels)), corrupt_labels]
        corrupt_should_abstain = ~corrupt_true_labels_in_set
        
        # Calculate abstention results for corrupted data
        corrupt_abstention_results = {}
        for threshold in abstention_thresholds:
            abstained = corrupt_nonconf_scores > threshold
            tp = np.sum(abstained & corrupt_should_abstain)
            fp = np.sum(abstained & ~corrupt_should_abstain)
            tn = np.sum(~abstained & ~corrupt_should_abstain)
            fn = np.sum(~abstained & corrupt_should_abstain)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            corrupt_abstention_results[threshold] = {
                'tpr': tpr,
                'fpr': fpr,
                'abstention_rate': np.mean(abstained)
            }
        
        # Calculate AUC for corrupted data
        corrupt_auc = calculate_auc(corrupt_abstention_results)
        
        # Store results
        results_by_corruption['rain'][severity] = {
            **corrupt_metrics,
            'abstention_qhat': abstention_qhat,
            'abstention_results': corrupt_abstention_results,
            'auc': corrupt_auc,
            'prediction_sets': corrupt_sets,
            'softmax_scores': corrupt_probs,  # Store softmax scores
            'uncertainty_metrics': compute_uncertainty_metrics(corrupt_probs)  # Store uncertainty metrics
        }
        set_sizes_by_corruption['rain'][severity] = corrupt_sets.sum(axis=1)
        uncertainty_metrics_by_corruption['rain'][severity] = compute_uncertainty_metrics(corrupt_probs)
        
        logger.info(f"Severity {severity} - Coverage: {corrupt_metrics['coverage']:.4f}, "
                   f"Avg Set Size: {corrupt_metrics['avg_set_size']:.4f}, "
                   f"AUC: {corrupt_auc:.4f}")
    
    # Create plots directory
    plot_dirs = create_plot_dirs('plots_imagenet')
    
    # Analyze severity impact on conformal scores
    logger.info("\nAnalyzing severity impact on conformal scores...")
    analyze_severity_impact(
        softmax_scores_by_corruption={
            'base': {1: test_probs},  # Base test set
            'rain': {severity: results_by_corruption['rain'][severity]['softmax_scores'] 
                    for severity in range(1, 6)}
        },
        conformal_qhat=conformal_qhat,
        k_reg=k_reg,
        lam_reg=lam_reg,
        save_dir=plot_dirs['metrics']
    )
    
    # Analyze uncertainty metrics
    logger.info("\nAnalyzing uncertainty metrics...")
    uncertainty_metrics = {
        'base': {1: compute_uncertainty_metrics(test_probs)},
        'rain': {severity: compute_uncertainty_metrics(results_by_corruption['rain'][severity]['softmax_scores'])
                for severity in range(1, 6)}
    }
    
    # Plot confidence and entropy distributions
    plot_confidence_distributions(
        uncertainty_metrics,
        colors={'base': '#2ecc71', 'rain': '#3498db'},
        save_dir=plot_dirs['metrics']
    )
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_roc_curves(results_by_corruption, plot_dirs['roc'])
    plot_set_size_distribution(set_sizes_by_corruption, plot_dirs['set_sizes'])
    plot_metrics_vs_severity(results_by_corruption, plot_dirs['metrics'])
    
    # Generate abstention analysis plots
    plot_abstention_analysis(
        thresholds=abstention_thresholds,
        results_base=abstention_results,
        save_dir=plot_dirs['abstention']
    )
    
    # Create paper plots
    create_paper_plots(
        results_by_corruption=results_by_corruption,
        uncertainty_metrics_by_corruption=uncertainty_metrics,
        set_sizes_by_corruption={
            'base': {1: prediction_sets.sum(axis=1)},
            'rain': {severity: results_by_corruption['rain'][severity]['prediction_sets'].sum(axis=1)
                    for severity in range(1, 6)}
        },
        save_dir=os.path.join(plot_dirs['paper'])
    )
    
    # Print final summary with detailed metrics
    logger.info("\n=== Final Results Summary ===")
    logger.info("\nBase Model Performance:")
    base_metrics = results_by_corruption['base'][1]
    logger.info(f"Coverage: {base_metrics['coverage']:.4f}")
    logger.info(f"Average Set Size: {base_metrics['avg_set_size']:.4f}")
    logger.info(f"AUC: {base_metrics['auc']:.4f}")
    
    logger.info("\nRain Corruption Performance by Severity:")
    for severity in range(1, 6):
        if severity in results_by_corruption['rain']:
            rain_metrics = results_by_corruption['rain'][severity]
            logger.info(f"\nSeverity {severity}:")
            logger.info(f"Coverage: {rain_metrics['coverage']:.4f}")
            logger.info(f"Average Set Size: {rain_metrics['avg_set_size']:.4f}")
            logger.info(f"AUC: {rain_metrics['auc']:.4f}")
            logger.info(f"Abstention Rate: {np.mean([res['abstention_rate'] for res in rain_metrics['abstention_results'].values()]):.4f}")
            
            # Add uncertainty metrics
            uncertainty = uncertainty_metrics['rain'][severity]
            logger.info(f"Average Entropy: {np.mean(uncertainty['entropy']):.4f}")
            logger.info(f"Average Confidence: {np.mean(uncertainty['confidence']):.4f}")
            logger.info(f"Average Margin: {np.mean(uncertainty['margin']):.4f}")
    
    logger.info("\n=== Plots Generated ===")
    logger.info(f"ROC Curves: {plot_dirs['roc']}")
    logger.info(f"Set Size Distributions: {plot_dirs['set_sizes']}")
    logger.info(f"Abstention Analysis: {plot_dirs['abstention']}")
    logger.info(f"Paper Plots: {plot_dirs['paper']}")
    logger.info(f"Metrics and Uncertainty Analysis: {plot_dirs['metrics']}")
    
    logger.info("\nAnalysis completed successfully. All plots saved in plots_imagenet directory.")
    
    # Record end time and calculate duration
    end_time = time.time()
    end_datetime = datetime.now()
    duration = end_time - start_time
    duration_formatted = str(timedelta(seconds=int(duration)))
    
    logger.info("\n=== Timing Information ===")
    logger.info(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration_formatted}")

if __name__ == '__main__':
    main() 