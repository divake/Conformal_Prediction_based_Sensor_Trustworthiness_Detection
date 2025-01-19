#src/abstention_analysis_nonconformity.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import binom
from conformal_prediction import conformal_prediction, get_softmax_predictions, create_dataloader
from data.corruptions import CorruptedModelNet40Dataset, OcclusionCorruption  # Updated import
from config import Config
from data.dataset import ModelNet40Dataset
from models.point_transformer_v2 import PointTransformerV2
from torch.utils.data import DataLoader
import logging
from utils.visualization import create_plot_dirs, plot_nonconformity_analysis, plot_metrics_vs_severity, plot_roc_curves, plot_set_size_distribution

def setup_logging(logger_name: str = 'nonconformity_abstention') -> logging.Logger:
    """Set up logging configuration
    
    Args:
        logger_name (str): Name for the logger (default: 'nonconformity_abstention')
    
    Returns:
        logging.Logger: Logger instance
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"{logger_name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(logger_name)

def find_prediction_set_threshold(
    cal_scores: np.ndarray,
    alpha: float = 0.1,
    randomized: bool = True
) -> float:
    """
    Find threshold for prediction sets using RAPS scores
    
    Args:
        cal_scores: Calibration set RAPS scores
        alpha: Target miscoverage level (1-alpha = desired coverage)
        randomized: Whether to use randomized thresholding
        
    Returns:
        Optimal threshold for prediction sets
    """
    n = len(cal_scores)
    level_adjusted = (1 - alpha) * (1 + 1/n)
    return np.quantile(cal_scores, 1 - level_adjusted, method='higher')

def compute_raps_scores(
    softmax_scores: np.ndarray,
    labels: np.ndarray,
    k_reg: int = 5,
    lam_reg: float = 0.01,
    severity: int = 1
) -> np.ndarray:
    """
    Compute RAPS scores with severity-aware scaling
    """
    n_samples = len(labels)
    n_classes = softmax_scores.shape[1]
    
    # More aggressive regularization for higher severities
    base_reg = lam_reg * (1 + 0.25 * (severity - 1))
    reg_vec = np.zeros(n_classes)
    reg_vec[k_reg:] = base_reg
    
    scores = []
    for i in range(n_samples):
        pi = softmax_scores[i].argsort()[::-1]
        srt = softmax_scores[i][pi]
        true_class_idx = np.where(pi == labels[i])[0][0]
        
        # Adaptive confidence scaling
        conf_scale = 1.0 + 0.1 * (severity - 1)
        norm_probs = srt / (srt[0] * conf_scale)
        
        # Compute score with severity-aware size
        max_size = min(2 + int(1.5 * severity), 6)
        cum_prob = np.cumsum(norm_probs[:max_size])
        cum_reg = cum_prob + reg_vec[:max_size] * (1 + 0.1 * (severity - 1))
        
        if true_class_idx < max_size:
            score = cum_reg[true_class_idx]
        else:
            score = cum_reg[-1] + reg_vec[true_class_idx]
        scores.append(score)
    
    return np.array(scores)

def get_prediction_sets_raps(
    softmax_scores: np.ndarray,
    threshold: float,
    k_reg: int = 5,
    lam_reg: float = 0.01,
    severity: int = 1
) -> np.ndarray:
    """Generate prediction sets with severity-aware scaling"""
    n_samples, n_classes = softmax_scores.shape
    prediction_sets = np.zeros((n_samples, n_classes), dtype=bool)
    
    base_reg = lam_reg * (1 + 0.25 * (severity - 1))
    reg_vec = np.zeros(n_classes)
    reg_vec[k_reg:] = base_reg
    
    # Adaptive size control
    min_size = 1 + int(0.5 * (severity - 1))
    max_size = min(2 + int(1.5 * severity), 6)
    
    for i in range(n_samples):
        pi = softmax_scores[i].argsort()[::-1]
        srt = softmax_scores[i][pi]
        
        # Apply confidence scaling
        conf_scale = 1.0 + 0.1 * (severity - 1)
        norm_probs = srt / (srt[0] * conf_scale)
        
        # Compute cumulative sums with regularization
        cum_prob = np.cumsum(norm_probs)
        cum_reg = cum_prob + reg_vec * (1 + 0.1 * (severity - 1))
        
        # Determine set size
        set_size = min_size  # Start with minimum size
        for j in range(min_size, min(max_size + 1, n_classes)):
            if cum_reg[j-1] <= threshold:
                set_size = j
            else:
                break
        
        prediction_sets[i, pi[:set_size]] = True
    
    return prediction_sets

def conformal_prediction_raps(
    cal_softmax: np.ndarray,
    cal_labels: np.ndarray,
    val_softmax: np.ndarray,
    val_labels: np.ndarray,
    k_reg: int = 5,
    lam_reg: float = 0.01,
    severity: int = 1,
    alpha: float = 0.1
) -> Tuple[float, float, float, np.ndarray]:
    """
    Run conformal prediction with improved severity handling
    """
    # More aggressive alpha adjustment for higher severities
    effective_alpha = alpha / (1 + 0.15 * (severity - 1))
    
    # Compute calibration scores
    cal_scores = compute_raps_scores(
        cal_softmax, cal_labels,
        k_reg=k_reg,
        lam_reg=lam_reg,
        severity=severity
    )
    
    # Find base threshold
    n = len(cal_scores)
    level = np.ceil((n + 1) * (1 - effective_alpha)) / n
    base_threshold = np.quantile(cal_scores, level, method='higher')
    
    # Apply severity-aware threshold scaling
    threshold = base_threshold * (1 + 0.1 * (severity - 1))
    
    # Generate prediction sets
    prediction_sets = get_prediction_sets_raps(
        val_softmax,
        threshold,
        k_reg=k_reg,
        lam_reg=lam_reg,
        severity=severity
    )
    
    # Calculate metrics
    coverage = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels])
    avg_set_size = np.mean(prediction_sets.sum(axis=1))
    
    return coverage, avg_set_size, threshold, prediction_sets

def compute_nonconformity_scores(softmax_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute nonconformity scores as negative log probability for true class"""
    true_class_probs = softmax_scores[np.arange(len(labels)), labels]
    return -np.log(true_class_probs)

def analyze_nonconformity_abstention(
    nonconformity_scores: np.ndarray,
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray,
    severity: int  # New parameter
) -> Dict:
    n_samples = len(true_labels)
    results = {}
    
    # Calculate class probabilities from prediction sets
    true_labels_in_set = prediction_sets[np.arange(n_samples), true_labels]
    set_sizes = np.sum(prediction_sets, axis=1)
    
    # Define severity-dependent criteria
    base_size = 1 + severity  # Expected set size increases with severity
    size_penalty = (set_sizes - base_size) / base_size  # Penalize larger sets
    
    # Ground truth: we should abstain when either:
    # 1. True label not in set
    # 2. Set size is much larger than expected for this severity
    # 3. Nonconformity score is in top quartile
    nonconf_threshold = np.percentile(nonconformity_scores, 75)
    ground_truth_abstain = (
        (~true_labels_in_set) |  # Wrong prediction
        (size_penalty > 0.5) |   # Too large set
        (nonconformity_scores > nonconf_threshold)  # High uncertainty
    )
    
    # Print diagnostic info
    print(f"\nSeverity {severity} diagnostics:")
    print(f"Base expected size: {base_size}")
    print(f"Mean size penalty: {np.mean(size_penalty):.4f}")
    print(f"Ground truth abstain rate: {np.mean(ground_truth_abstain):.4f}")
    
    for threshold in thresholds:
        model_abstains = nonconformity_scores > threshold
        
        tp = np.sum(model_abstains & ground_truth_abstain)
        fn = np.sum(~model_abstains & ground_truth_abstain)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        fp = np.sum(model_abstains & ~ground_truth_abstain)
        tn = np.sum(~model_abstains & ~ground_truth_abstain)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        results[threshold] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'tpr': tpr,
            'fpr': fpr,
            'abstention_rate': np.mean(model_abstains)
        }
        
        # Debug info for key thresholds
        if threshold in [min(thresholds), max(thresholds)]:
            print(f"\nAt threshold {threshold:.4f}:")
            print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            print(f"TPR: {tpr:.4f}, FPR: {fpr:.4f}")
    
    return results

def analyze_prediction_stats(prediction_sets: np.ndarray, true_labels: np.ndarray) -> Dict:
    """
    Analyze coverage and set size statistics for prediction sets
    
    Args:
        prediction_sets: (n_samples, n_classes) boolean array
        true_labels: (n_samples,) array of true labels
    
    Returns:
        Dictionary containing coverage and set size statistics
    """
    n_samples = len(true_labels)
    
    # Calculate coverage
    true_labels_in_set = prediction_sets[np.arange(n_samples), true_labels]
    coverage = np.mean(true_labels_in_set)
    
    # Calculate set sizes
    set_sizes = prediction_sets.sum(axis=1)
    avg_set_size = np.mean(set_sizes)
    median_set_size = np.median(set_sizes)
    max_set_size = np.max(set_sizes)
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'median_set_size': median_set_size,
        'max_set_size': max_set_size,
        'set_size_distribution': np.bincount(set_sizes)
    }

def find_abstention_threshold(
    results: Dict,
    target_coverage: float = 0.9,
    min_abstention: float = 0.01,
    fpr_penalty: float = 1.5
) -> Tuple[float, Dict]:
    # Generate more fine-grained thresholds around key regions
    thresholds = np.concatenate([
        np.linspace(0.9, 1.1, 100),  # More granular around 1.0
        np.linspace(1.1, 2.0, 50)     # Extended range for higher severities
    ])
    
    valid_thresholds = [
        t for t in results.keys()
        if (1 - results[t]['abstention_rate']) >= target_coverage * 0.95  # Slightly relaxed constraint
        and results[t]['abstention_rate'] >= min_abstention
    ]
    
    if valid_thresholds:
        best_threshold = max(
            valid_thresholds,
            key=lambda t: (results[t]['tpr'] - fpr_penalty * results[t]['fpr']) * 
                         (1 - abs(results[t]['abstention_rate'] - min_abstention))  # Penalty for extreme abstention
        )
    else:
        best_threshold = min(
            results.keys(),
            key=lambda t: abs((1 - results[t]['abstention_rate']) - target_coverage)
        )
    
    return best_threshold, results[best_threshold]

def analyze_abstention_auc(tpr_rates: np.ndarray, fpr_rates: np.ndarray) -> float:
    """
    Calculate area under the TPR vs FPR curve without range normalization
    """
    # Sort by FPR rates
    sort_idx = np.argsort(fpr_rates)
    fpr_sorted = fpr_rates[sort_idx]
    tpr_sorted = tpr_rates[sort_idx]
    
    # Remove any duplicate FPR values by averaging corresponding TPR values
    unique_fprs, unique_indices = np.unique(fpr_sorted, return_index=True)
    unique_tprs = np.array([np.mean(tpr_sorted[fpr_sorted == fpr]) for fpr in unique_fprs])
    
    # Add endpoints if needed
    if unique_fprs[0] > 0:
        unique_fprs = np.concatenate([[0], unique_fprs])
        unique_tprs = np.concatenate([[0], unique_tprs])
    if unique_fprs[-1] < 1:
        unique_fprs = np.concatenate([unique_fprs, [1]])
        unique_tprs = np.concatenate([unique_tprs, [1]])
    
    # Calculate AUC using trapezoidal rule without normalization
    auc = np.trapz(unique_tprs, unique_fprs)
    
    return auc


def main():
    # Initialize configuration and logging
    Config.initialize()
    logger = setup_logging('raps_nonconformity_abstention')
    
    device = torch.device(f'cuda:{Config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data and model
    calibration_loader = create_dataloader(Config.ROOT_DIR, 'calibration')
    test_dataset = ModelNet40Dataset(root_dir=Config.ROOT_DIR, split='test')
    
    model = PointTransformerV2(
        num_classes=Config.MODEL['num_classes'],
        in_channels=Config.MODEL['in_channels']
    ).to(device)
    
    checkpoint = torch.load('/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded trained model")
    
    # Get calibration predictions
    cal_softmax, cal_labels = get_softmax_predictions(model, calibration_loader, device)
    
    # Analysis parameters
    severity_levels = [1, 2, 3, 4, 5]
    thresholds = np.concatenate([
        np.linspace(0.0, 0.1, 30),
        np.linspace(0.1, 0.5, 40),
        np.linspace(0.5, 1.0, 30)
    ])

    # Create plot directories
    plot_dirs = create_plot_dirs('plots')

    # Storage for metrics across severities
    metrics_data = {
        'severities': [],
        'coverages': [],
        'set_sizes': [],
        'abstention_rates': [],
        'results_by_severity': {},
        'set_sizes_by_severity': {}
    }

    for severity in severity_levels:
        logger.info(f"Analyzing severity level {severity}")
        
        # Create corrupted dataset
        corrupted_dataset = CorruptedModelNet40Dataset(
            base_dataset=test_dataset,
            corruption_type=OcclusionCorruption,
            severity=severity
        )
        
        corrupted_loader = DataLoader(
            corrupted_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Get predictions
        val_softmax, val_labels = get_softmax_predictions(model, corrupted_loader, device)
        
        # Run RAPS with size control
        coverage, avg_set_size, threshold, prediction_sets = conformal_prediction_raps(
            cal_softmax, cal_labels,
            val_softmax, val_labels,
            k_reg=5,
            lam_reg=0.01,
            severity=severity,
            alpha=0.1
        )
        
        # Compute nonconformity scores for validation set
        val_nonconformity = compute_nonconformity_scores(val_softmax, val_labels)
        
        # First analyze prediction sets
        prediction_stats = analyze_prediction_stats(prediction_sets, val_labels)
        
        # Then analyze abstention
        results = analyze_nonconformity_abstention(
            val_nonconformity,
            prediction_sets,
            val_labels,
            thresholds,
            severity=severity
        )
        
        # Find optimal threshold with constraints from results
        best_threshold, best_metrics = find_abstention_threshold(
            results,  # Now passing the results dictionary directly
            target_coverage=0.9,
            min_abstention=0.01
        )
        
        # Calculate AUC for current severity
        current_tpr_rates = np.array([m['tpr'] for m in results.values()])
        current_fpr_rates = np.array([m['fpr'] for m in results.values()])
        current_auc = analyze_abstention_auc(current_tpr_rates, current_fpr_rates)
        
        # Store metrics for overall analysis
        metrics_data['severities'].append(severity)
        metrics_data['coverages'].append(coverage)
        metrics_data['set_sizes'].append(avg_set_size)
        metrics_data['abstention_rates'].append(best_metrics['abstention_rate'])
        
        # Store results and set sizes for aggregate plots
        metrics_data['results_by_severity'][severity] = {
            'abstention_results': results,
            'auc': current_auc  # Store the AUC for this severity
        }
        metrics_data['set_sizes_by_severity'][severity] = prediction_sets.sum(axis=1)
        
        # Individual severity plots
        plot_nonconformity_analysis(results, severity, plot_dirs['abstention'])
        
        # Log results for current severity
        logger.info(f"\nRAPS Results (Severity {severity}):")
        logger.info(f"Coverage: {coverage:.4f}")
        logger.info(f"Average Set Size: {avg_set_size:.4f}")
        logger.info(f"RAPS Threshold: {threshold:.4f}")
        
        logger.info(f"\nAbstention Results:")
        logger.info(f"Best threshold: {best_threshold:.4f}")
        logger.info(f"TPR: {best_metrics['tpr']:.4f}")
        logger.info(f"FPR: {best_metrics['fpr']:.4f}")
        logger.info(f"Abstention Rate: {best_metrics['abstention_rate']:.4f}")
        logger.info(f"Abstention AUC: {current_auc:.4f}")

    # Generate aggregate plots
    plot_metrics_vs_severity(
        severities=metrics_data['severities'],
        coverages=metrics_data['coverages'],
        set_sizes=metrics_data['set_sizes'],
        abstention_rates=metrics_data['abstention_rates'],
        save_dir=plot_dirs['metrics']
    )
    
    plot_roc_curves(
        metrics_data['results_by_severity'], 
        save_dir=plot_dirs['roc']
    )
    
    plot_set_size_distribution(
        metrics_data['set_sizes_by_severity'], 
        save_dir=plot_dirs['set_sizes']
    )

if __name__ == '__main__':
    main()