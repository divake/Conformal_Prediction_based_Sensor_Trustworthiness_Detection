#src/abstention_analysis_nonconformity.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import binom
from conformal_prediction import conformal_prediction, get_softmax_predictions, create_dataloader
from data.corrupted_dataset import CorruptedModelNet40Dataset
from data.corruptions import OcclusionCorruption
from config import Config
from main import setup_logging
from data.dataset import ModelNet40Dataset
from models.point_transformer_v2 import PointTransformerV2
from torch.utils.data import DataLoader
import logging



def find_qhat_for_target_coverage(
    val_softmax: np.ndarray,
    val_labels: np.ndarray,
    target_coverage: float = 0.9,
    k_reg: int = 5,
    lam_reg: float = 0.01
) -> float:
    scores = []
    for i in range(len(val_labels)):
        pi = val_softmax[i].argsort()[::-1]
        srt = val_softmax[i][pi]
        L = np.where(pi == val_labels[i])[0][0]
        
        # Increase set sizes for lower confidence predictions
        score = srt[L] / (1 - target_coverage)
        scores.append(score)
    
    # Use a higher quantile to ensure coverage
    return np.quantile(scores, 0.95)

def setup_logging(logger_name: str = 'abstention_analysis') -> logging.Logger:
    """Set up logging configuration
    
    Args:
        logger_name (str): Name for the logger (default: 'abstention_analysis')
    
    Returns:
        logging.Logger: Logger instance
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "abstention_analysis.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(logger_name)


def compute_nonconformity_scores(softmax_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute nonconformity scores as negative log of softmax probability for true class
    
    Args:
        softmax_scores: (n_samples, n_classes) array of softmax probabilities
        labels: (n_samples,) array of true labels
    
    Returns:
        Array of nonconformity scores
    """
    true_class_probs = softmax_scores[np.arange(len(labels)), labels]
    return -np.log(true_class_probs + 1e-7)  # Add small epsilon for numerical stability

def find_abstention_threshold(
    cal_nonconformity: np.ndarray,
    target_error_rate: float = 0.1
) -> float:
    """
    Find optimal abstention threshold using calibration data
    
    Args:
        cal_nonconformity: Array of nonconformity scores from calibration set
        target_error_rate: Target error rate (default: 0.1)
    
    Returns:
        Optimal threshold for abstention
    """
    sorted_scores = np.sort(cal_nonconformity)
    n = len(sorted_scores)
    index = int(np.ceil((1 - target_error_rate) * (n + 1))) - 1
    return sorted_scores[index]

def analyze_nonconformity_abstention(
    nonconformity_scores: np.ndarray,
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray
) -> Dict:
    """
    Analyze abstention performance using nonconformity scores
    
    Args:
        nonconformity_scores: Array of nonconformity scores
        prediction_sets: (n_samples, n_classes) boolean array of prediction sets
        true_labels: (n_samples,) array of true labels
        thresholds: Array of threshold values to try
    """
    results = {}
    n_samples = len(true_labels)
    
    # Check if true labels are in prediction sets
    true_labels_in_set = prediction_sets[np.arange(n_samples), true_labels]
    
    for threshold in thresholds:
        # Abstain if nonconformity score is high
        abstained = nonconformity_scores > threshold
        
        # Calculate TPA and FPA
        tpa = np.sum(abstained & ~true_labels_in_set)
        fpa = np.sum(abstained & true_labels_in_set)
        
        total_abstentions = np.sum(abstained)
        
        if total_abstentions > 0:
            tpa_rate = tpa / total_abstentions
            fpa_rate = fpa / total_abstentions
        else:
            tpa_rate = 0.0
            fpa_rate = 0.0
            
        abstention_rate = total_abstentions / n_samples
        
        results[threshold] = {
            'tpa': tpa,
            'fpa': fpa,
            'tpa_rate': tpa_rate,
            'fpa_rate': fpa_rate,
            'abstention_rate': abstention_rate
        }
    
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

def plot_set_size_distribution(
    set_size_stats: Dict,
    severity: int,
    save_dir: str = 'plots/prediction_stats'
):
    """Plot set size distribution and statistics"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dist = set_size_stats['set_size_distribution']
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(dist)), dist)
    plt.axvline(set_size_stats['avg_set_size'], color='r', linestyle='--', 
                label=f'Mean: {set_size_stats["avg_set_size"]:.2f}')
    plt.axvline(set_size_stats['median_set_size'], color='g', linestyle='--', 
                label=f'Median: {set_size_stats["median_set_size"]:.2f}')
    
    plt.xlabel('Set Size')
    plt.ylabel('Count')
    plt.title(f'Prediction Set Size Distribution (Severity {severity})\n' +
              f'Coverage: {set_size_stats["coverage"]:.2%}')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_dir / f'set_size_distribution_severity_{severity}.png')
    plt.close()

def analyze_abstention_auc(tpa_rates: np.ndarray, fpa_rates: np.ndarray) -> float:
    """Calculate area under the TPA vs FPA curve"""
    # Sort by FPA rates for proper AUC calculation
    sort_idx = np.argsort(fpa_rates)
    fpa_sorted = fpa_rates[sort_idx]
    tpa_sorted = tpa_rates[sort_idx]
    
    return np.trapz(tpa_sorted, fpa_sorted)

def find_optimal_threshold_with_constraints(
    results: Dict,
    min_coverage: float = 0.9,
    min_abstention: float = 0.01
) -> Tuple[float, Dict]:
    """
    Find optimal threshold considering coverage and abstention constraints
    
    Args:
        results: Dictionary of results for different thresholds
        min_coverage: Minimum required coverage (default: 0.9)
        min_abstention: Minimum required abstention rate (default: 0.01)
        
    Returns:
        Tuple of (optimal threshold, metrics at that threshold)
    """
    best_score = -float('inf')
    best_threshold = None
    best_metrics = None
    
    for threshold, metrics in results.items():
        # Check constraints
        effective_coverage = 1 - metrics['abstention_rate']
        if (effective_coverage >= min_coverage and 
            metrics['abstention_rate'] >= min_abstention):
            
            # Score based on TPA-FPA difference and abstention rate
            score = (metrics['tpa_rate'] - metrics['fpa_rate']) 
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
    
    # If no threshold satisfies constraints, take the one with best TPA-FPA difference
    if best_threshold is None:
        for threshold, metrics in results.items():
            score = (metrics['tpa_rate'] - metrics['fpa_rate'])
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
    
    return best_threshold, best_metrics



def plot_nonconformity_analysis(
    results: Dict,
    severity: int,
    save_dir: str = 'plots/nonconformity_abstention'
) -> None:
    """Plot nonconformity-based abstention analysis results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = sorted(list(results.keys()))
    tpa_rates = [results[t]['tpa_rate'] for t in thresholds]
    fpa_rates = [results[t]['fpa_rate'] for t in thresholds]
    abstention_rates = [results[t]['abstention_rate'] for t in thresholds]
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Nonconformity-based Abstention Analysis (Severity {severity})', fontsize=14)
    
    # 1. TPA vs FPA curve
    axs[0, 0].plot(fpa_rates, tpa_rates, 'o-')
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axs[0, 0].set_xlabel('False Positive Abstention Rate')
    axs[0, 0].set_ylabel('True Positive Abstention Rate')
    axs[0, 0].set_title('TPA vs FPA')
    axs[0, 0].grid(True)
    
    # 2. Abstention rate vs threshold
    axs[0, 1].plot(thresholds, abstention_rates, 'o-')
    axs[0, 1].set_xlabel('Nonconformity Threshold')
    axs[0, 1].set_ylabel('Abstention Rate')
    axs[0, 1].set_title('Abstention Rate vs Threshold')
    axs[0, 1].grid(True)
    
    # 3. TPA and FPA rates vs threshold
    axs[1, 0].plot(thresholds, tpa_rates, 'o-', label='TPA Rate')
    axs[1, 0].plot(thresholds, fpa_rates, 'o-', label='FPA Rate')
    axs[1, 0].set_xlabel('Nonconformity Threshold')
    axs[1, 0].set_ylabel('Rate')
    axs[1, 0].set_title('TPA and FPA Rates vs Threshold')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # 4. TPA-FPA difference vs threshold
    diff_rates = [tpa - fpa for tpa, fpa in zip(tpa_rates, fpa_rates)]
    axs[1, 1].plot(thresholds, diff_rates, 'o-')
    axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axs[1, 1].set_xlabel('Nonconformity Threshold')
    axs[1, 1].set_ylabel('TPA - FPA')
    axs[1, 1].set_title('TPA-FPA Difference vs Threshold')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'nonconformity_abstention_analysis_severity_{severity}.png')
    plt.close()

def main():
    # Initialize configuration and logging
    Config.initialize()
    logger = setup_logging('nonconformity_abstention')
    
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
    
    # Analyze each severity level
    severity_levels = [1, 2, 3, 4, 5]
    thresholds = np.linspace(0.1, 5.0, 50)  # Range for -log(softmax)
    
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
        
        # Run conformal prediction
        coverage, avg_set_size, qhat, prediction_sets = conformal_prediction(
            cal_softmax, cal_labels,
            val_softmax, val_labels,
            alpha=0.1,
            k_reg=5,
            lam_reg=0.01
        )
        
        # Compute nonconformity scores for validation set
        val_nonconformity = compute_nonconformity_scores(val_softmax, val_labels)
        
        # Analyze abstention
        results = analyze_nonconformity_abstention(
            val_nonconformity,
            prediction_sets,
            val_labels,
            thresholds
        )
        
        # Find optimal threshold with constraints
        best_threshold, best_metrics = find_optimal_threshold_with_constraints(
            results,
            min_coverage=0.9,
            min_abstention=0.01
        )
        
        if best_threshold is None:
            logger.warning("No threshold found satisfying constraints. Using fallback selection.")
            # Use simple threshold selection as fallback
            best_threshold = min(results.keys(), 
                               key=lambda t: (results[t]['tpa_rate'] < results[t]['fpa_rate'], 
                                            -results[t]['tpa_rate'] + results[t]['fpa_rate']))
            best_metrics = results[best_threshold]
        
        # Plot results
        plot_nonconformity_analysis(results, severity)
        
        # Analyze prediction sets
        prediction_stats = analyze_prediction_stats(prediction_sets, val_labels)
        plot_set_size_distribution(prediction_stats, severity)
        
        # Log all results
        logger.info(f"\nNonconformity Abstention Results (Severity {severity}):")
        logger.info(f"qhat: {qhat:.4f}")
        logger.info(f"Best threshold: {best_threshold:.4f}")
        logger.info(f"TPA Rate: {best_metrics['tpa_rate']:.4f}")
        logger.info(f"FPA Rate: {best_metrics['fpa_rate']:.4f}")
        logger.info(f"Abstention Rate: {best_metrics['abstention_rate']:.4f}")

        logger.info(f"\nPrediction Set Statistics (Severity {severity}):")
        logger.info(f"Coverage: {prediction_stats['coverage']:.4f}")
        logger.info(f"Average Set Size: {prediction_stats['avg_set_size']:.4f}")
        logger.info(f"Median Set Size: {prediction_stats['median_set_size']:.4f}")
        logger.info(f"Maximum Set Size: {prediction_stats['max_set_size']}")
        
        # Calculate AUC for abstention
        tpa_rates = np.array([m['tpa_rate'] for m in results.values()])
        fpa_rates = np.array([m['fpa_rate'] for m in results.values()])
        auc = analyze_abstention_auc(tpa_rates, fpa_rates)
        logger.info(f"Abstention AUC: {auc:.4f}")

if __name__ == '__main__':
    main()