#src/abstention_analysis_hybrid.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import brentq
from scipy.stats import binom
from conformal_prediction import conformal_prediction, get_softmax_predictions
from data.corrupted_dataset import CorruptedModelNet40Dataset
from data.corruptions import OcclusionCorruption
from conformal_prediction import conformal_prediction, get_softmax_predictions, create_dataloader
from evaluation.corruption_analysis import evaluate_corruption_robustness
from config import Config
from main import setup_logging
from data.dataset import ModelNet40Dataset
from models.point_transformer_v2 import PointTransformerV2
from torch.utils.data import DataLoader, Dataset
import logging


def setup_logging(logger_name='abstention_analysis'):
    """Set up logging configuration
    
    Args:
        logger_name: Name for the logger (default: 'abstention_analysis')
    
    Returns:
        Logger instance
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

def analyze_hybrid_abstention(
    softmax_scores: np.ndarray,
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
    lambda_thresholds: np.ndarray = np.linspace(0, 1, 100),
    size_thresholds: List[int] = [1, 2, 3]
) -> Dict:
    """
    Analyze abstention using both softmax scores and set sizes
    
    Args:
        softmax_scores: (n_samples, n_classes) array of softmax probabilities
        prediction_sets: (n_samples, n_classes) boolean array of prediction sets
        true_labels: (n_samples,) array of true labels
        lambda_thresholds: Array of softmax threshold values
        size_thresholds: List of set size thresholds
    
    Returns:
        Dictionary containing TPA rates, FPA rates for different thresholds
    """
    results = {}
    n_samples = len(true_labels)
    
    # Get maximum softmax scores and set sizes
    max_softmax = np.max(softmax_scores, axis=1)
    set_sizes = prediction_sets.sum(axis=1)
    
    # Check if true labels are in prediction sets
    true_labels_in_set = prediction_sets[np.arange(n_samples), true_labels]
    
    for lambda_t in lambda_thresholds:
        for size_t in size_thresholds:
            # Hybrid abstention criterion
            # Abstain if: max_softmax < lambda_t OR set_size > size_t
            abstained = (max_softmax < lambda_t) | (set_sizes > size_t)
            
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
            
            results[(lambda_t, size_t)] = {
                'tpa': tpa,
                'fpa': fpa,
                'tpa_rate': tpa_rate,
                'fpa_rate': fpa_rate,
                'abstention_rate': abstention_rate,
                'total_abstentions': total_abstentions
            }
    
    return results

def find_optimal_thresholds(results: Dict) -> Tuple:
    """Find thresholds that maximize TPA-FPA difference with sufficient abstentions"""
    best_diff = -float('inf')
    best_thresholds = None
    
    for thresholds, metrics in results.items():
        # Only consider points with enough abstentions (e.g., >1% of data)
        if metrics['abstention_rate'] >= 0.01:
            diff = metrics['tpa_rate'] - metrics['fpa_rate']
            if diff > best_diff:
                best_diff = diff
                best_thresholds = thresholds
    
    return best_thresholds

def plot_hybrid_abstention_analysis(
    results: Dict,
    severity: int,
    save_dir: str = 'plots/hybrid_abstention'
) -> None:
    """Plot hybrid abstention analysis results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract unique thresholds
    lambda_thresholds = sorted(list(set(t[0] for t in results.keys())))
    size_thresholds = sorted(list(set(t[1] for t in results.keys())))
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Hybrid Abstention Analysis (Severity {severity})', fontsize=14)
    
    # 1. TPA vs FPA for different size thresholds
    for size_t in size_thresholds:
        tpa_rates = []
        fpa_rates = []
        for lambda_t in lambda_thresholds:
            metrics = results[(lambda_t, size_t)]
            tpa_rates.append(metrics['tpa_rate'])
            fpa_rates.append(metrics['fpa_rate'])
        
        axs[0, 0].plot(fpa_rates, tpa_rates, 'o-', label=f'Size Threshold={size_t}')
    
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # diagonal line
    axs[0, 0].set_xlabel('False Positive Abstention Rate')
    axs[0, 0].set_ylabel('True Positive Abstention Rate')
    axs[0, 0].set_title('TPA vs FPA (ROC-like curve)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 2. Abstention rates vs lambda for different size thresholds
    for size_t in size_thresholds:
        abstention_rates = []
        for lambda_t in lambda_thresholds:
            abstention_rates.append(results[(lambda_t, size_t)]['abstention_rate'])
        
        axs[0, 1].plot(lambda_thresholds, abstention_rates, 
                       label=f'Size Threshold={size_t}')
    
    axs[0, 1].set_xlabel('Lambda Threshold')
    axs[0, 1].set_ylabel('Abstention Rate')
    axs[0, 1].set_title('Abstention Rate vs Lambda')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # 3. TPA-FPA difference heatmap
    diff_matrix = np.zeros((len(size_thresholds), len(lambda_thresholds)))
    for i, size_t in enumerate(size_thresholds):
        for j, lambda_t in enumerate(lambda_thresholds):
            metrics = results[(lambda_t, size_t)]
            diff_matrix[i, j] = metrics['tpa_rate'] - metrics['fpa_rate']
    
    im = axs[1, 0].imshow(diff_matrix, aspect='auto', cmap='RdBu', 
                         extent=[min(lambda_thresholds), max(lambda_thresholds),
                                min(size_thresholds)-0.5, max(size_thresholds)+0.5])
    plt.colorbar(im, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Lambda Threshold')
    axs[1, 0].set_ylabel('Size Threshold')
    axs[1, 0].set_title('TPA-FPA Difference')
    
    # 4. Abstention rate heatmap
    rate_matrix = np.zeros((len(size_thresholds), len(lambda_thresholds)))
    for i, size_t in enumerate(size_thresholds):
        for j, lambda_t in enumerate(lambda_thresholds):
            rate_matrix[i, j] = results[(lambda_t, size_t)]['abstention_rate']
    
    im = axs[1, 1].imshow(rate_matrix, aspect='auto', cmap='viridis',
                         extent=[min(lambda_thresholds), max(lambda_thresholds),
                                min(size_thresholds)-0.5, max(size_thresholds)+0.5])
    plt.colorbar(im, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Lambda Threshold')
    axs[1, 1].set_ylabel('Size Threshold')
    axs[1, 1].set_title('Abstention Rate')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'hybrid_abstention_analysis_severity_{severity}.png')
    plt.close()

def main():
    # Initialize configuration and logging
    Config.initialize()
    logger = setup_logging('hybrid_abstention')
    
    device = torch.device(f'cuda:{Config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load calibration data and model
    calibration_loader = create_dataloader(Config.ROOT_DIR, 'calibration')
    test_dataset = ModelNet40Dataset(root_dir=Config.ROOT_DIR, split='test')
    
    # Initialize model with correct parameters
    model = PointTransformerV2(
        num_classes=Config.MODEL['num_classes'],
        in_channels=Config.MODEL['in_channels']
    ).to(device)
    
    # Load model checkpoint
    checkpoint = torch.load('/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded trained model")
    
    # Get calibration predictions
    cal_softmax, cal_labels = get_softmax_predictions(model, calibration_loader, device)
    
    # Parameters for hybrid analysis
    lambda_thresholds = np.linspace(0.1, 0.9, 50)  # softmax thresholds
    size_thresholds = [1, 2, 3]  # set size thresholds
    severity_levels = [1, 2, 3, 4, 5]
    
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
        
        # Get predictions and run conformal prediction
        val_softmax, val_labels = get_softmax_predictions(model, corrupted_loader, device)
        coverage, avg_set_size, qhat, prediction_sets = conformal_prediction(
            cal_softmax, cal_labels,
            val_softmax, val_labels,
            alpha=0.1,
            k_reg=5,
            lam_reg=0.01
        )
        
        # Analyze hybrid abstention
        results = analyze_hybrid_abstention(
            val_softmax, prediction_sets, val_labels,
            lambda_thresholds, size_thresholds
        )
        
        # Find optimal thresholds
        best_lambda, best_size = find_optimal_thresholds(results)
        
        # Plot results
        plot_hybrid_abstention_analysis(results, severity)
        
        # Log results
        logger.info(f"\nHybrid Abstention Analysis Results (Severity {severity}):")
        logger.info(f"Optimal thresholds - Lambda: {best_lambda:.3f}, Size: {best_size}")
        
        metrics = results[(best_lambda, best_size)]
        logger.info(f"At optimal thresholds:")
        logger.info(f"  TPA Rate: {metrics['tpa_rate']:.4f}")
        logger.info(f"  FPA Rate: {metrics['fpa_rate']:.4f}")
        logger.info(f"  Abstention Rate: {metrics['abstention_rate']:.4f}")

if __name__ == '__main__':
    main()