#src/abstention_analysis.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from data.corruptions import OcclusionCorruption
from data.corrupted_dataset import CorruptedModelNet40Dataset
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


def analyze_abstention(
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
    thresholds: List[int] = [1, 2, 3, 4, 5]
) -> Dict:
    """
    Analyze abstention performance for different set size thresholds
    
    Args:
        prediction_sets: (n_samples, n_classes) boolean array of prediction sets
        true_labels: (n_samples,) array of true labels
        thresholds: List of set size thresholds for abstention
    
    Returns:
        Dictionary containing TPA rates, FPA rates, and abstention rates
    """
    results = {}
    n_samples = len(true_labels)
    
    for threshold in thresholds:
        # Get set sizes and abstention decisions
        set_sizes = prediction_sets.sum(axis=1)
        abstained = set_sizes > threshold
        
        # Count TPA and FPA
        true_labels_in_set = prediction_sets[np.arange(n_samples), true_labels]
        tpa = np.sum(abstained & ~true_labels_in_set)  # Abstain and true label not in set
        fpa = np.sum(abstained & true_labels_in_set)   # Abstain but true label in set
        
        total_abstentions = np.sum(abstained)
        
        # Calculate rates
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
            'abstention_rate': abstention_rate,
            'total_abstentions': total_abstentions
        }
    
    return results

def plot_abstention_analysis(
    results: Dict,
    severity: int,
    save_dir: str = 'plots/abstention'
) -> None:
    """Plot abstention analysis results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = list(results.keys())
    tpa_rates = [results[t]['tpa_rate'] for t in thresholds]
    fpa_rates = [results[t]['fpa_rate'] for t in thresholds]
    abstention_rates = [results[t]['abstention_rate'] for t in thresholds]
    
    # 1. ROC-like curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpa_rates, tpa_rates, 'o-')
    for i, t in enumerate(thresholds):
        plt.annotate(f'T={t}', (fpa_rates[i], tpa_rates[i]))
    plt.xlabel('False Positive Abstention Rate')
    plt.ylabel('True Positive Abstention Rate')
    plt.title(f'Abstention ROC (Severity {severity})')
    plt.grid(True)
    
    # 2. Abstention rate vs threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, abstention_rates, 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Abstention Rate')
    plt.title(f'Abstention Rate vs Threshold (Severity {severity})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'abstention_analysis_severity_{severity}.png')
    plt.close()
    
    # 3. TPA and FPA counts vs threshold
    plt.figure(figsize=(8, 6))
    tpa_counts = [results[t]['tpa'] for t in thresholds]
    fpa_counts = [results[t]['fpa'] for t in thresholds]
    
    plt.plot(thresholds, tpa_counts, 'o-', label='TPA')
    plt.plot(thresholds, fpa_counts, 'o-', label='FPA')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title(f'TPA and FPA Counts (Severity {severity})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'abstention_counts_severity_{severity}.png')
    plt.close()

def main():
    # Initialize configuration and logging
    Config.initialize()
    logger = setup_logging('abstention_analysis')
    
    device = torch.device(f'cuda:{Config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load calibration data and model
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
    thresholds = [1, 2, 3, 4, 5]
    
    for severity in severity_levels:
        logger.info(f"\nAnalyzing severity level {severity}")
        
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
        
        # Analyze abstention
        results = analyze_abstention(prediction_sets, val_labels, thresholds)
        
        # Plot results
        plot_abstention_analysis(results, severity)
        
        # Log results
        logger.info(f"\nAbstention Analysis Results (Severity {severity}):")
        for threshold in thresholds:
            r = results[threshold]
            logger.info(f"\nThreshold {threshold}:")
            logger.info(f"  TPA Rate: {r['tpa_rate']:.4f}")
            logger.info(f"  FPA Rate: {r['fpa_rate']:.4f}")
            logger.info(f"  Abstention Rate: {r['abstention_rate']:.4f}")

if __name__ == '__main__':
    main()