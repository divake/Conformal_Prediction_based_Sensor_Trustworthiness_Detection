#src/abstention_analysis_nonconformity.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import binom
from conformal_prediction import conformal_prediction, get_softmax_predictions, create_dataloader
from data.corruptions import CorruptedModelNet40Dataset, OcclusionCorruption, RainCorruption  # Updated import
from config import Config
from data.dataset import ModelNet40Dataset
from models.point_transformer_v2 import PointTransformerV2
from torch.utils.data import DataLoader
import logging
from utils.visualization import create_plot_dirs, plot_nonconformity_analysis, plot_metrics_vs_severity, plot_roc_curves, plot_set_size_distribution
from sklearn.neighbors import KDTree

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

def find_prediction_set_threshold(cal_scores: np.ndarray, alpha: float = 0.1) -> float:
    """Find prediction set threshold with proper finite sample correction"""
    n = len(cal_scores)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    return qhat

def compute_uncertainty_metrics(
    softmax_scores: np.ndarray, 
    point_features: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute uncertainty metrics from softmax scores and optionally point cloud features.
    Args:
        softmax_scores: Classification probabilities
        point_features: Optional point cloud features (N, C) where C includes xyz
    """
    entropy = -np.sum(softmax_scores * np.log(softmax_scores + 1e-7), axis=1)
    max_probs = np.max(softmax_scores, axis=1)
    margin = np.sort(softmax_scores, axis=1)[:, -1] - np.sort(softmax_scores, axis=1)[:, -2]
    normalized_entropy = entropy / np.log(softmax_scores.shape[1])
    
    metrics = {
        'entropy': entropy,
        'confidence': max_probs,
        'margin': margin,
        'normalized_entropy': normalized_entropy
    }
    
    if point_features is not None:
        # Add point cloud specific metrics if features are provided
        xyz = point_features[:, :3]  # Assuming first 3 dims are xyz
        # Compute local density variation as geometric uncertainty
        kdtree = KDTree(xyz)
        dists, _ = kdtree.query(xyz, k=6)  # Get 5 nearest neighbors
        local_density = np.mean(dists[:, 1:], axis=1)  # Skip first (self)
        density_variation = np.std(local_density) / np.mean(local_density)
        
        metrics['geometric_uncertainty'] = density_variation
        
        # Adjust normalized entropy based on geometric features
        metrics['combined_uncertainty'] = (
            0.7 * normalized_entropy + 
            0.3 * (density_variation / np.max(density_variation))
        )
    
    return metrics

def compute_raps_scores(
    softmax_scores: np.ndarray,
    labels: np.ndarray,
    k_reg: int = 5,
    lam_reg: float = 0.02
) -> np.ndarray:
    """Compute RAPS scores with proper regularization"""
    n_samples = len(labels)
    n_classes = softmax_scores.shape[1]
    
    # Create regularization vector
    reg_vec = np.array([0]*k_reg + [lam_reg]*(n_classes-k_reg))[None,:]
    
    # Sort probabilities and get true class positions
    sort_idx = softmax_scores.argsort(axis=1)[:,::-1]
    sorted_probs = np.take_along_axis(softmax_scores, sort_idx, axis=1)
    true_class_pos = np.where(sort_idx == labels[:,None])[1]
    
    # Add regularization and compute cumulative sums
    sorted_reg = sorted_probs + reg_vec
    cumsum_reg = sorted_reg.cumsum(axis=1)
    
    # Compute scores with randomization
    rand_terms = np.random.rand(n_samples) * sorted_reg[np.arange(n_samples), true_class_pos]
    scores = cumsum_reg[np.arange(n_samples), true_class_pos] - rand_terms
    
    return scores

def get_prediction_sets_raps(
    softmax_scores: np.ndarray,
    threshold: float,
    k_reg: int = 5,
    lam_reg: float = 0.01
) -> np.ndarray:
    """Generate prediction sets with proper conformal guarantee"""
    n_samples, n_classes = softmax_scores.shape
    
    # Create regularization vector
    reg_vec = np.array([0]*k_reg + [lam_reg]*(n_classes-k_reg))[None,:]
    
    # Sort probabilities
    sort_idx = softmax_scores.argsort(axis=1)[:,::-1]
    sorted_probs = np.take_along_axis(softmax_scores, sort_idx, axis=1)
    
    # Add regularization and compute cumulative sums
    sorted_reg = sorted_probs + reg_vec
    cumsum_reg = sorted_reg.cumsum(axis=1)
    
    # Generate random terms and compute indicators
    rand_terms = np.random.rand(n_samples, 1) * sorted_reg
    indicators = (cumsum_reg - rand_terms) <= threshold
    
    # Map back to original class order
    prediction_sets = np.take_along_axis(indicators, sort_idx.argsort(axis=1), axis=1)
    
    return prediction_sets

def conformal_prediction_raps(
    cal_softmax: np.ndarray,
    cal_labels: np.ndarray,
    val_softmax: np.ndarray,
    val_labels: np.ndarray,
    k_reg: int = 5,
    lam_reg: float = 0.01,
    alpha: float = 0.1
) -> Tuple[float, float, float, np.ndarray]:
    """Run conformal prediction with exact coverage guarantee"""
    # Compute calibration scores
    cal_scores = compute_raps_scores(cal_softmax, cal_labels, k_reg, lam_reg)
    
    # Find threshold using proper quantile
    n = len(cal_scores)
    threshold = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    
    # Generate prediction sets
    prediction_sets = get_prediction_sets_raps(val_softmax, threshold, k_reg, lam_reg)
    
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
    softmax_scores: np.ndarray,
    point_features: np.ndarray = None
) -> Dict:
    """Analyze abstention with balanced criteria"""
    n_samples = len(true_labels)
    results = {}
    
    # Calculate basic metrics
    true_labels_in_set = prediction_sets[np.arange(n_samples), true_labels]
    set_sizes = np.sum(prediction_sets, axis=1)
    
    # Define simpler abstention criteria
    wrong_predictions = ~true_labels_in_set
    large_sets = set_sizes > np.median(set_sizes)
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        model_abstains = nonconformity_scores > threshold
        
        tp = np.sum(model_abstains & wrong_predictions)
        fp = np.sum(model_abstains & ~wrong_predictions)
        tn = np.sum(~model_abstains & ~wrong_predictions)
        fn = np.sum(~model_abstains & wrong_predictions)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        results[threshold] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'abstention_rate': np.mean(model_abstains)
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
    """Calculate AUC with proper interpolation"""
    # Ensure arrays are numpy arrays
    tpr_rates = np.array(tpr_rates)
    fpr_rates = np.array(fpr_rates)
    
    # Sort by FPR
    sort_idx = np.argsort(fpr_rates)
    fpr_rates = fpr_rates[sort_idx]
    tpr_rates = tpr_rates[sort_idx]
    
    # Remove duplicates while keeping the max TPR for each FPR
    unique_fprs, unique_idx = np.unique(fpr_rates, return_index=True)
    tpr_max = np.maximum.accumulate(tpr_rates)
    unique_tprs = tpr_max[unique_idx]
    
    # Add endpoints if needed
    if unique_fprs[0] > 0:
        unique_fprs = np.r_[0, unique_fprs]
        unique_tprs = np.r_[0, unique_tprs]
    if unique_fprs[-1] < 1:
        unique_fprs = np.r_[unique_fprs, 1]
        unique_tprs = np.r_[unique_tprs, 1]
    
    # Calculate AUC using trapezoidal rule
    return np.trapz(unique_tprs, unique_fprs)

def compute_conformal_scores(
    softmax_scores: np.ndarray, 
    labels: np.ndarray, 
    k_reg: int = 5, 
    lam_reg: float = 0.02
) -> np.ndarray:
    """Compute conformal scores using vision approach"""
    n_samples, n_classes = softmax_scores.shape
    reg_vec = np.array([0]*k_reg + [lam_reg]*(n_classes-k_reg))[None,:]
    
    sort_idx = np.argsort(softmax_scores, axis=1)[:,::-1]
    sorted_probs = np.take_along_axis(softmax_scores, sort_idx, axis=1)
    sorted_reg = sorted_probs + reg_vec
    
    true_label_pos = np.where(sort_idx == labels[:,None])[1]
    rand_terms = np.random.rand(n_samples) * sorted_reg[np.arange(n_samples), true_label_pos]
    scores = sorted_reg.cumsum(axis=1)[np.arange(n_samples), true_label_pos] - rand_terms
    
    return scores

def get_prediction_sets(*args, **kwargs):
    raise DeprecationWarning("Use get_prediction_sets_raps instead")

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
    corruptions = {
        'occlusion': OcclusionCorruption,
        'rain': RainCorruption
    }
    
    severity_levels = [1, 2, 3, 4, 5]
    alpha = 0.1  # Target 90% coverage
    k_reg = 5
    lam_reg = 0.01
    
    # Define abstention thresholds - combine linear and log space for better coverage
    thresholds = np.unique(np.concatenate([
        np.linspace(0.0, 0.1, 30),  # Fine-grained for small values
        np.linspace(0.1, 1.0, 40),  # Medium range
        np.exp(np.linspace(0, np.log(5), 30))  # Extended range for high uncertainty
    ]))
    
    # Print initial calibration set performance
    cal_scores = compute_raps_scores(cal_softmax, cal_labels, k_reg, lam_reg)
    cal_threshold = np.quantile(cal_scores, np.ceil((len(cal_scores)+1)*(1-alpha))/len(cal_scores), method='higher')
    cal_sets = get_prediction_sets_raps(cal_softmax, cal_threshold, k_reg, lam_reg)
    cal_coverage = np.mean(cal_sets[np.arange(len(cal_labels)), cal_labels])
    logger.info(f"\nInitial calibration set coverage: {cal_coverage:.4f}")
    logger.info(f"Target coverage: {1-alpha:.4f}")
    
    # Create plot directories
    plot_dirs = create_plot_dirs('plots')
    
    # Results storage for all corruptions
    metrics_data = {
        corruption_name: {
            'severities': [],
            'coverages': [],
            'set_sizes': [],
            'abstention_rates': [],
            'results_by_severity': {},
            'set_sizes_by_severity': {}
        }
        for corruption_name in corruptions.keys()
    }

    for corruption_name, corruption_type in corruptions.items():
        logger.info(f"\nAnalyzing {corruption_name} corruption")
        
        for severity in severity_levels:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing {corruption_name} severity level {severity}")
            
            # Create corrupted dataset
            corrupted_dataset = CorruptedModelNet40Dataset(
                base_dataset=test_dataset,
                corruption_type=corruption_type,
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
                alpha=0.1  # Fixed alpha without adjustment
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
                softmax_scores=val_softmax
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
            metrics_data[corruption_name]['severities'].append(severity)
            metrics_data[corruption_name]['coverages'].append(coverage)
            metrics_data[corruption_name]['set_sizes'].append(avg_set_size)
            metrics_data[corruption_name]['abstention_rates'].append(best_metrics['abstention_rate'])
            
            # Store results and set sizes for aggregate plots
            metrics_data[corruption_name]['results_by_severity'][severity] = {
                'abstention_results': results,
                'auc': current_auc  # Store the AUC for this severity
            }
            metrics_data[corruption_name]['set_sizes_by_severity'][severity] = prediction_sets.sum(axis=1)
            
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

    # Generate aggregate plots for all corruptions
    plot_metrics_vs_severity(
        severities=[metrics_data[name]['severities'] for name in corruptions.keys()],
        coverages=[metrics_data[name]['coverages'] for name in corruptions.keys()],
        set_sizes=[metrics_data[name]['set_sizes'] for name in corruptions.keys()],
        abstention_rates=[metrics_data[name]['abstention_rates'] for name in corruptions.keys()],
        labels=list(corruptions.keys()),
        save_dir=plot_dirs['metrics']
    )
    
    plot_roc_curves(
        {name: metrics_data[name]['results_by_severity'] for name in corruptions.keys()},
        save_dir=plot_dirs['roc']
    )
    
    plot_set_size_distribution(
        {name: metrics_data[name]['set_sizes_by_severity'] for name in corruptions.keys()},
        save_dir=plot_dirs['set_sizes']
    )
    
    logger.info("\nAnalysis completed successfully.")
    logger.info(f"Plots saved in {plot_dirs['metrics']}")

if __name__ == '__main__':
    main()