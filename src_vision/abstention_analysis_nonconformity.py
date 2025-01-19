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
from data.corruptions import CorruptedCIFAR100Dataset, FogCorruption, SnowCorruption, RainCorruption, MotionBlurCorruption
from utils.visualization import plot_metrics_vs_severity, plot_roc_curves, plot_set_size_distribution, create_plot_dirs, plot_abstention_analysis,plot_confidence_distributions, analyze_severity_impact
from utils.model_utils import get_model_predictions
from utils.logging_utils import setup_logging


def compute_conformal_scores(
    softmax_scores: np.ndarray, 
    labels: np.ndarray, 
    k_reg: int = 5, 
    lam_reg: float = 0.02
) -> np.ndarray:
    """
    Compute adaptive conformal scores.
    For calibration data, we use the base regularization (no severity adjustment).
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

def analyze_distribution_shift(softmax_scores_by_severity):
    """
    Analyze distribution shift patterns in softmax scores across severities.
    Args:
        softmax_scores_by_severity: Dict mapping severity levels to softmax score arrays
    """
    for severity, scores in softmax_scores_by_severity.items():
        # Analyze probability distribution patterns
        top_k_probs = np.sort(scores, axis=1)[:, -5:]  # Top 5 probabilities
        prob_spread = np.mean(top_k_probs, axis=0)
        entropy = -np.sum(scores * np.log(scores + 1e-7), axis=1).mean()
        
        print(f"\nSeverity {severity}:")
        print(f"Average top-5 probs: {prob_spread}")
        print(f"Average entropy: {entropy}")

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
    lam_reg = 0.02  # Regularization strength
    alpha = 0.08  # Target 90% coverage
    
    # Analysis parameters
    severity_levels = [1, 2, 3, 4, 5]
    
    # Increase the abstention span with a combination of linear and log-spaced thresholds
    low_range = np.linspace(0, 0.1, 50)
    mid_range = np.linspace(0.1, 1, 50)
    high_range = np.exp(np.linspace(0, np.log(5), 50))
    abstention_thresholds = np.unique(np.concatenate([low_range, mid_range, high_range]))
    
    # Get calibration predictions
    cal_probs, cal_labels = get_model_predictions(model, cal_loader, device)
    
    # Calibration phase
    logger.info("Starting calibration phase...")
    cal_scores = compute_conformal_scores(cal_probs, cal_labels, k_reg, lam_reg)
    conformal_qhat = find_prediction_set_threshold(cal_scores, alpha=alpha)
    
    # Print the value of conformal_qhat
    print(f"Conformal qhat: {conformal_qhat}")
    
    # Validate calibration set performance
    cal_sets = get_prediction_sets(cal_probs, conformal_qhat, k_reg, lam_reg)
    cal_metrics = evaluate_sets(cal_sets, cal_labels)
    logger.info(f"Calibration set metrics:")
    logger.info(f"Coverage: {cal_metrics['coverage']:.4f}")
    logger.info(f"Average set size: {cal_metrics['avg_set_size']:.4f}")
    
    # Compute abstention threshold on calibration data
    cal_prediction_sets = get_prediction_sets(cal_probs, conformal_qhat, k_reg, lam_reg)
    cal_nonconf_scores = compute_nonconformity_scores(cal_probs, cal_labels)
    abstention_qhat, _ = analyze_abstention(
        cal_nonconf_scores, 
        cal_prediction_sets,
        cal_labels,
        abstention_thresholds
    )
    
    print(f"Abstention qhat from calibration: {abstention_qhat}")
    
    # Results storage for all corruptions
    results_by_severity_fog = {}
    results_by_severity_snow = {}
    results_by_severity_rain = {}  # Add rain
    results_by_severity_motionblur = {}  # Add motion blur
    
    severities_fog, coverages_fog, set_sizes_fog, abstention_rates_fog = [], [], [], []
    severities_snow, coverages_snow, set_sizes_snow, abstention_rates_snow = [], [], [], []
    severities_rain, coverages_rain, set_sizes_rain, abstention_rates_rain = [], [], [], []  # Add rain
    severities_motionblur, coverages_motionblur, set_sizes_motionblur, abstention_rates_motionblur = [], [], [], []  # Add motion blur
    
    set_sizes_by_severity_fog = {}
    set_sizes_by_severity_snow = {}
    set_sizes_by_severity_rain = {}  # Add rain
    set_sizes_by_severity_motionblur = {}  # Add motion blur
    
    softmax_scores_by_severity_fog = {}
    softmax_scores_by_severity_snow = {}
    softmax_scores_by_severity_rain = {}  # Add rain
    softmax_scores_by_severity_motionblur = {}  # Add motion blur
    
    # Track uncertainty metrics
    uncertainty_metrics_by_severity_fog = {}
    uncertainty_metrics_by_severity_snow = {}
    uncertainty_metrics_by_severity_rain = {}  # Add rain
    uncertainty_metrics_by_severity_motionblur = {}  # Add motion blur

    # Analyze both corruptions
    for severity in severity_levels:
        logger.info(f"\nAnalyzing severity {severity}")
        
        # FOG ANALYSIS
        logger.info("Processing Fog corruption...")
        corrupted_dataset = CorruptedCIFAR100Dataset(
            test_dataset, FogCorruption, severity=severity
        )
        corrupted_loader = DataLoader(
            corrupted_dataset, batch_size=128, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_probs_fog, test_labels = get_model_predictions(model, corrupted_loader, device)
        
        # Store fog results
        softmax_scores_by_severity_fog[severity] = test_probs_fog
        uncertainty_metrics = compute_uncertainty_metrics(test_probs_fog)
        uncertainty_metrics_by_severity_fog[severity] = uncertainty_metrics
        
        # Generate prediction sets for fog
        prediction_sets_fog = get_prediction_sets(test_probs_fog, conformal_qhat, k_reg, lam_reg)
        metrics_fog = evaluate_sets(prediction_sets_fog, test_labels)
        set_sizes_by_severity_fog[severity] = prediction_sets_fog.sum(axis=1)
        
        # Compute abstention metrics for fog
        nonconf_scores_fog = compute_nonconformity_scores(test_probs_fog, test_labels)
        true_labels_in_set_fog = prediction_sets_fog[np.arange(len(test_labels)), test_labels]
        should_abstain_fog = ~true_labels_in_set_fog

        # SNOW ANALYSIS
        logger.info("Processing Snow corruption...")
        corrupted_dataset = CorruptedCIFAR100Dataset(
            test_dataset, SnowCorruption, severity=severity
        )
        corrupted_loader = DataLoader(
            corrupted_dataset, batch_size=128, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_probs_snow, test_labels = get_model_predictions(model, corrupted_loader, device)
        
        # Store snow results
        softmax_scores_by_severity_snow[severity] = test_probs_snow
        uncertainty_metrics = compute_uncertainty_metrics(test_probs_snow)
        uncertainty_metrics_by_severity_snow[severity] = uncertainty_metrics
        
        # Generate prediction sets for snow
        prediction_sets_snow = get_prediction_sets(test_probs_snow, conformal_qhat, k_reg, lam_reg)
        metrics_snow = evaluate_sets(prediction_sets_snow, test_labels)
        set_sizes_by_severity_snow[severity] = prediction_sets_snow.sum(axis=1)
        
        # Compute abstention metrics for snow
        nonconf_scores_snow = compute_nonconformity_scores(test_probs_snow, test_labels)
        true_labels_in_set_snow = prediction_sets_snow[np.arange(len(test_labels)), test_labels]
        should_abstain_snow = ~true_labels_in_set_snow

        # Calculate abstention results for both
        abstention_results_fog = {}
        abstention_results_snow = {}
        for threshold in abstention_thresholds:
            # Fog abstention
            abstained_fog = nonconf_scores_fog > threshold
            tp_fog = np.sum(abstained_fog & should_abstain_fog)
            fp_fog = np.sum(abstained_fog & ~should_abstain_fog)
            tn_fog = np.sum(~abstained_fog & ~should_abstain_fog)
            fn_fog = np.sum(~abstained_fog & should_abstain_fog)
            
            tpr_fog = tp_fog / (tp_fog + fn_fog) if (tp_fog + fn_fog) > 0 else 0
            fpr_fog = fp_fog / (fp_fog + tn_fog) if (fp_fog + tn_fog) > 0 else 0
            
            abstention_results_fog[threshold] = {
                'tpr': tpr_fog,
                'fpr': fpr_fog,
                'abstention_rate': np.mean(abstained_fog)
            }
            
            # Snow abstention
            abstained_snow = nonconf_scores_snow > threshold
            tp_snow = np.sum(abstained_snow & should_abstain_snow)
            fp_snow = np.sum(abstained_snow & ~should_abstain_snow)
            tn_snow = np.sum(~abstained_snow & ~should_abstain_snow)
            fn_snow = np.sum(~abstained_snow & should_abstain_snow)
            
            tpr_snow = tp_snow / (tp_snow + fn_snow) if (tp_snow + fn_snow) > 0 else 0
            fpr_snow = fp_snow / (fp_snow + tn_snow) if (fp_snow + tn_snow) > 0 else 0
            
            abstention_results_snow[threshold] = {
                'tpr': tpr_snow,
                'fpr': fpr_snow,
                'abstention_rate': np.mean(abstained_snow)
            }
        
        # Calculate AUC for both
        auc_fog = calculate_auc(abstention_results_fog)
        auc_snow = calculate_auc(abstention_results_snow)
        
        # Store metrics for both
        severities_fog.append(severity)
        coverages_fog.append(metrics_fog['coverage'])
        set_sizes_fog.append(metrics_fog['avg_set_size'])
        abstention_rates_fog.append(np.mean([res['abstention_rate'] for res in abstention_results_fog.values()]))
        
        severities_snow.append(severity)
        coverages_snow.append(metrics_snow['coverage'])
        set_sizes_snow.append(metrics_snow['avg_set_size'])
        abstention_rates_snow.append(np.mean([res['abstention_rate'] for res in abstention_results_snow.values()]))
        
        # Store results for both
        results_by_severity_fog[severity] = {
            **metrics_fog,
            'abstention_qhat': abstention_qhat,
            'abstention_results': abstention_results_fog,
            'auc': auc_fog,
            'prediction_sets': prediction_sets_fog
        }
        
        results_by_severity_snow[severity] = {
            **metrics_snow,
            'abstention_qhat': abstention_qhat,
            'abstention_results': abstention_results_snow,
            'auc': auc_snow,
            'prediction_sets': prediction_sets_snow
        }
        
        # Log results for both
        logger.info("\nFog Results:")
        logger.info(f"Coverage: {metrics_fog['coverage']:.4f}")
        logger.info(f"Average set size: {metrics_fog['avg_set_size']:.4f}")
        logger.info(f"Set size std: {metrics_fog['std_set_size']:.4f}")
        logger.info(f"AUC: {auc_fog:.4f}")
        
        logger.info("\nSnow Results:")
        logger.info(f"Coverage: {metrics_snow['coverage']:.4f}")
        logger.info(f"Average set size: {metrics_snow['avg_set_size']:.4f}")
        logger.info(f"Set size std: {metrics_snow['std_set_size']:.4f}")
        logger.info(f"AUC: {auc_snow:.4f}")

        # RAIN ANALYSIS
        logger.info("Processing Rain corruption...")
        corrupted_dataset = CorruptedCIFAR100Dataset(
            test_dataset, RainCorruption, severity=severity
        )
        corrupted_loader = DataLoader(
            corrupted_dataset, batch_size=128, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_probs_rain, test_labels = get_model_predictions(model, corrupted_loader, device)
        
        # Store rain results
        softmax_scores_by_severity_rain[severity] = test_probs_rain
        uncertainty_metrics = compute_uncertainty_metrics(test_probs_rain)
        uncertainty_metrics_by_severity_rain[severity] = uncertainty_metrics
        
        # Generate prediction sets for rain
        prediction_sets_rain = get_prediction_sets(test_probs_rain, conformal_qhat, k_reg, lam_reg)
        metrics_rain = evaluate_sets(prediction_sets_rain, test_labels)
        set_sizes_by_severity_rain[severity] = prediction_sets_rain.sum(axis=1)
        
        # Compute abstention metrics for rain
        nonconf_scores_rain = compute_nonconformity_scores(test_probs_rain, test_labels)
        true_labels_in_set_rain = prediction_sets_rain[np.arange(len(test_labels)), test_labels]
        should_abstain_rain = ~true_labels_in_set_rain
        
        # Add rain abstention results
        abstention_results_rain = {}
        for threshold in abstention_thresholds:
            abstained_rain = nonconf_scores_rain > threshold
            tp_rain = np.sum(abstained_rain & should_abstain_rain)
            fp_rain = np.sum(abstained_rain & ~should_abstain_rain)
            tn_rain = np.sum(~abstained_rain & ~should_abstain_rain)
            fn_rain = np.sum(~abstained_rain & should_abstain_rain)
            
            tpr_rain = tp_rain / (tp_rain + fn_rain) if (tp_rain + fn_rain) > 0 else 0
            fpr_rain = fp_rain / (fp_rain + tn_rain) if (fp_rain + tn_rain) > 0 else 0
            
            abstention_results_rain[threshold] = {
                'tpr': tpr_rain,
                'fpr': fpr_rain,
                'abstention_rate': np.mean(abstained_rain)
            }
        
        # Calculate AUC for rain
        auc_rain = calculate_auc(abstention_results_rain)
        
        # Store metrics for rain
        severities_rain.append(severity)
        coverages_rain.append(metrics_rain['coverage'])
        set_sizes_rain.append(metrics_rain['avg_set_size'])
        abstention_rates_rain.append(np.mean([res['abstention_rate'] for res in abstention_results_rain.values()]))
        
        # Store rain results
        results_by_severity_rain[severity] = {
            **metrics_rain,
            'abstention_qhat': abstention_qhat,
            'abstention_results': abstention_results_rain,
            'auc': auc_rain,
            'prediction_sets': prediction_sets_rain
        }
        
        # Log rain results
        logger.info("\nRain Results:")
        logger.info(f"Coverage: {metrics_rain['coverage']:.4f}")
        logger.info(f"Average set size: {metrics_rain['avg_set_size']:.4f}")
        logger.info(f"Set size std: {metrics_rain['std_set_size']:.4f}")
        logger.info(f"AUC: {auc_rain:.4f}")

        # MOTION BLUR ANALYSIS
        logger.info("Processing Motion Blur corruption...")
        corrupted_dataset = CorruptedCIFAR100Dataset(
            test_dataset, MotionBlurCorruption, severity=severity
        )
        corrupted_loader = DataLoader(
            corrupted_dataset, batch_size=128, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_probs_motionblur, test_labels = get_model_predictions(model, corrupted_loader, device)
        
        # Store motion blur results
        softmax_scores_by_severity_motionblur[severity] = test_probs_motionblur
        uncertainty_metrics = compute_uncertainty_metrics(test_probs_motionblur)
        uncertainty_metrics_by_severity_motionblur[severity] = uncertainty_metrics
        
        # Generate prediction sets for motion blur
        prediction_sets_motionblur = get_prediction_sets(test_probs_motionblur, conformal_qhat, k_reg, lam_reg)
        metrics_motionblur = evaluate_sets(prediction_sets_motionblur, test_labels)
        set_sizes_by_severity_motionblur[severity] = prediction_sets_motionblur.sum(axis=1)
        
        # Compute abstention metrics for motion blur
        nonconf_scores_motionblur = compute_nonconformity_scores(test_probs_motionblur, test_labels)
        true_labels_in_set_motionblur = prediction_sets_motionblur[np.arange(len(test_labels)), test_labels]
        should_abstain_motionblur = ~true_labels_in_set_motionblur
        
        # Add motion blur abstention results
        abstention_results_motionblur = {}
        for threshold in abstention_thresholds:
            abstained_motionblur = nonconf_scores_motionblur > threshold
            tp_motionblur = np.sum(abstained_motionblur & should_abstain_motionblur)
            fp_motionblur = np.sum(abstained_motionblur & ~should_abstain_motionblur)
            tn_motionblur = np.sum(~abstained_motionblur & ~should_abstain_motionblur)
            fn_motionblur = np.sum(~abstained_motionblur & should_abstain_motionblur)
            
            tpr_motionblur = tp_motionblur / (tp_motionblur + fn_motionblur) if (tp_motionblur + fn_motionblur) > 0 else 0
            fpr_motionblur = fp_motionblur / (fp_motionblur + tn_motionblur) if (fp_motionblur + tn_motionblur) > 0 else 0
            
            abstention_results_motionblur[threshold] = {
                'tpr': tpr_motionblur,
                'fpr': fpr_motionblur,
                'abstention_rate': np.mean(abstained_motionblur)
            }
        
        # Calculate AUC for motion blur
        auc_motionblur = calculate_auc(abstention_results_motionblur)
        
        # Store metrics for motion blur
        severities_motionblur.append(severity)
        coverages_motionblur.append(metrics_motionblur['coverage'])
        set_sizes_motionblur.append(metrics_motionblur['avg_set_size'])
        abstention_rates_motionblur.append(np.mean([res['abstention_rate'] for res in abstention_results_motionblur.values()]))
        
        # Store motion blur results
        results_by_severity_motionblur[severity] = {
            **metrics_motionblur,
            'abstention_qhat': abstention_qhat,
            'abstention_results': abstention_results_motionblur,
            'auc': auc_motionblur,
            'prediction_sets': prediction_sets_motionblur
        }
        
        # Log motion blur results
        logger.info("\nMotion Blur Results:")
        logger.info(f"Coverage: {metrics_motionblur['coverage']:.4f}")
        logger.info(f"Average set size: {metrics_motionblur['avg_set_size']:.4f}")
        logger.info(f"Set size std: {metrics_motionblur['std_set_size']:.4f}")
        logger.info(f"AUC: {auc_motionblur:.4f}")
    
    # Create plot directories and update visualization calls
    plot_dirs = create_plot_dirs('plots_vision')
    
    # Generate visualizations for all corruptions
    plot_metrics_vs_severity(
        severities=[severities_fog, severities_snow, severities_rain, severities_motionblur],
        coverages=[coverages_fog, coverages_snow, coverages_rain, coverages_motionblur],
        set_sizes=[set_sizes_fog, set_sizes_snow, set_sizes_rain, set_sizes_motionblur],
        abstention_rates=[abstention_rates_fog, abstention_rates_snow, abstention_rates_rain, abstention_rates_motionblur],
        labels=['Fog', 'Snow', 'Rain', 'Motion Blur'],
        save_dir=plot_dirs['metrics']
    )
    
    plot_roc_curves(
        {
            'fog': results_by_severity_fog, 
            'snow': results_by_severity_snow,
            'rain': results_by_severity_rain,
            'motionblur': results_by_severity_motionblur
        }, 
        save_dir=plot_dirs['roc']
    )
    
    plot_set_size_distribution(
        {
            'fog': set_sizes_by_severity_fog, 
            'snow': set_sizes_by_severity_snow,
            'rain': set_sizes_by_severity_rain,
            'motionblur': set_sizes_by_severity_motionblur
        },
        save_dir=plot_dirs['set_sizes']
    )
    
    # Generate abstention analysis plots for all
    for severity in severity_levels:
        plot_abstention_analysis(
            thresholds=abstention_thresholds,
            results_fog=results_by_severity_fog[severity]['abstention_results'],
            results_snow=results_by_severity_snow[severity]['abstention_results'],
            results_rain=results_by_severity_rain[severity]['abstention_results'],
            results_motionblur=results_by_severity_motionblur[severity]['abstention_results'],
            severity=severity,
            save_dir=plot_dirs['abstention']
        )
    
    logger.info("\nAnalysis completed. Plots saved in plots_vision directory.")

if __name__ == '__main__':
    main()