import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.dataset import ModelNet40Dataset
from models.point_transformer_v2 import PointTransformerV2
from config import Config

def setup_logging(name='conformal_prediction'):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "conformal_prediction.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def create_dataloader(root_dir, split, batch_size=32, num_workers=4):
    dataset = ModelNet40Dataset(
        root_dir=root_dir,
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader

def get_softmax_predictions(model, dataloader, device):
    """Get softmax predictions for all samples"""
    model.eval()
    all_softmax = []
    all_labels = []
    
    with torch.no_grad():
        for points, targets in dataloader:
            points, targets = points.to(device), targets.to(device)
            outputs = model(points)
            softmax = F.softmax(outputs, dim=1)
            all_softmax.append(softmax.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    
    return np.concatenate(all_softmax), np.concatenate(all_labels)

def conformal_prediction(cal_softmax, cal_labels, val_softmax, val_labels, alpha=0.1, 
                        k_reg=5, lam_reg=0.01, rand=True, disallow_zero_sets=True):
    """Implement RAPS conformal prediction"""
    n = len(cal_labels)
    n_classes = cal_softmax.shape[1]
    
    # Create regularization vector
    reg_vec = np.array(k_reg*[0,] + (n_classes-k_reg)*[lam_reg,])[None,:]
    
    # Get scores for calibration set
    cal_pi = cal_softmax.argsort(1)[:,::-1]
    cal_srt = np.take_along_axis(cal_softmax, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    
    if rand:
        rand_vals = np.random.rand(n)
        cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - rand_vals*cal_srt_reg[np.arange(n),cal_L]
    else:
        cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - cal_srt_reg[np.arange(n),cal_L]
    
    # Get score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
    
    # Deploy on validation set
    n_val = len(val_labels)
    val_pi = val_softmax.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_softmax, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec
    
    if rand:
        rand_vals_val = np.random.rand(n_val,1)
        indicators = (val_srt_reg.cumsum(axis=1) - rand_vals_val*val_srt_reg) <= qhat
    else:
        indicators = val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        
    if disallow_zero_sets:
        indicators[:,0] = True
        
    prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
    
    # Calculate metrics
    empirical_coverage = prediction_sets[np.arange(n_val), val_labels].mean()
    set_sizes = prediction_sets.sum(axis=1)
    avg_set_size = set_sizes.mean()
    
    return empirical_coverage, avg_set_size, qhat, prediction_sets

def plot_set_size_analysis(prediction_sets, coverage, save_dir='plots'):
    """Create detailed visualizations of prediction set sizes"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    set_sizes = prediction_sets.sum(axis=1)
    unique_sizes, size_counts = np.unique(set_sizes, return_counts=True)
    size_percentages = (size_counts / len(set_sizes)) * 100
    
    plt.figure(figsize=(15, 10))
    
    # Create subplot layout
    gs = plt.GridSpec(2, 2)
    
    # 1. Set Size Distribution (Histogram)
    ax1 = plt.subplot(gs[0, 0])
    ax1.hist(set_sizes, bins=range(int(min(set_sizes)), int(max(set_sizes)) + 2, 1),
             rwidth=0.8, alpha=0.7)
    ax1.set_title('Distribution of Prediction Set Sizes')
    ax1.set_xlabel('Set Size')
    ax1.set_ylabel('Number of Predictions')
    ax1.grid(True, alpha=0.3)
    
    # 2. Set Size Frequency (Bar Plot with percentages)
    ax2 = plt.subplot(gs[0, 1])
    bars = ax2.bar(unique_sizes, size_percentages)
    ax2.set_title('Percentage of Each Set Size')
    ax2.set_xlabel('Set Size')
    ax2.set_ylabel('Percentage of Predictions (%)')
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative Distribution
    ax3 = plt.subplot(gs[1, 0])
    cum_percentages = np.cumsum(size_percentages)
    ax3.plot(unique_sizes, cum_percentages, marker='o')
    ax3.set_title('Cumulative Distribution of Set Sizes')
    ax3.set_xlabel('Set Size')
    ax3.set_ylabel('Cumulative Percentage (%)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    stats_text = (
        f"Summary Statistics:\n\n"
        f"Coverage: {coverage:.1f}%\n"
        f"Average Set Size: {set_sizes.mean():.2f}\n"
        f"Median Set Size: {np.median(set_sizes):.1f}\n"
        f"Mode Set Size: {unique_sizes[np.argmax(size_counts)]}\n"
        f"Max Set Size: {np.max(set_sizes)}\n"
        f"Min Set Size: {np.min(set_sizes)}\n\n"
        f"Single Class: {size_percentages[0]:.1f}%\n"
        f"Multiple Classes: {100-size_percentages[0]:.1f}%"
    )
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'set_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    Config.initialize()
    logger = setup_logging()
    
    device = torch.device(f'cuda:{Config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    calibration_loader = create_dataloader(Config.ROOT_DIR, 'calibration')
    test_loader = create_dataloader(Config.ROOT_DIR, 'test')
    
    # Initialize and load model
    model = PointTransformerV2(
        num_classes=Config.MODEL['num_classes'],
        in_channels=Config.MODEL['in_channels']
    ).to(device)
    
    checkpoint = torch.load('/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded trained model")
    
    # Get predictions
    logger.info("Getting calibration set predictions...")
    cal_softmax, cal_labels = get_softmax_predictions(model, calibration_loader, device)
    
    logger.info("Getting test set predictions...")
    test_softmax, test_labels = get_softmax_predictions(model, test_loader, device)
    
    # Run conformal prediction
    logger.info("Running conformal prediction...")
    coverage, avg_set_size, qhat, prediction_sets = conformal_prediction(
        cal_softmax, cal_labels,
        test_softmax, test_labels,
        alpha=0.1,  # 90% target coverage
        k_reg=5,
        lam_reg=0.01
    )
    
    # Log results
    logger.info(f"Quantile (qhat): {qhat:.4f}")
    logger.info(f"Empirical coverage: {coverage:.4f}")
    logger.info(f"Average set size: {avg_set_size:.2f}")
    logger.info(f"Median set size: {np.median(prediction_sets.sum(axis=1)):.2f}")
    logger.info(f"Max set size: {prediction_sets.sum(axis=1).max()}")
    logger.info(f"Min set size: {prediction_sets.sum(axis=1).min()}")
    
    # Create visualizations
    logger.info("Creating detailed set size analysis plots...")
    plot_set_size_analysis(prediction_sets, coverage*100)
    logger.info("Analysis plots saved to plots/set_size_analysis.png")

if __name__ == '__main__':
    main()