# src/config.py
from pathlib import Path

class Config:
    # Device settings
    GPU_ID = 1  # Specify which GPU to use
    
    # Paths
    ROOT_DIR = "/ssd_4TB/divake/CP_trust_IJCNN/dataset/modelnet40_normal_resampled"
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    
    # Dataset parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_POINTS = 1024
    
    # Model parameters
    MODEL = {
        'trans_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'drop_path_rate': 0.1,
        'num_classes': 40,
        'in_channels': 6  # New parameter for 6-channel input
    }
    
    # Training parameters
    TRAIN = {
        'num_epochs': 300,
        'learning_rate': 0.001,  # Slightly higher for training from scratch
        'weight_decay': 0.05,
        'warmup_epochs': 10,
        'grad_norm_clip': 10,
        'min_lr': 1e-5  # Minimum learning rate for cosine scheduling
    }

    @classmethod
    def initialize(cls):
        """Create necessary directories for checkpoints and logs"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Convert string paths to Path objects
        if not isinstance(cls.ROOT_DIR, Path):
            cls.ROOT_DIR = Path(cls.ROOT_DIR)