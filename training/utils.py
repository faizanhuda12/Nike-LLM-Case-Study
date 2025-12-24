"""
Utility functions for training: logging, validation, reproducibility
"""

import os
import sys
import logging
import torch
import transformers
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging for production"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility"""
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_cache(cache_dir: Optional[str]) -> None:
    """Setup HuggingFace cache directory"""
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    # Configure HuggingFace cache directories
    # Note: TRANSFORMERS_CACHE is deprecated; HF_HOME is the primary cache location
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")


def pick_precision() -> Tuple[bool, bool]:
    """
    Select optimal precision configuration based on hardware capabilities.
    
    Returns:
        Tuple of (bf16_enabled, fp16_enabled) flags.
        Prefers bf16 on supported hardware (e.g., RTX 4090/4080), falls back to fp16 otherwise.
    """
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if bf16_ok:
        return True, False
    return False, True


def validate_args(args) -> None:
    """Validate command-line arguments"""
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")
    
    if args.val_split < 0 or args.val_split >= 1:
        raise ValueError(f"val_split must be in [0, 1), got {args.val_split}")
    
    if args.lora_r <= 0:
        raise ValueError(f"lora_r must be > 0, got {args.lora_r}")
    
    if args.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be > 0, got {args.lora_alpha}")
    
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {args.learning_rate}")
    
    if args.num_train_epochs <= 0:
        raise ValueError(f"num_train_epochs must be > 0, got {args.num_train_epochs}")


def log_system_info(logger: logging.Logger) -> None:
    """Log system and environment information"""
    logger.info("=" * 72)
    logger.info("System Information")
    logger.info("=" * 72)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    logger.info("=" * 72)

