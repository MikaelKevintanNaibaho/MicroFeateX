import yaml
import random
import numpy as np
import torch
import os
from pathlib import Path
from typing import Dict, Any


def seed_everything(seed: int):
    """
    Ensures reproducibility across PyTorch, NumPy, and Python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Safely loads a YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging_dir(
    base_path: str, model_name: str, is_explicit_log_path: bool = False
) -> str:
    """
    Creates a logging directory named after the model.
    
    Uses just the model_name (no timestamp) so logs persist across training sessions.
    This means resuming training will continue logging to the same directory.
    """
    if is_explicit_log_path:
        # User defined a specific "logs" folder in config
        log_dir = os.path.join(base_path, model_name)
    else:
        # Default behavior: nest inside "logdir" to avoid cluttering checkpoints
        log_dir = os.path.join(base_path, "logdir", model_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
