"""
SPR 2026 Mammography Report Classification - General Utilities
"""

import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge two config dictionaries, with override taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def create_experiment_dir(
    base_dir: str,
    model_name: str,
    experiment_name: Optional[str] = None,
) -> Path:
    """Create directory for experiment outputs."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{model_name}_{experiment_name}_{timestamp}"
    else:
        dir_name = f"{model_name}_{timestamp}"
    
    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    
    return experiment_dir


def create_submission(
    ids: np.ndarray,
    predictions: np.ndarray,
    output_path: str,
):
    """Create submission file in Kaggle format."""
    import pandas as pd
    
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    submission = pd.DataFrame({
        "ID": ids,
        "target": predictions,
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Prediction distribution:\n{submission['target'].value_counts().sort_index()}")
    
    return submission
