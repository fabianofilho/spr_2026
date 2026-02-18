"""
SPR 2026 Mammography Report Classification - Evaluation Metrics
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 7,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for the competition.
    Primary metric: F1-score macro
    """
    
    # Ensure predictions are class indices
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    metrics = {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
    }
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    for i, f1 in enumerate(f1_per_class):
        metrics[f"f1_class_{i}"] = f1
    
    return metrics


def print_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    target_names: Optional[List[str]] = None,
):
    """Print detailed classification report."""
    
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    if target_names is None:
        target_names = [f"BI-RADS {i}" for i in range(7)]
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(labels, predictions, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))


def compute_metrics_hf(eval_pred):
    """
    Compute metrics in HuggingFace Trainer format.
    Use this with transformers.Trainer.
    """
    predictions, labels = eval_pred
    
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    return {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }
