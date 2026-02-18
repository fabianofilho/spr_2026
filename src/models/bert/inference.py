"""
Inference script for BERT-based models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.utils import (
    set_seed,
    load_config,
    merge_configs,
    get_device,
    MammographyDataset,
    create_submission,
)


def load_models_for_ensemble(model_dirs: list, device: torch.device):
    """Load multiple fold models for ensemble prediction."""
    models = []
    for model_dir in model_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def predict_ensemble(
    models: list,
    test_dataset,
    device: torch.device,
    batch_size: int = 32,
):
    """Make predictions using ensemble of models."""
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    
    for model in models:
        model_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                model_probs.append(probs.cpu().numpy())
        
        model_probs = np.concatenate(model_probs, axis=0)
        all_probs.append(model_probs)
    
    # Average probabilities across models
    ensemble_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(ensemble_probs, axis=1)
    
    return predictions, ensemble_probs


def main():
    parser = argparse.ArgumentParser(description="Inference for BERT model")
    parser.add_argument("--config", type=str, default="src/configs/bert_config.yaml")
    parser.add_argument("--base-config", type=str, default="src/configs/base_config.yaml")
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--test-path", type=str, default="data/test.csv")
    parser.add_argument("--output", type=str, default="submissions/submission.csv")
    args = parser.parse_args()
    
    # Load configs
    base_config = load_config(args.base_config)
    model_config = load_config(args.config)
    config = merge_configs(base_config, model_config)
    
    set_seed(config["training"]["seed"])
    device = get_device()
    
    # Load test data
    test_df = pd.read_csv(args.test_path)
    print(f"Test shape: {test_df.shape}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Create test dataset
    test_dataset = MammographyDataset(
        texts=test_df[config["data"]["text_column"]].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
    )
    
    # Find all fold models
    model_dirs = []
    for fold in range(config["training"]["num_folds"]):
        fold_dir = os.path.join(args.experiment_dir, f"fold_{fold}/best_model")
        if os.path.exists(fold_dir):
            model_dirs.append(fold_dir)
    
    print(f"Found {len(model_dirs)} fold models")
    
    # Load models
    models = load_models_for_ensemble(model_dirs, device)
    
    # Make predictions
    predictions, probs = predict_ensemble(
        models,
        test_dataset,
        device,
        batch_size=config["training"]["batch_size"] * 2,
    )
    
    # Create submission
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_submission(
        test_df[config["data"]["id_column"]].values,
        predictions,
        args.output,
    )


if __name__ == "__main__":
    main()
