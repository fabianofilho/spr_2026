"""
Training script for BERT-based models on Mammography Classification
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.utils import (
    set_seed,
    load_config,
    merge_configs,
    get_device,
    count_parameters,
    create_experiment_dir,
    create_submission,
    MammographyDataset,
    load_data,
    create_folds,
    get_class_weights,
    compute_metrics_hf,
    print_classification_report,
)


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    config: dict,
    fold: int,
    output_dir: str,
):
    """Train model for a single fold."""
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = MammographyDataset(
        texts=train_df[config["data"]["text_column"]].tolist(),
        labels=train_df[config["data"]["target_column"]].tolist(),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
    )
    
    val_dataset = MammographyDataset(
        texts=val_df[config["data"]["text_column"]].tolist(),
        labels=val_df[config["data"]["target_column"]].tolist(),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=config["data"]["num_classes"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
    )
    
    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/fold_{fold}",
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"] * 2,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"]["fp16"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=config["logging"]["log_steps"],
        report_to="wandb" if config["logging"]["use_wandb"] else "none",
        seed=config["training"]["seed"],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_hf,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["evaluation"]["early_stopping_patience"]
            )
        ],
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Fold {fold} Results: {eval_results}")
    
    # Get predictions for validation set
    predictions = trainer.predict(val_dataset)
    val_preds = np.argmax(predictions.predictions, axis=1)
    
    # Print detailed report
    print_classification_report(
        val_preds,
        val_df[config["data"]["target_column"]].values,
    )
    
    # Save best model
    trainer.save_model(f"{output_dir}/fold_{fold}/best_model")
    
    return trainer, eval_results, val_preds


def main():
    parser = argparse.ArgumentParser(description="Train BERT model for mammography classification")
    parser.add_argument("--config", type=str, default="src/configs/bert_config.yaml")
    parser.add_argument("--base-config", type=str, default="src/configs/base_config.yaml")
    parser.add_argument("--fold", type=int, default=None, help="Train specific fold only")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()
    
    # Load configs
    base_config = load_config(args.base_config)
    model_config = load_config(args.config)
    config = merge_configs(base_config, model_config)
    
    # Set seed
    set_seed(config["training"]["seed"])
    
    # Create experiment directory
    model_name_short = config["model"]["name"].split("/")[-1]
    experiment_dir = create_experiment_dir(
        config["output"]["model_dir"],
        model_name_short,
        args.experiment_name,
    )
    print(f"Experiment directory: {experiment_dir}")
    
    # Load data
    train_df, _ = load_data(
        config["data"]["train_path"],
        text_column=config["data"]["text_column"],
        target_column=config["data"]["target_column"],
    )
    
    # Create folds
    train_df = create_folds(
        train_df,
        target_column=config["data"]["target_column"],
        n_folds=config["training"]["num_folds"],
        seed=config["training"]["seed"],
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Train folds
    folds_to_train = [args.fold] if args.fold is not None else range(config["training"]["num_folds"])
    
    all_results = []
    for fold in folds_to_train:
        fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
        
        _, results, _ = train_fold(
            fold_train_df,
            fold_val_df,
            tokenizer,
            config,
            fold,
            str(experiment_dir),
        )
        all_results.append(results)
    
    # Print summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    avg_f1 = np.mean([r["eval_f1_macro"] for r in all_results])
    print(f"Average F1-Macro across folds: {avg_f1:.4f}")
    
    return avg_f1


if __name__ == "__main__":
    main()
