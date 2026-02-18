"""
T5-based Model for Mammography Report Classification
Uses sequence-to-sequence approach for classification
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.utils import (
    set_seed,
    load_config,
    merge_configs,
    get_device,
    count_parameters,
    create_experiment_dir,
    create_submission,
    MammographySeq2SeqDataset,
    load_data,
    create_folds,
    compute_metrics,
)


def compute_metrics_seq2seq(eval_pred, tokenizer):
    """Compute metrics for seq2seq model."""
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (padding) with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert to integers
    pred_labels = []
    true_labels = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            pred_labels.append(int(pred.strip()))
        except ValueError:
            pred_labels.append(0)  # Default to 0 if parsing fails
        
        try:
            true_labels.append(int(label.strip()))
        except ValueError:
            true_labels.append(0)
    
    # Clip to valid range
    pred_labels = np.clip(pred_labels, 0, 6)
    
    return compute_metrics(np.array(pred_labels), np.array(true_labels))


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    config: dict,
    fold: int,
    output_dir: str,
):
    """Train T5 model for a single fold."""
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    
    input_prefix = config.get("t5", {}).get("input_prefix", "classificar mamografia: ")
    
    # Create datasets
    train_dataset = MammographySeq2SeqDataset(
        texts=train_df[config["data"]["text_column"]].tolist(),
        labels=train_df[config["data"]["target_column"]].tolist(),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        max_target_length=config["model"].get("max_target_length", 8),
        input_prefix=input_prefix,
    )
    
    val_dataset = MammographySeq2SeqDataset(
        texts=val_df[config["data"]["text_column"]].tolist(),
        labels=val_df[config["data"]["target_column"]].tolist(),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        max_target_length=config["model"].get("max_target_length", 8),
        input_prefix=input_prefix,
    )
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(config["model"]["name"])
    
    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        generation_max_length=config["model"].get("max_target_length", 8),
        logging_steps=config["logging"]["log_steps"],
        report_to="wandb" if config["logging"]["use_wandb"] else "none",
        seed=config["training"]["seed"],
    )
    
    # Trainer with custom compute_metrics
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics_seq2seq(eval_pred, tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
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
    
    # Save best model
    trainer.save_model(f"{output_dir}/fold_{fold}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/fold_{fold}/best_model")
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train T5 model for mammography classification")
    parser.add_argument("--config", type=str, default="src/configs/t5_config.yaml")
    parser.add_argument("--base-config", type=str, default="src/configs/base_config.yaml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()
    
    # Load configs
    base_config = load_config(args.base_config)
    model_config = load_config(args.config)
    config = merge_configs(base_config, model_config)
    
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
        
        _, results = train_fold(
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


if __name__ == "__main__":
    main()
