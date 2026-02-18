"""
GEMMA 3 Model for Mammography Report Classification
Uses LoRA/QLoRA for efficient fine-tuning
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.utils import (
    set_seed,
    load_config,
    merge_configs,
    get_device,
    count_parameters,
    create_experiment_dir,
    load_data,
    create_folds,
    compute_metrics,
)


class GemmaClassificationDataset(Dataset):
    """Dataset for Gemma classification with prompts."""
    
    def __init__(
        self,
        texts: list,
        labels: list = None,
        tokenizer=None,
        max_length: int = 2048,
        prompt_template: str = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.prompt_template = prompt_template or """<bos><start_of_turn>user
Classifique este relatório de mamografia na categoria BI-RADS (0-6):

{report}
<end_of_turn>
<start_of_turn>model
"""
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        prompt = self.prompt_template.format(report=text)
        
        if self.labels is not None:
            # Training: include the answer
            full_text = prompt + str(self.labels[idx]) + "<end_of_turn>"
            
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            # Create labels: -100 for prompt tokens (don't compute loss)
            prompt_encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = prompt_encoding["input_ids"].shape[1]
            
            labels = encoding["input_ids"].clone().squeeze()
            labels[:prompt_len] = -100
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels,
            }
        else:
            # Inference
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
            }


def setup_model_with_lora(config: dict, device: torch.device):
    """Setup Gemma model with QLoRA."""
    
    # Quantization config
    if config.get("quantization", {}).get("enabled", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["quantization"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare for k-bit training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    if config.get("lora", {}).get("enabled", False):
        lora_config = LoraConfig(
            r=config["lora"].get("r", 16),
            lora_alpha=config["lora"].get("lora_alpha", 32),
            lora_dropout=config["lora"].get("lora_dropout", 0.05),
            target_modules=config["lora"].get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        print("LoRA config applied:")
        model.print_trainable_parameters()
    
    return model


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    config: dict,
    fold: int,
    output_dir: str,
    device: torch.device,
):
    """Train Gemma model for a single fold."""
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    
    # Get prompt template from config
    prompt_template = config.get("prompt", {}).get("template", None)
    
    # Create datasets
    train_dataset = GemmaClassificationDataset(
        texts=train_df[config["data"]["text_column"]].tolist(),
        labels=train_df[config["data"]["target_column"]].tolist(),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        prompt_template=prompt_template,
    )
    
    val_dataset = GemmaClassificationDataset(
        texts=val_df[config["data"]["text_column"]].tolist(),
        labels=val_df[config["data"]["target_column"]].tolist(),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        prompt_template=prompt_template,
    )
    
    # Setup model with LoRA
    model = setup_model_with_lora(config, device)
    
    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/fold_{fold}",
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"]["fp16"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=config["logging"]["log_steps"],
        report_to="wandb" if config["logging"]["use_wandb"] else "none",
        seed=config["training"]["seed"],
        optim="paged_adamw_8bit" if config.get("quantization", {}).get("enabled") else "adamw_torch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Fold {fold} Results: {eval_results}")
    
    # Save model (LoRA adapters only)
    model.save_pretrained(f"{output_dir}/fold_{fold}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/fold_{fold}/best_model")
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train Gemma model for mammography classification")
    parser.add_argument("--config", type=str, default="src/configs/gemma_config.yaml")
    parser.add_argument("--base-config", type=str, default="src/configs/base_config.yaml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()
    
    # Load configs
    base_config = load_config(args.base_config)
    model_config = load_config(args.config)
    config = merge_configs(base_config, model_config)
    
    set_seed(config["training"]["seed"])
    device = get_device()
    
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
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
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
            device,
        )
        all_results.append(results)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
