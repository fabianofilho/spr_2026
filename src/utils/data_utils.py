"""
SPR 2026 Mammography Report Classification - Data Utilities
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset


class MammographyDataset(Dataset):
    """Dataset for mammography report classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item


class MammographySeq2SeqDataset(Dataset):
    """Dataset for seq2seq models (T5, etc.)."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 512,
        max_target_length: int = 8,
        input_prefix: str = "classificar mamografia: ",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.input_prefix = input_prefix
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.input_prefix + str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        if self.labels is not None:
            target_text = str(self.labels[idx])
            target_encoding = self.tokenizer(
                target_text,
                truncation=True,
                max_length=self.max_target_length,
                padding="max_length",
                return_tensors="pt",
            )
            item["labels"] = target_encoding["input_ids"].squeeze()
            
        return item


def load_data(
    train_path: str,
    test_path: Optional[str] = None,
    text_column: str = "report",
    target_column: str = "target",
    id_column: str = "ID",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load training and test data."""
    
    train_df = pd.read_csv(train_path)
    print(f"Train shape: {train_df.shape}")
    print(f"Target distribution:\n{train_df[target_column].value_counts().sort_index()}")
    
    test_df = None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df


def create_folds(
    df: pd.DataFrame,
    target_column: str = "target",
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create stratified k-fold splits."""
    
    df = df.copy()
    df["fold"] = -1
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (_, val_idx) in enumerate(skf.split(df, df[target_column])):
        df.loc[val_idx, "fold"] = fold
    
    print(f"Fold distribution:")
    for fold in range(n_folds):
        fold_df = df[df["fold"] == fold]
        print(f"  Fold {fold}: {len(fold_df)} samples")
        
    return df


def get_class_weights(labels: List[int], num_classes: int = 7) -> torch.Tensor:
    """Calculate class weights for imbalanced data."""
    from collections import Counter
    
    counts = Counter(labels)
    total = sum(counts.values())
    
    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            weights.append(total / (num_classes * counts[i]))
        else:
            weights.append(1.0)
    
    return torch.tensor(weights, dtype=torch.float32)


# BI-RADS category descriptions (useful for prompts)
BIRADS_DESCRIPTIONS = {
    0: "Incompleto - necessita avaliação adicional",
    1: "Negativo - achado normal",
    2: "Achado benigno",
    3: "Provavelmente benigno - seguimento de curto intervalo sugerido",
    4: "Anormalidade suspeita - considerar biópsia",
    5: "Altamente sugestivo de malignidade",
    6: "Malignidade comprovada por biópsia",
}
