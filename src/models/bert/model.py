"""
BERT-based Model for Mammography Report Classification
Supports: BERTimbau, mBERT, XLM-RoBERTa
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
from typing import Optional


class BERTClassifier(nn.Module):
    """BERT-based classifier for BI-RADS classification."""
    
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        num_classes: int = 7,
        dropout: float = 0.1,
        pooling: str = "cls",  # cls, mean, max
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Pooling strategy
        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            hidden_states = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        elif self.pooling == "max":
            hidden_states = outputs.last_hidden_state
            hidden_states[attention_mask == 0] = -1e9
            pooled = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}


def get_bert_tokenizer(model_name: str = "neuralmind/bert-base-portuguese-cased"):
    """Load tokenizer for BERT model."""
    return AutoTokenizer.from_pretrained(model_name)


def get_bert_model(
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    num_classes: int = 7,
    dropout: float = 0.1,
    pooling: str = "cls",
) -> BERTClassifier:
    """Create BERT classifier model."""
    return BERTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
    )
