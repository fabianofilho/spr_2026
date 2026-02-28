# Skill: HuggingFace Local/Offline

## 🎯 Quando Ativar

- Notebooks Kaggle com transformers
- Carregar modelos offline
- Evitar erros de validação HuggingFace Hub

---

## ⚠️ Regra Obrigatória

**SEMPRE usar `local_files_only=True`** para evitar erros de validação:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=7,
    local_files_only=True
)
```

---

## 🔧 TrainingArguments (transformers >= 4.46)

**USAR `eval_strategy`** em vez de `evaluation_strategy` (deprecado):

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='/kaggle/working/model',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    fp16=torch.cuda.is_available(),
    eval_strategy='epoch',      # ⚠️ NÃO usar evaluation_strategy
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    report_to='none',           # Desabilita wandb/tensorboard
)
```

---

## 📝 Template: Carregamento Completo

```python
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ============================================
# CONFIGURAÇÕES
# ============================================
NUM_LABELS = 7
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

# ============================================
# ENCONTRAR MODELO
# ============================================
def find_model_path():
    """Busca recursiva por config.json"""
    base = '/kaggle/input'
    
    def has_config(path):
        return os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.json'))
    
    def search_dir(directory, depth=0, max_depth=6):
        if depth > max_depth:
            return None
        try:
            for item in os.listdir(directory):
                path = os.path.join(directory, item)
                if os.path.isdir(path):
                    if has_config(path):
                        return path
                    result = search_dir(path, depth + 1, max_depth)
                    if result:
                        return result
        except:
            pass
        return None
    
    return search_dir(base)

MODEL_PATH = find_model_path()
print(f"Modelo: {MODEL_PATH}")

# ============================================
# CARREGAR TOKENIZER E MODELO
# ============================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=NUM_LABELS,
    local_files_only=True
)

# ============================================
# TRAINING ARGUMENTS
# ============================================
training_args = TrainingArguments(
    output_dir='/kaggle/working/model',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LR,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    report_to='none',
    logging_steps=50,
)
```

---

## ⚠️ Parâmetros Deprecados

| Antigo (deprecado) | Novo (usar este) |
|--------------------|------------------|
| `evaluation_strategy` | `eval_strategy` |
| `push_to_hub_model_id` | `hub_model_id` |

---

## 📋 Checklist

- [ ] `local_files_only=True` em todos os `from_pretrained()`
- [ ] `eval_strategy` em vez de `evaluation_strategy`
- [ ] `report_to='none'` (desabilita tracking)
- [ ] `fp16=torch.cuda.is_available()` para GPU
- [ ] Path do modelo usando `find_model_path()`
