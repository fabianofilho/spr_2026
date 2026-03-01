# Skill: Fine-tuning de Transformers

## 🎯 Quando Ativar

- Treinar BERTimbau ou outros transformers
- Otimizar para classificação multiclasse desbalanceada
- Melhorar F1-Macro além do baseline

---

## 🏆 Melhor Configuração Comprovada

**BERTimbau v4 → Score: 0.82073** (superou baseline 0.77885 em +5.4%)

```python
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
SEED = 42
```

---

## ⚡ Técnicas que FUNCIONAM

### 1. Focal Loss (OBRIGATÓRIO para desbalanceamento)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# Usar no training loop:
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, labels)
```

**Parâmetros testados:**
- γ (gamma): 2.0 é o sweet spot
- α (alpha): 0.25 funciona bem

---

### 2. Threshold Tuning (OBRIGATÓRIO - até +3% F1)

```python
import numpy as np
from sklearn.metrics import f1_score

def find_thresholds(probs, labels, num_classes=7):
    """Grid search de thresholds por classe."""
    thresholds = {}
    for c in range(num_classes):
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.02):
            preds = np.argmax(probs, axis=1)
            mask = (probs[:, c] >= t) & (probs[:, c] == probs.max(axis=1))
            preds[mask] = c
            f1 = f1_score(labels, preds, average='macro')
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = round(best_t, 2)
    return thresholds

def apply_thresholds(probs, thresholds):
    """Aplica thresholds com prioridade para classes raras."""
    preds = []
    for i in range(len(probs)):
        pred = np.argmax(probs[i])
        # Prioridade reversa (classes raras primeiro)
        for c in sorted(thresholds.keys(), reverse=True):
            if probs[i, c] >= thresholds[c] and probs[i, c] > probs[i, pred] * 0.85:
                pred = c
                break
        preds.append(pred)
    return np.array(preds)
```

**Thresholds de referência (BERTimbau v4):**
```python
THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.30, 6: 0.25}
```

---

### 3. Seed Ensemble (RECOMENDADO - +0.5-1% F1)

```python
SEEDS = [42, 123, 456]  # 3 seeds
all_probs = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Treinar modelo...
    all_probs.append(test_probs)

# Média das probabilidades
final_probs = np.mean(all_probs, axis=0)
```

---

### 4. K-Fold CV (RECOMENDADO para estabilidade)

```python
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Treinar e inferir...
    test_probs += fold_probs / N_FOLDS
```

---

## ❌ Técnicas que NÃO FUNCIONAM

| Técnica | Resultado | Motivo |
|---------|-----------|--------|
| SMOTE | Regrediu | Desbalanceamento é bem tratado por Focal Loss |
| Tratamento de texto pesado | -2% F1 | Perde informações importantes |
| Label Smoothing alto (>0.1) | Prejudica | Confunde threshold tuning |
| MAX_LEN=512 | Timeout | Relatórios são curtos (~100 tokens) |
| EPOCHS>5 | Overfit | 5 épocas é suficiente |

---

## 📋 Checklist de Fine-tuning

- [ ] Focal Loss com γ=2.0, α=0.25
- [ ] `local_files_only=True` em todos carregamentos
- [ ] MAX_LEN=256 (suficiente para relatórios)
- [ ] BATCH_SIZE=8 ou 16
- [ ] Grid search de thresholds após treino
- [ ] Prioridade para classes raras (6, 5, 4...)
- [ ] Seed ensemble se tempo permitir

---

## 🔧 Template: Dataset PyTorch

```python
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
```

---

## 📊 Evolução de Scores

| Versão | Técnicas | Score |
|--------|----------|-------|
| Baseline TF-IDF | LinearSVC | 0.77885 |
| BERTimbau v1 | CrossEntropy | ~0.79 |
| BERTimbau v2 | + Focal Loss | ~0.80 |
| BERTimbau v3 | + Threshold Tuning | ~0.81 |
| BERTimbau v4 | + Seed Ensemble | **0.82073** |

---

## ⚠️ Avisos Importantes

1. **Sempre usar `local_files_only=True`** - evita erros no Kaggle offline
2. **Não usar `evaluation_strategy`** - deprecado, usar `eval_strategy`
3. **Salvar probabilidades, não predições** - permite threshold tuning pós-treino
