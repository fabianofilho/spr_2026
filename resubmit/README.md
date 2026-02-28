# Resubmit - Modelos Top 5 (Score > 0.7)

## Objetivo
Melhorar os melhores modelos baseline com técnicas para lidar com o **desbalanceamento de classes**.

**Limite:** 5 submissões/dia

---

## 🏆 Melhor Modelo: BERTimbau + Focal Loss (0.79696)

### ✅ Modelos que Mantiveram/Melhoraram
| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 🏆 | BERTimbau + Focal Loss | 0.79696 | ✅ Baseline campeão |
| 2 | BERTimbau + Focal Loss v2 | 0.79505 | ✅ Estável (-0.2%) |
| 3 | Ensemble Soft Voting | 0.78049 | ✅ Baseline estável |
| 4 | **SGDClassifier v3** | **0.77036** | 🚀 **MELHOROU +2.7%** |
| 5 | LinearSVC | 0.77885 | ✅ Baseline estável |

---

## 📅 Submissões por Data

### 2026-02-24 (5/5 submissões)
| Modelo | Notebook | Score | Delta |
|--------|----------|-------|-------|
| BERTimbau + Focal Loss v2 | `resubmit_bertimbau_v2.ipynb` | **0.79505** | -0.19 pp |
| BERTimbau + Focal Loss v3 | `resubmit_bertimbau_focal_v3.ipynb` | 0.72625 | -7.07 pp |
| Ensemble Soft Voting v2 | `resubmit_ensemble_voting_v2.ipynb` | 0.76387 | -1.66 pp |
| Custom Transformer v2 | `resubmit_custom_transformer_v2.ipynb` | 0.41721 | -35.55 pp |
| BioBERTpt + Focal Loss v2 | `resubmit_biobertpt_focal_v2.ipynb` | 0.26099 | -46.38 pp |

### 2026-02-25 (5/5 submissões)
| Modelo | Notebook | Score | Status |
|--------|----------|-------|--------|
| LightGBM v2 | `resubmit_lgbm_v2.ipynb` | 0.60915 | ⚠️ Regrediu |
| LinearSVC v2 | `resubmit_linearsvc_v2.ipynb` | 0.77885 | ✅ Estável |
| LogReg v2 | `resubmit_logreg_v2.ipynb` | 0.72935 | ✅ Estável |
| SGD v2 | `resubmit_sgd_v2.ipynb` | 0.75019 | ✅ Estável |
| Qwen3 Zero-Shot | `resubmit_qwen3_zeroshot.ipynb` | 0.13261 | ❌ Falhou |

### 2026-02-26 (5/5 submissões) ✅
| Modelo | Notebook | Score | Status |
|--------|----------|-------|--------|
| **SGDClassifier v3** | `resubmit_sgd_v3.ipynb` | **0.77036** | 🚀 **+2.7%** |
| SGDClassifier v3 | - | 0.77036 | ⚠️ Duplicado |
| LinearSVC v3 | `resubmit_linearsvc_v3.ipynb` | 0.75966 | ⚠️ -2.5% |
| LogisticRegression v3 | `resubmit_logreg_v3.ipynb` | 0.71303 | ⚠️ -2.2% |
| Qwen3 One-Shot | `resubmit_qwen3_oneshot.ipynb` | 0.13261 | ❌ Falhou |

### 2026-02-27 (1/5 submissões) - HOJE
| Modelo | Notebook | Score | Status |
|--------|----------|-------|--------|
| LightGBM v3 | `resubmit_lgbm_v3.ipynb` | ⏳ | Pendente (reagendado) |

---

## 🚀 Plano: Próximos 3 Dias

> **Foco:** Replicar sucesso do SGDClassifier v3 (+2.7%) em outros modelos

### 2026-02-28 (5/5 planejadas)
| # | Modelo | Notebook | Estratégia |
|---|--------|----------|------------|
| 1 | SGDClassifier v4 | `resubmit_sgd_v4.ipynb` | RandomSearch 50 iter + SMOTE classes 5/6 |
| 2 | Ensemble v3 | `resubmit_ensemble_v3.ipynb` | SGD v3 + LinearSVC + LogReg (weighted blend) |
| 3 | BERTimbau v4 | `resubmit_bertimbau_v4.ipynb` | Threshold tuning apenas (não mexer no modelo) |
| 4 | LinearSVC v4 | `resubmit_linearsvc_v4.ipynb` | Calibration + Platt Scaling |
| 5 | **Gemma 3 4B** | `resubmit_gemma3_oneshot.ipynb` | 🆕 LLM One-Shot |

### 2026-03-01 (5/5 planejadas)
| # | Modelo | Notebook | Estratégia |
|---|--------|----------|------------|
| 1 | SGDClassifier v5 | `resubmit_sgd_v5.ipynb` | SGD v4 + sem classe 2 (hipótese pegadinha) |
| 2 | Ensemble v4 | `resubmit_ensemble_v4.ipynb` | BERTimbau + SGD v3 blend (0.6/0.4) |
| 3 | XLM-RoBERTa v2 | `resubmit_xlmroberta_v2.ipynb` | Focal Loss (copiar config BERTimbau) |
| 4 | ModernBERT v2 | `resubmit_modernbert_v2.ipynb` | Focal Loss (copiar config BERTimbau) |
| 5 | **Llama 3.2 3B** | `resubmit_llama3_oneshot.ipynb` | 🆕 LLM One-Shot |

### 2026-03-02 (5/5 planejadas)
| # | Modelo | Notebook | Estratégia |
|---|--------|----------|------------|
| 1 | Ensemble v5 | `resubmit_ensemble_v5.ipynb` | Top 3 voting: BERTimbau + SGD v4 + Ensemble v4 |
| 2 | SGDClassifier v6 | `resubmit_sgd_v6.ipynb` | SGD v5 + threshold por classe otimizado |
| 3 | BERTimbau v5 | `resubmit_bertimbau_v5.ipynb` | Data augmentation conservadora (EDA) |
| 4 | BioBERTpt v3 | `resubmit_biobertpt_v3.ipynb` | Config exata do BERTimbau v1 (sem mexer) |
| 5 | **Qwen3 4B** | `resubmit_qwen3_4b_oneshot.ipynb` | 🆕 LLM One-Shot (modelo maior) |

---

## 📊 Estratégias v4/v5/v6

### SGD v4: RandomSearch Intensivo + SMOTE
```python
N_SEARCH_ITER = 50          # vs 20 no v3
USE_SMOTE = True            # Oversample classes 5/6
sampling_strategy = {5: 500, 6: 500}
```

### Ensemble v3: Weighted Blend com SGD v3
```python
# Pesos baseados em scores públicos
weights = {
    'sgd_v3': 0.35,      # 0.77036 - único que melhorou
    'linearsvc': 0.35,   # 0.77885 - baseline estável  
    'logreg': 0.30       # 0.72935
}
final_pred = sum(w * proba for w, proba in zip(weights, probas))
```

### BERTimbau v4: Threshold Tuning (NÃO MEXER NO MODELO)
```python
# Apenas ajustar thresholds na inferência
thresholds = {
    0: 0.50, 1: 0.50, 2: 0.50, 
    3: 0.50, 4: 0.50, 
    5: 0.30,  # Mais sensível
    6: 0.25   # Muito mais sensível
}
```

### Ensemble v4: BERTimbau + SGD Blend
```python
# Hipótese: Combinar deep learning + clássico
final_proba = 0.6 * bertimbau_proba + 0.4 * sgd_v3_proba
```

### Transformers v2: Copiar Config Exata do BERTimbau
```python
# COPIAR EXATAMENTE - não inventar
focal_loss_gamma = 2.0
learning_rate = 2e-5
epochs = 5
batch_size = 16
```

---
- 7 epochs (vs 5)
- LR 3e-5 (vs 2e-5)

---

## Notebooks Disponíveis

### Versão 2 (Baseline com class_weight='balanced')
| Rank | Modelo | Score Base | Notebook |
|------|--------|------------|----------|
| 🏆 | BERTimbau + Focal | **0.79696** | `resubmit_bertimbau_v2.ipynb` |
| 2 | LinearSVC | 0.77885 | `resubmit_linearsvc_v2.ipynb` |
| 3 | SGDClassifier | 0.75019 | `resubmit_sgd_v2.ipynb` |
| 4 | Logistic Regression | 0.72935 | `resubmit_logreg_v2.ipynb` |
| 5 | LightGBM | 0.70273 | `resubmit_lgbm_v2.ipynb` |

### Versão 3 (RandomizedSearchCV + Estratégias)
| Modelo | Notebook | Estratégias |
|--------|----------|-------------|
| LinearSVC | `resubmit_linearsvc_v3.ipynb` | RandomSearch + Calibração + Thresholds |
| SGDClassifier | `resubmit_sgd_v3.ipynb` | RandomSearch + Thresholds |
| Logistic Regression | `resubmit_logreg_v3.ipynb` | RandomSearch + Thresholds |
| LightGBM | `resubmit_lgbm_v3.ipynb` | RandomSearch + Early Stop + Thresholds |

---

## Estratégias Implementadas (v3)

### 1. RandomizedSearchCV
- Tuning automático de hiperparâmetros
- 20-30 iterações com cross-validation 5-fold
- Métrica: F1-macro

### 2. Class Weights (já ativo em todos)
```python
class_weight='balanced'  # sklearn e LightGBM
```

### 3. SMOTE para Classes Minoritárias (5/6)
```python
USE_SMOTE = True  # Ativar no notebook
# Faz oversampling das classes 5 e 6 para 500 amostras
```

### 4. Threshold Tuning por Classe
```python
USE_THRESHOLD_TUNING = True  # Ativo por padrão
# Classes 5 e 6 têm threshold 0.35 (mais sensível)
# Outras classes têm threshold 0.5
```

### 5. Hipótese Classe 2 (Teste)
```python
REMOVE_CLASS_2 = True  # Ativar para testar
# Remove classe 2 do treino (suspeita de pegadinha)
```

---

## Como Usar

1. **Clonar** o notebook desejado (v3)
2. **Ajustar FLAGS** no início:
   - `REMOVE_CLASS_2`: True/False
   - `USE_SMOTE`: True/False
   - `USE_THRESHOLD_TUNING`: True/False
   - `N_SEARCH_ITER`: Número de iterações (20-50)
3. **Submeter** no Kaggle com Internet OFF

---

## Experimentos Planejados

| Versão | Melhoria | Status |
|--------|----------|--------|
| v2 | class_weight='balanced' | ⏳ |
| v3 | + tuning hiperparâmetros | ⏳ |
| v4 | + threshold optimization | ⏳ |
| v5 | sem classe 2 | ⏳ |

### BERTimbau + Focal Loss (melhorias)
| Versão | Estratégia | Status |
|--------|------------|--------|
| v3 | Alpha por classe + Threshold + Label smoothing | ⏳ |
| v4 | + SMOTE classes 5/6 | ⏳ |
| v5 | + Mixup augmentation | ⏳ |

---

## Como Usar

1. Editar notebook com melhoria desejada
2. Upload no Kaggle
3. Submeter
4. Atualizar `TODO.md` com resultado
