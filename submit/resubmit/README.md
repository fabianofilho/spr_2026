# Resubmit - Modelos Top 5 (Score > 0.7)

## Objetivo
Melhorar os melhores modelos baseline com técnicas para lidar com o **desbalanceamento de classes**.

---

## 🏆 Melhor Modelo: BERTimbau + Focal Loss (0.79696)

| Versão | Notebook | Estratégias | Status |
|--------|----------|-------------|--------|
| v3 | `resubmit_bertimbau_focal_v3.ipynb` | Alpha por classe + Threshold tuning + Label smoothing | ⏳ |

---

## 🆕 Novos Resubmits (Top 4 - Fev/2026)

| Modelo Base | Score | Notebook v2 | Estratégia |
|-------------|-------|-------------|------------|
| Ensemble Soft Voting | 0.78049 | `resubmit_ensemble_voting_v2.ipynb` | +SGD (3 modelos), pesos otimizados |
| Custom Transformer | 0.77272 | `resubmit_custom_transformer_v2.ipynb` | **Focal Loss** + label smoothing + warmup |
| BioBERTpt | 0.72480 | `resubmit_biobertpt_focal_v2.ipynb` | **Focal Loss** (γ=2) + gradient accumulation |

### Melhorias Implementadas:

**Ensemble Soft Voting v2:**
- Adiciona SGDClassifier ao voting (3 modelos vs 2)
- Pesos baseados nos F1-Scores públicos: SVC=0.345, SGD=0.332, LR=0.323
- TF-IDF com 25k features (vs 20k)

**Custom Transformer v2:**
- Focal Loss (γ=2) - mesma estratégia do BERTimbau campeão
- Label smoothing (0.1) para regularização
- Learning rate scheduler com warmup
- 15 epochs (vs 10)

**BioBERTpt v2:**
- Focal Loss (γ=2) com class weights
- Gradient accumulation (2 steps)
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
