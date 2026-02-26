# Resubmit - Modelos Top 5 (Score > 0.7)

## Objetivo
Melhorar os melhores modelos baseline com técnicas para lidar com o **desbalanceamento de classes**.

**Limite:** 5 submissões/dia

---

## 🏆 Melhor Modelo: BERTimbau + Focal Loss v2 (0.79505)

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
| LightGBM v2 | `resubmit_lgbm_v2.ipynb` | 0.60915 | ✅ |
| LinearSVC v2 | `resubmit_linearsvc_v2.ipynb` | 0.77885 | ✅ |
| LogReg v2 | `resubmit_logreg_v2.ipynb` | 0.72935 | ✅ |
| SGD v2 | `resubmit_sgd_v2.ipynb` | 0.75019 | ✅ |
| Qwen3 Zero-Shot | `resubmit_qwen3_zeroshot.ipynb` | 0.13261 | ✅ |

### 2026-02-26 (0/5 submissões) - HOJE
| Modelo | Notebook | Score | Status |
|--------|----------|-------|--------|
| LightGBM v3 | `resubmit_lgbm_v3.ipynb` | - | ⏳ |
| LinearSVC v3 | `resubmit_linearsvc_v3.ipynb` | - | ⏳ |
| LogReg v3 | `resubmit_logreg_v3.ipynb` | - | ⏳ |
| SGD v3 | `resubmit_sgd_v3.ipynb` | - | ⏳ |
| Qwen3 One-Shot | `resubmit_qwen3_oneshot.ipynb` | - | ⏳ |

---

## ⚠️ Insights dos Resultados (25/02)

1. **LinearSVC v2 mantém top** - Score 0.77885 (igual baseline, sem degradação)
2. **SGD v2 estável** - Score 0.75019 (similar ao esperado)
3. **LogReg v2 estável** - Score 0.72935
4. **LightGBM v2 piorou** - Score 0.60915 (abaixo do baseline 0.70273)
5. **Qwen3 Zero-Shot MUITO ruim** - Score 0.13261 (LLM sem fine-tuning não funciona)
   - LLMs zero-shot não entendem o contexto médico específico
   - **Tentativa:** One-shot com exemplo no prompt

---

## ⚠️ Insights dos Resultados (24/02)

1. **BERTimbau v2 é estável** - Score próximo ao baseline (0.79505 vs 0.79696)
2. **v3 piorou** - Estratégias extras (alpha por classe, label smoothing) não ajudaram
3. **Modelos transformers quebraram** - Custom Transformer e BioBERTpt colapsaram
   - Possível overfitting ou instabilidade no fine-tuning
4. **Ensemble regrediu** - Adicionar SGD e ajustar pesos não melhorou

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
