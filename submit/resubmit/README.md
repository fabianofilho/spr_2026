# Resubmit - Modelos Top 4 (Score > 0.7)

## Objetivo
Melhorar os melhores modelos baseline com técnicas para lidar com o **desbalanceamento de classes**.

---

## Notebooks Disponíveis

### Versão 2 (Baseline com class_weight='balanced')
| Rank | Modelo | Score Base | Notebook |
|------|--------|------------|----------|
| 1 | LinearSVC | 0.77885 | `resubmit_linearsvc_v2.ipynb` |
| 2 | SGDClassifier | 0.75019 | `resubmit_sgd_v2.ipynb` |
| 3 | Logistic Regression | 0.72935 | `resubmit_logreg_v2.ipynb` |
| 4 | LightGBM | 0.70273 | `resubmit_lgbm_v2.ipynb` |

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

---

## Como Usar

1. Editar notebook com melhoria desejada
2. Upload no Kaggle
3. Submeter
4. Atualizar `TODO.md` com resultado
