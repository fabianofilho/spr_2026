# NEXT.md - Próximos Passos

> Baseado em 45+ submissões (Fev/2026)

## Estado Atual

| Métrica | Valor |
|---------|-------|
| 🏆 Melhor Score | **0.82073** (BERTimbau v4 + Threshold Tuning) |
| 2º Melhor | 0.79696 (BERTimbau + Focal Loss) |
| 3º Melhor | 0.78729 (Super Ensemble v1) |
| Baseline TF-IDF | 0.77885 |
| Total Submissões | 45+ |

---

## 🚀 Pasta 2026-03-01 (Rodar Amanhã)

### Ultimate v6 - PRIORIDADE MÁXIMA
```python
# Combina TODAS as técnicas validadas:
SEEDS = [42, 123, 456]      # 3-seed ensemble
N_FOLDS = 5                  # Cross-validation
FOCAL_GAMMA = 2.0            # Focal Loss
FOCAL_ALPHA = 0.25
# + Grid Search Thresholds por classe
```
**Expectativa:** >0.82073, meta 0.83+

### Threshold Grid v5
Grid search mais fino de thresholds (step=0.02).

### CV Thresholds v5
Thresholds estimados via média de 5 folds para estabilidade.

### Qwen + BI-RADS Instruction
LLM com prompt detalhado das categorias BI-RADS (experimental).

---

## ❌ O Que Evitar

| Técnica | Por que falhou |
|---------|----------------|
| LoRA offline | Não funciona no Kaggle |
| LLMs zero/one-shot | Não entendem contexto médico |
| SMOTE | v4/v5 regrediram |
| Tratamento de texto | v5 com normalização piorou -2% |
| Label Smoothing alto | Prejudica threshold tuning |
| MAX_LEN=512 | Timeout, relatórios são curtos (~100 tokens) |
| Iterar sobre sucesso | SGD v4/v5 pioraram vs v3 |

---

## 🧪 Backlog de Experimentos

### Alta Prioridade (BERTimbau v5)
| Notebook | Técnica | Risco |
|----------|---------|-------|
| `alpha_weights` | α=0.3, 0.4 no Focal Loss | Baixo |
| `gamma_search` | γ=1.5, 2.5, 3.0 | Baixo |
| `seed_ensemble` | 5 seeds | Baixo |
| `lr_search` | LR=1e-5, 3e-5 | Baixo |

### LLMs Médicos (Experimental)
| Notebook | Modelo | Tamanho |
|----------|--------|---------|
| `qwen_birads` | Qwen 2.5 1.5B | 1.5B |
| `medgemma_birads` | MedGemma 4B | 4B |
| `medgemma_27b` | MedGemma 27B Text IT | 27B |
| `biogpt_large` | BioGPT Large | 1.5B |
| `clinicalbert` | ClinicalBERT (fine-tune) | 110M |

---

## 🔑 Configuração Vencedora (Referência)

```python
# BERTimbau v4 = 0.82073
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
SEED = 42

# Thresholds
THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.30, 6: 0.25}
```
