# Validação: DistilBERT Multilingual

**Modelo:** DistilBERT Multilingual (`distilbert-base-multilingual-cased`)

---

## 📊 Histórico
- Score anterior: **0.55229** (pior que BERT multilingual)
- Apenas testado com CrossEntropy

## 🎯 Objetivo
Testar se Focal Loss melhora DistilBERT para uso em ensembles.

## 📊 Vantagens
- 40% mais rápido que BERT full
- 60% menor
- Pode ser útil em ensembles rápidos

---

## Ambiente
- **Device:** cuda
- **Model:** `/kaggle/input/models/fabianofilho/distilbert-multilingual-base-ff/pytorch/default/1/distilbert-multilingual`
- **Train:** 14,617 | **Val:** 3,655

---

## Experimentos

### 1. CrossEntropy (Baseline)

**Config:** `lr=2e-5, batch_size=16, max_length=256`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | - | 0.479555 | 0.345001 |
| 2 | 0.966525 | 0.392219 | 0.428633 |
| 3 | 0.353652 | 0.299228 | 0.550548 |
| 4 | 0.268554 | 0.304691 | 0.556865 |
| 5 | 0.201357 | 0.312368 | 0.592671 |

**Resultado:** F1-Macro = **0.59267** | Tempo: 20.5 min

---

### 2. Focal Loss γ=2.0

**Config:** `lr=2e-5, batch_size=16, max_length=256, focal_gamma=2.0`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | - | 0.030777 | 0.348481 |
| 2 | 0.072534 | 0.022323 | 0.452379 |
| 3 | 0.023021 | 0.018751 | 0.536635 |
| 4 | 0.017249 | 0.018441 | 0.541592 |
| 5 | 0.012048 | 0.017524 | 0.549557 |

**Resultado:** F1-Macro = **0.54956** | Tempo: 20.8 min

---

### 3. Focal Loss γ=3.0

**Config:** `lr=2e-5, batch_size=16, max_length=256, focal_gamma=3.0`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | - | 0.024144 | 0.358405 |
| 2 | 0.057724 | 0.016859 | 0.451458 |
| 3 | 0.017360 | 0.013497 | 0.538726 |
| 4 | 0.012296 | 0.013244 | 0.535301 |
| 5 | 0.008242 | 0.012479 | 0.546987 |

**Resultado:** F1-Macro = **0.54699** | Tempo: 20.8 min

---

### 4. Focal Loss γ=2.0 + LR=3e-5

**Config:** `lr=3e-5, batch_size=16, max_length=256, focal_gamma=2.0`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | - | 0.033556 | 0.364081 |
| 2 | 0.065913 | 0.020959 | 0.510159 |
| 3 | 0.020878 | 0.017100 | 0.540683 |
| 4 | 0.014823 | 0.017184 | 0.587497 |
| 5 | 0.009647 | 0.016839 | 0.584801 |

**Resultado:** F1-Macro = **0.58750** | Tempo: 20.8 min

---

## 📊 RESUMO

| Config | F1-Macro | Tempo (min) |
|--------|----------|-------------|
| **CE_baseline** | **0.59267** | 20.5 |
| Focal_lr3e5 | 0.58750 | 20.8 |
| Focal_g2 | 0.54956 | 20.8 |
| Focal_g3 | 0.54699 | 20.8 |

### Comparação
| Modelo | F1-Macro |
|--------|----------|
| BERTimbau v4 | **0.82073** |
| DistilBERT CE (novo) | 0.59267 |
| DistilBERT (anterior) | 0.55229 |

**Melhoria:** +4% sobre score anterior (0.55 → 0.59)

---

## 📝 INSIGHTS

1. **Focal Loss PIOROU Performance:**
   - ❌ CE (0.593) > Focal γ=2 (0.550) > Focal γ=3 (0.547)
   - Focal Loss não é universal - depende do modelo
   - DistilBERT tem capacidade limitada para classes raras

2. **CrossEntropy é Melhor:**
   - ✅ Melhor F1 com loss padrão
   - Modelo menor se beneficia de loss mais simples
   - Focal Loss pode overcompensar em modelos pequenos

3. **LR Maior Ajudou (parcialmente):**
   - Focal + LR=3e-5 (0.588) quase alcançou CE (0.593)
   - Mas ainda não superou baseline

4. **Velocidade:**
   - ⏱️ ~20 min (vs ~43 min BERTimbau)
   - ~2x mais rápido que BERT full

5. **⚠️ Problema de Checkpoint:**
   - Missing LayerNorm.weight/bias
   - Mesmo problema do BERTimbau Large
   - Pesos não carregados corretamente

6. **Recomendação:**
   - ❌ **NÃO usar DistilBERT** para esta competição
   - Gap de 23 pontos para BERTimbau (0.59 vs 0.82)
   - Velocidade não compensa a perda de qualidade
   - Para ensemble: usar apenas se diversidade > accuracy
