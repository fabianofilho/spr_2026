# Validação: BERTimbau Large

**Modelo:** BERTimbau Large (`neuralmind/bert-large-portuguese-cased`)

---

## 📊 Histórico
- BERTimbau BASE + Focal = **0.79696**
- BERTimbau v4 (threshold tuning) = **0.82073**
- Large NUNCA testado sistematicamente

## 🎯 Objetivo
Testar se o modelo LARGE supera o base.

## 📊 Configurações Testadas
- Large vs Base (comparação direta)
- Focal Loss γ=2,3
- Learning rate 1e-5 vs 2e-5
- Threshold Tuning

---

## Ambiente
- **Device:** cuda
- **Model:** `/kaggle/input/models/fabianofilho/bertimbau-ptbr-complete/pytorch/default/1`
- **Train:** 14,617 | **Val:** 3,655

---

## Experimentos

### 1. CrossEntropy (Baseline)

**Config:** `lr=2e-5, batch_size=4, max_length=256, grad_accum=2`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | 2.269227 | 0.456333 | 0.404461 |
| 2 | 0.676170 | 0.318769 | 0.544779 |
| 3 | 0.534280 | 0.314676 | 0.609128 |
| 4 | 0.374651 | 0.332033 | 0.688748 |
| 5 | 0.291792 | 0.374913 | 0.694080 |

**Resultado:** F1-Macro = **0.69408** | Tempo: 48.3 min

```
              precision    recall  f1-score   support
           0       0.82      0.80      0.81       122
           1       0.94      0.95      0.94       138
           2       0.98      0.99      0.98      3194
           3       0.65      0.59      0.62       143
           4       0.76      0.74      0.75        43
           5       0.00      0.00      0.00         6
           6       0.86      0.67      0.75         9
```

---

### 2. Focal Loss γ=2.0

**Config:** `lr=2e-5, batch_size=4, max_length=256, grad_accum=2, focal_gamma=2.0`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | 0.156257 | 0.024598 | 0.400220 |
| 2 | 0.040472 | 0.017143 | 0.560873 |
| 3 | 0.025082 | 0.014434 | 0.652111 |
| 4 | 0.015951 | 0.014861 | 0.711486 |
| 5 | 0.010231 | 0.015118 | 0.722288 |

**Resultado:** F1-Macro = **0.72229** | Tempo: 48.5 min

```
              precision    recall  f1-score   support
           0       0.86      0.80      0.83       122
           1       0.96      0.95      0.96       138
           2       0.98      0.99      0.98      3194
           3       0.64      0.63      0.63       143
           4       0.78      0.74      0.76        43
           5       0.33      0.17      0.22         6
           6       0.83      0.56      0.67         9
```

---

### 3. Focal Loss γ=3.0

**Config:** `lr=2e-5, batch_size=4, max_length=256, grad_accum=2, focal_gamma=3.0`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | 0.124797 | 0.018906 | 0.411974 |
| 2 | 0.031225 | 0.013563 | 0.513529 |
| 3 | 0.018480 | 0.010450 | 0.663928 |
| 4 | 0.010985 | 0.010948 | 0.669148 |
| 5 | 0.007016 | 0.010748 | 0.715082 |

**Resultado:** F1-Macro = **0.71508** | Tempo: 48.5 min

```
              precision    recall  f1-score   support
           0       0.87      0.78      0.82       122
           1       0.95      0.95      0.95       138
           2       0.98      0.99      0.98      3194
           3       0.60      0.62      0.61       143
           4       0.72      0.65      0.68        43
           5       0.50      0.17      0.25         6
           6       0.75      0.67      0.71         9
```

---

### 4. Focal Loss γ=2.0 + LR=1e-5

**Config:** `lr=1e-5, batch_size=4, max_length=256, grad_accum=2, focal_gamma=2.0`

| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | 0.188834 | 0.025139 | 0.390835 |
| 2 | 0.042541 | 0.019256 | 0.510011 |
| 3 | 0.028286 | 0.015417 | 0.672177 |
| 4 | 0.018502 | 0.016352 | 0.636904 |
| 5 | 0.014359 | 0.015535 | 0.713161 |

**Resultado:** F1-Macro = **0.71316** | Tempo: 48.5 min

```
              precision    recall  f1-score   support
           0       0.87      0.79      0.83       122
           1       0.94      0.94      0.94       138
           2       0.98      0.99      0.98      3194
           3       0.61      0.61      0.61       143
           4       0.74      0.60      0.67        43
           5       0.50      0.17      0.25         6
           6       1.00      0.56      0.71         9
```

---

## Threshold Tuning (melhor modelo: Focal γ=2)

| Classe | Threshold |
|--------|-----------|
| 0 | 0.35 |
| 1 | 0.50 |
| 2 | 0.50 |
| 3 | 0.45 |
| 4 | 0.45 |
| 5 | 0.30 |
| 6 | 0.25 |

**F1 com Thresholds:** **0.73976**

---

## 📊 RESUMO

| Config | F1-Macro |
|--------|----------|
| **+ Thresholds** | **0.73976** |
| Large_Focal_g2 | 0.72229 |
| Large_Focal_g3 | 0.71508 |
| Large_Focal_lr1e5 | 0.71316 |
| Large_CE | 0.69408 |

### Comparação com Base
| Modelo | F1-Macro |
|--------|----------|
| **BERTimbau Base v4** | **0.82073** |
| BERTimbau Base + Focal | 0.79696 |
| BERTimbau Large + Thresholds | 0.73976 |

---

## 📝 INSIGHTS

1. **Large vs Base:**
   - ❌ **Large PERDEU para Base** (0.74 vs 0.82)
   - Large tem 335M vs 110M parâmetros
   - Overfitting mais severo em dataset pequeno
   - Batch menor (4 vs 8) prejudica estabilidade

2. **Focal Loss:**
   - ✅ Focal γ=2 foi o melhor (0.722 vs 0.694 CE)
   - Ganho de +2.8% com Focal Loss
   - γ=3 muito agressivo para Large

3. **Threshold Tuning:**
   - ✅ Ganho adicional de +1.7% (0.722 → 0.740)
   - Classes raras precisam thresholds baixos (5: 0.30, 6: 0.25)

4. **⚠️ PROBLEMA IDENTIFICADO:**
   - Missing keys em LayerNorm (beta/gamma vs weight/bias)
   - Checkpoint pode estar corrompido ou incompatível
   - Modelo não carregou corretamente os pesos do encoder!

5. **Recomendação:**
   - ❌ **NÃO usar BERTimbau Large** para esta competição
   - ✅ Manter **BERTimbau Base v4** como melhor modelo
   - Large precisa de mais dados para justificar complexidade
