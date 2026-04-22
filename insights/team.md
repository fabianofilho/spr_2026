# Trabalho do Team (Augusto Serpa)

Repositorio clonado em `team/` (nao versionado). Fonte: https://github.com/AugustoSSerpa/spr-2026-mammography-challenge

---

## Stack tecnico

| Aspecto | Augusto | Meu (winner 0.84027) |
|---------|---------|----------------------|
| Framework | HuggingFace Trainer + YAML + wandb | PyTorch puro |
| Modelo base | `neuralmind/bert-base-portuguese-cased` (BASE) | BERTimbau Large |
| MAX_LEN | 256 | 512 |
| Batch size | 16 (7-classes), 32 (binario) | 8 |
| Epochs | 50 + early stopping patience=5 | 5 fixos |
| LR | 1e-5 | 2e-5 |
| Loss | weighted_cross_entropy (class_weight auto) | Focal Loss (γ=2.0, α=0.25) |
| Validacao | train_test_split 80/20 estratificado | 5-fold StratifiedKFold |
| Calibracao | Threshold binario 0.3 para override 2→3 | Temperatura + thresholds por classe |

---

## Ideia central: Cascata BI-RADS 2 vs 3

O trabalho do Augusto tem uma tecnica que **nao existe no meu pipeline**: um classificador **binario especializado** para distinguir apenas BI-RADS 2 (benigno) de BI-RADS 3 (provavelmente benigno).

### Pipeline de inferencia (cascata)

1. Roda o modelo 7-classes BERTimbau base
2. Identifica candidatos de override:
   - **Tipo A:** `pred == 2` e `2a_classe == 3` → possivel upgrade 2→3
   - **Tipo B:** `pred == 3` e `2a_classe == 2` → possivel downgrade 3→2
3. Para candidatos, consulta classificador binario 2v3
4. Se `P(BI-RADS 3) >= 0.3`, faz override para 3 (e vice-versa para downgrade)

### Por que isso e interessante

**BI-RADS 2 e 3 sao clinicamente proximos.** A diferenca e sutil (achado conclusivamente benigno vs achado provavelmente benigno). Essa confusao provavelmente e o maior ponto de erro do modelo 7-classes. Um modelo especializado nessa fronteira tem 2 vantagens:

1. **Mais capacidade alocada ao problema dificil.** Em vez de espalhar capacidade em 7 classes, concentra em 2.
2. **Class weight auto-balanceado em amostra 2+3 apenas** eleva o recall da classe 3 (minoritaria).

### Config do modelo 2v3

```yaml
model: bert-base-portuguese-cased
max_length: 256
batch_size: 32
lr: 1e-5
weight_decay: 0.01
loss: weighted_cross_entropy
class_weights: auto
inference.threshold_override_to_3: 0.3
```

---

## Notebooks do Augusto

| Notebook | Objetivo |
|----------|----------|
| `analise_laudos.ipynb` | EDA: distribuicao de classes, comprimento, palavras-chave por classe, padroes de secoes |
| `analise_erros.ipynb` | Analise de confusao do modelo 7-classes |
| `analise_erros_23.ipynb` | Analise especifica do binario 2v3, simulacao de thresholds |
| `submissao_kaggle.ipynb` | Submissao do modelo 7-classes puro |
| `submissao_bertimbau_23.ipynb` | Submissao com pipeline cascata (override bidirecional) |
| `submissao_ensemble_melhor_publico.ipynb` | Ensemble (arquivo com conflito git nao resolvido) |

---

## Oportunidades de combinacao

### 1. Cascata 2v3 em cima do winner (prioridade MAXIMA)

Treinar o binario 2v3 com **BERTimbau Large + Focal Loss + MAX_LEN=512** (stack do winner) em vez do base. Aplicar override bidirecional no output do winner 5-fold. Hipotese: se os erros do winner concentram em 2↔3, isso sozinho pode dar +0.5 a +1.5 pp.

**Acao:** criar `submit_winner_cascade_2v3.ipynb`.

### 2. Cascata multipla (fronteiras criticas)

Alem de 2v3, as fronteiras **0 vs 1** (exame incompleto vs negativo) e **4 vs 5** (suspeito vs altamente suspeito) tambem podem ser duvidosas clinicamente. Treinar binarios adicionais.

### 3. Blend de arquiteturas

Ensemble **probabilistico**:
```
final = 0.5 * winner_large_5fold + 0.5 * bertimbau_base_8020
```

O modelo base do Augusto provavelmente tem correlacao baixa com o Large pelos hiperparametros diferentes (LR menor, loss diferente, max_len menor, split diferente). **Diversidade entre modelos = ganho em ensemble.**

### 4. Stacking com meta-learner

Usar probabilidades do winner + probabilidades do Augusto como features (14 dim) num LightGBM meta-learner treinado em OOF.

---

## Arquivos importantes (caminhos em team/)

- `team/config.yaml`: hiperparametros 7-classes
- `team/config_birads23.yaml`: hiperparametros binario 2v3
- `team/src/trainer.py`: Custom Trainer com Focal/Weighted CE
- `team/src/birads_23/trainer.py`: trainer binario
- `team/src/birads_23/predict.py`: `apply_23_override()` funcao de pos-processamento
- `team/notebooks/analise_laudos.ipynb`: EDA valioso
- `team/notebooks/submissao_bertimbau_23.ipynb`: implementacao da cascata

---

*Clonado em 22/04/2026*
