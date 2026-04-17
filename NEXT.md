# NEXT.md - Próximos Passos

> Atualizado 2026-04-17 apos regressao do AWP+Optuna (0.77969)

## 🚨 Diagnostico da rodada 2026-04-17

**AWP + Optuna 300 trials = 0.77969** (regressao de 6pp vs winner 0.84027).

| Causa | Impacto |
|-------|---------|
| Falta `build_input_text` (texto raw) | -3 a -5 pp |
| Optuna overfitou OOF | -1 a -2 pp |
| Tripla regularizacao | -0.5 a -1 pp |
| AWP_LR=1e-1 excessivo | -0.5 a -1 pp |

---

## 🎯 Nova Estrategia: Voltar ao basico

**Parar de inventar combinacoes.** Primeiro reproduzir o winner, depois inovar isoladamente.

### Plano de 4 submissoes (2026-04-18)

| # | Notebook | Tecnica | Meta |
|---|----------|---------|------|
| 1 | `submit_winner_reproduce.ipynb` | Winner exato (controle) | ~0.84 |
| 2 | `submit_winner_pseudolabel.ipynb` | Winner + pseudo-labeling conf>0.95 | +0.5 a +1 pp |
| 3 | `submit_winner_msdropout.ipynb` | Winner + Multi-Sample Dropout 5x | +0.2 a +0.5 pp |
| 4 | `submit_winner_ensemble.ipynb` | Media das probs de 1+2+3 | +0.3 a +0.8 pp |

### Regras desta rodada

- ✅ SEMPRE usar `build_input_text` (extracao de Indicacao/Achados/Comparativa)
- ✅ Thresholds fixos do winner: `[0.95, 1.6, 1.35, 1.0, 0.4, 1.15, 0.1]`
- ✅ Temperatura fixa: T=0.958
- ✅ Focal Loss γ=2.0, α=0.25
- ❌ SEM AWP
- ❌ SEM Optuna
- ❌ SEM label smoothing (redundante com focal)
- ❌ SEM texto raw

### Se submissao 1 nao atingir ~0.84

**Tem bug no pipeline.** Investigar:
- Seed do StratifiedKFold (`random_state=42`)
- Ordem das colunas no tokenizer output
- `token_type_ids` presente/ausente
- FP16 vs FP32 (BERTimbau large pode dar NaN em FP16)

---

## Historico (antes de 2026-04-17)

## Estado Atual

| Métrica | Valor |
|---------|-------|
| 🏆 Melhor Score | **0.84027** (BERTimbau 5-Fold + MAX_LEN=512) |
| 🥈 2º Melhor | **0.81301** (BERTimbau Threshold v3) |
| 3º Melhor | 0.81213 (BERTimbau Raw Data v9) |
| 4º Melhor | 0.81100 (BERTimbau v5 + Label Smoothing) |
| 5º Melhor | 0.80950 (BERTimbau 5-Fold v11) |
| Baseline TF-IDF | 0.77885 |
| Total Submissões | 55+ |

---

## 🔥 Descoberta Chave: MAX_LEN=512

O notebook `resubmit_bertimbau_maxlen512_v0` alcançou **0.84027** usando:
- 5-Fold Ensemble
- MAX_LEN=512 (vs 192 anterior)
- model-v4 (pesos recentes)
- Calibração com thresholds

**Próximos passos:** Combinar MAX_LEN=512 com Label Smoothing e Threshold Tuning otimizado.

---

## ❌ O Que Evitar

| Técnica | Por que falhou |
|---------|----------------|
| LoRA offline | Não funciona no Kaggle |
| LLMs zero/one-shot | Não entendem contexto médico |
| SMOTE | v4/v5 regrediram |
| Tratamento de texto | v5 com normalização piorou -2% |
| Seed Ensemble v5 | Score 0.72135 (pior que v4) |
| MAX_LEN=512 | Timeout, relatórios são curtos (~100 tokens) |
| Iterar sobre sucesso | SGD v4/v5 pioraram vs v3 |

---

## 📊 Resultados Recentes (2026-03-09)

| Notebook | Score | Resultado |
|----------|-------|----------|
| BERTimbau v5 + Label Smoothing | **0.81100** | ✅ 2º melhor! |
| Super Ensemble v3 + Threshold | 0.78660 | ✅ Top 5 |
| BERTimbau v5 Ensemble Threshold | 0.77385 | ✅ Acima baseline |
| BERTimbau v5 Seed Ensemble (08/03) | 0.72135 | ❌ Abaixo do v4 |

**Conclusão:** Label Smoothing com v5 quase igualou o v4! Próximo passo: combinar Label Smoothing + Threshold Tuning.

---

## 🧪 Backlog de Experimentos

### Alta Prioridade (BERTimbau v5)
| Notebook | Técnica | Risco | Status |
|----------|---------|-------|--------|
| `bertimbau_v5_label_smooth_threshold` | Label Smoothing + Threshold Tuning | Baixo | 🔥 PRÓXIMO |
| `alpha_weights` | α=0.3, 0.4 no Focal Loss | Baixo | Pendente |
| `gamma_search` | γ=1.5, 2.5, 3.0 | Baixo | ❌ 0.75574 |
| `seed_ensemble` | 5 seeds | Baixo | ❌ 0.72135 |
| `lr_search` | LR=1e-5, 3e-5 | Baixo | ❌ 0.75508 |

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
