# Backlog - Notebooks para Submissões Futuras

Notebooks baseados na arquitetura campeã: **BERTimbau v4 Threshold Tuning (0.82073)** 🏆

---

## 🎯 Prioridade Alta (BERTimbau v5 Series)

| Notebook | Melhoria | Hipótese |
|----------|----------|----------|
| `resubmit_bertimbau_v5_threshold_grid.ipynb` | Grid search fino | Thresholds mais precisos (+0.5-1%) |
| `resubmit_bertimbau_v5_cv_thresholds.ipynb` | CV para thresholds | Mais estabilidade |
| `resubmit_bertimbau_v5_alpha_weights.ipynb` | α adaptativo por classe | Foco em minoritárias |
| `resubmit_super_ensemble_v3_threshold.ipynb` | Super Ensemble + Threshold | Combinar tudo |

## 🔬 Prioridade Média (Hyperparameter Search)

| Notebook | Melhoria | Valores |
|----------|----------|---------|
| `resubmit_bertimbau_v5_gamma_search.ipynb` | Testar γ Focal Loss | 1.0, 1.5, 2.0, 2.5, 3.0 |
| `resubmit_bertimbau_v5_lr_search.ipynb` | Learning rate | 1e-5, 2e-5, 3e-5, 5e-5 |
| `resubmit_bertimbau_v5_maxlen_512.ipynb` | Sequências maiores | 512 tokens |
| `resubmit_bertimbau_v5_label_smoothing.ipynb` | Regularização | ε = 0.1 |

## 🔄 Prioridade Baixa (Ensemble Variations)

| Notebook | Melhoria | Descrição |
|----------|----------|-----------|
| `resubmit_bertimbau_v5_seed_ensemble.ipynb` | Multi-seed | 5 seeds diferentes |
| `resubmit_bertimbau_v5_ensemble_threshold.ipynb` | Multi-model | BERTimbau + BioBERTpt |

## 🤖 LLMs (BI-RADS Instruction)

| Notebook | Modelo | Características |
|----------|--------|-----------------|
| `resubmit_qwen_birads_instruction.ipynb` | Qwen 1.5B | Prompt PT-BR |
| `resubmit_medgemma_birads_instruction.ipynb` | MedGemma (geral) | Google médico |
| `resubmit_medgemma_1_5_4b_it.ipynb` | MedGemma 1.5 4B | Multimodal, reasoning |
| `resubmit_medgemma_4b_it.ipynb` | MedGemma 4B IT | Texto/visão clínico |
| `resubmit_medgemma_27b_text_it.ipynb` | MedGemma 27B | Grande, quantizado 4-bit |
| `resubmit_medmo_birads_instruction.ipynb` | MedMO 4B/8B | MBZUAI médico |
| `resubmit_biogpt_large_birads.ipynb` | BioGPT Large | Microsoft biomédico |
| `resubmit_clinicalbert_finetune.ipynb` | ClinicalBERT | Fine-tuning encoder |

---

## 📊 Arquitetura Base (BERTimbau v4)

```python
# Config campeã
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.30, 6: 0.25}
```

## 🏆 Score Atual
- **BERTimbau v4 (Threshold Tuning): 0.82073** (+3% sobre baseline)
