# SPR 2026 - TODO

## Leaderboard (Public Score)

| Rank | Modelo | Score | Data |
|------|--------|-------|------|
| 🏆 | **BERTimbau v4 (Threshold Tuning)** | **0.82073** | 28/02 |
| 2 | BERTimbau + Focal Loss | 0.79696 | 24/02 |
| 3 | Super Ensemble v1 | 0.78729 | 27/02 |
| 4 | Ensemble Soft Voting | 0.78049 | 24/02 |
| 5 | TF-IDF + LinearSVC | 0.77885 | 24/02 |
| 6 | Custom Transformer | 0.77272 | 24/02 |
| 7 | LinearSVC v4 | 0.77244 | 28/02 |
| 8 | SGDClassifier v3 | 0.77036 | 26/02 |
| 9 | Ensemble v3 | 0.76567 | 28/02 |
| 10 | SGDClassifier v4 | 0.76503 | 28/02 |

> Ver lista completa em [insights/README.md](insights/README.md)

---

## Próximos Experimentos

Ver [NEXT.md](NEXT.md) para estratégias detalhadas.

### Pasta 2026-03-01 (Prioridade Máxima)
| Notebook | Técnica | Expectativa |
|----------|---------|-------------|
| `resubmit_bertimbau_ultimate_v6` | 3-Seed + 5-Fold CV + Grid Search Thresholds | >0.82073 |
| `resubmit_bertimbau_v5_threshold_grid` | Grid search fino de thresholds | +0.5-1% |
| `resubmit_bertimbau_v5_cv_thresholds` | CV para thresholds estáveis | +0.5% |
| `resubmit_qwen_birads_instruction` | LLM com prompt BI-RADS | Experimental |

### Backlog de Alta Prioridade
| Notebook | Técnica |
|----------|---------|
| `resubmit_bertimbau_v5_alpha_weights` | Testar α=0.3, 0.4 no Focal Loss |
| `resubmit_bertimbau_v5_gamma_search` | Testar γ=1.5, 2.5, 3.0 |
| `resubmit_bertimbau_v5_seed_ensemble` | Ensemble de 5 seeds |

### LLMs Médicos (Experimental)
| Notebook | Modelo |
|----------|--------|
| `resubmit_medgemma_birads_instruction` | MedGemma 4B |
| `resubmit_medgemma_27b_text_it` | MedGemma 27B |
| `resubmit_biogpt_large_birads` | BioGPT Large |

---

## Lições Aprendidas

### ✅ O que funciona
- **Threshold Tuning** por classe (+3% no BERTimbau v4!) 🔥
- **Focal Loss** (γ=2.0, α=0.25) → essencial
- **Seed Ensemble** (3 seeds) → +0.5-1% robustez
- **5-Fold CV** → thresholds estáveis
- **BERTimbau** > modelos multilingual

### ❌ O que não funciona
- **LoRA offline** no Kaggle
- **LLMs zero/one-shot** para este problema
- **SMOTE** com v4/v5 (todas regrediram)
- **Tratamento de texto** com v5 (piorou -2%)
- **Label Smoothing alto** → prejudica threshold tuning
- **MAX_LEN=512** → timeout (relatórios são curtos)

---

## Tracking de Submissões

| Mês | Total | Melhorias | Regressões | Falhas |
|-----|-------|-----------|------------|--------|
| Fev | 42+ | 2 | 10 | 7 |

**Taxa de sucesso:** ~20% das resubmissões melhoraram ou mantiveram
