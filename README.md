# SPR 2026 Mammography Report Classification

Classificação de relatórios de mamografia em categorias BI-RADS (0-6).

- **Competição:** [SPR 2026](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification)
- **Métrica:** F1-Score Macro
- **Melhor Score:** **0.82073** (BERTimbau v4 + Focal Loss + Threshold Tuning)

## Leaderboard (Top 5)

| Rank | Modelo | Score |
|------|--------|-------|
| 🥇 | **BERTimbau v4 (Threshold Tuning)** | **0.82073** |
| 🥈 | BERTimbau + Focal Loss | 0.79696 |
| 🥉 | Super Ensemble v1 | 0.78729 |
| 4 | Ensemble Soft Voting | 0.78049 |
| 5 | TF-IDF + LinearSVC | 0.77885 |

> Ver [TODO.md](TODO.md) para lista completa de 40+ submissões.

## Estrutura

```
submit/           # Notebooks para Kaggle (versionado)
submissions/      # Cópias submetidas (local)
insights/         # Análises metodológicas
skills/           # Instruções para o agente
models/           # Scripts de download
tests/            # Experimentos
```

## Quick Start (Kaggle)

1. Upload notebook de `submit/`
2. Add Input → selecionar modelo/dataset
3. Execute

## Insights

- [transformers.md](insights/transformers.md) - BERTimbau domina
- [tfidf.md](insights/tfidf.md) - Baseline forte
- [ensemble.md](insights/ensemble.md) - Soft Voting eficaz
- [NEXT.md](NEXT.md) - Próximas estratégias

## Técnicas que Funcionam

1. **BERTimbau + Focal Loss + Threshold Tuning** → 0.82073 (+5.4% vs baseline)
2. **Focal Loss** (γ=2.0, α=0.25) → essencial para desbalanceamento
3. **Threshold Tuning** por classe → +3% F1 adicional
4. **Seed Ensemble** (3 seeds) → +0.5-1% estabilidade

## Técnicas que NÃO Funcionam

- **SMOTE** → regrediu o score
- **LLMs zero/one-shot** → não entendem contexto médico
- **Tratamento de texto pesado** → -2% F1
