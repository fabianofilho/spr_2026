# SPR 2026 Mammography Report Classification

Classificação de relatórios de mamografia em categorias BI-RADS (0-6).

- **Competição:** [SPR 2026](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification)
- **Métrica:** F1-Score Macro
- **Melhor Score:** **0.79696** (BERTimbau + Focal Loss)

## Leaderboard (Top 5)

| Rank | Modelo | Score |
|------|--------|-------|
| 🥇 | BERTimbau + Focal Loss | **0.79696** |
| 🥈 | Super Ensemble v1 | 0.78729 |
| 🥉 | Ensemble Soft Voting | 0.78049 |
| 4 | TF-IDF + LinearSVC | 0.77885 |
| 5 | Custom Transformer | 0.77272 |

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

## Lições Principais

1. **BERTimbau + Focal Loss** é o padrão ouro
2. **Modelos PT nativos** > multilingual
3. **Soft Voting** supera stacking complexo
4. **Resubmissões são arriscadas** - apenas 1 em 5 melhora
