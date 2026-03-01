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

| Prioridade | Experimento | Hipótese |
|------------|-------------|----------|
| 🔴 Alta | BERTimbau v4 Threshold Tuning | Ajustar thresholds pode melhorar F1-Macro |
| 🟡 Média | Ensemble v4 (pesos otimizados) | Otimizar pesos do Super Ensemble |
| 🟢 Baixa | Focal Loss em XLM-RoBERTa | Replicar sucesso do BERTimbau |

---

## Lições Aprendidas

### ✅ O que funciona
- **Threshold Tuning** por classe (+3% no BERTimbau v4!) 🔥
- **BERTimbau** > modelos multilingual
- **Focal Loss** (γ=2) para classes desbalanceadas
- **Soft Voting** entre modelos diversos

### ❌ O que não funciona
- **LoRA offline** no Kaggle
- **LLMs zero/one-shot** para este problema
- **SMOTE** com v4/v5 (todas regrediram)
- **Tratamento de texto** com v5 (piorou scores)
- **Muitas alterações** de uma vez (3/5 falharam)

---

## Tracking de Submissões

| Mês | Total | Melhorias | Regressões | Falhas |
|-----|-------|-----------|------------|--------|
| Fev | 42+ | 2 | 10 | 7 |

**Taxa de sucesso:** ~20% das resubmissões melhoraram ou mantiveram
