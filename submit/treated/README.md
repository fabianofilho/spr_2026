# TF-IDF com Tratamento de Texto

## Motivação

> "Acho que tem que tratar os dados muito bem antes. Os laudos são muito parecidos. Tem o glossário do BI-RADS." - Eduardo Farina (22/02/2026)

## Tratamento Aplicado

1. **Normalização de texto**
   - Lowercase
   - Remoção de acentos (`calcificação` → `calcificacao`)
   - Remoção de caracteres especiais

2. **Stop words customizadas**
   - Remove pronomes, preposições, artigos
   - **Preserva** termos médicos e BI-RADS

3. **Preservação de vocabulário BI-RADS**
   - `birads`, `categoria`, `calcificacao`, `nodulo`, `massa`
   - `benigno`, `maligno`, `suspeito`, `biopsia`
   - Termos morfológicos: `espiculado`, `circunscrito`, `oval`, etc.

4. **Normalização BI-RADS**
   - `BI-RADS`, `bi rads`, `birads` → `birads`

## Notebooks

| Notebook | Modelo Base | Score Original | Status |
|----------|-------------|----------------|--------|
| submit_treated_linearsvc.ipynb | LinearSVC | 0.77885 | ⏳ A submeter |
| submit_treated_sgd.ipynb | SGDClassifier | 0.75019 | ⏳ A submeter |
| submit_treated_logreg.ipynb | LogisticRegression | 0.72935 | ⏳ A submeter |
| submit_treated_lgbm.ipynb | LightGBM | 0.70273 | ⏳ A submeter |
| submit_treated_xgboost.ipynb | XGBoost | 0.69482 | ⏳ A submeter |

## Hipótese

Laudos muito similares → tratamento cuidadoso pode:
- Reduzir ruído (stop words, caracteres especiais)
- Destacar termos discriminativos (BI-RADS)
- Melhorar separação entre categorias

## Próximos Passos

- [ ] Submeter notebooks e comparar com baseline
- [ ] Se melhorar: expandir glossário BI-RADS
- [ ] Se piorar: testar tratamento mais conservador
