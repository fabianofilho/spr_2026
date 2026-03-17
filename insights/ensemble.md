# Ensemble - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| � | **Super Ensemble v1** | **0.78729** | ✅ Submetido |
| 🥈 | **Super Ensemble v3 + Threshold** | **0.78660** | ✅ Submetido |
| 3 | **VotingClassifier soft** | **0.78049** | ✅ Submetido |
| 4 | BERTimbau v5 Ensemble Threshold | 0.77385 | ✅ Submetido |
| 5 | TF-IDF + W2V voting | 0.74667 | ✅ Submetido |
| 6 | Stacking Meta-Learner | 0.73852 | ✅ Submetido |

---

## 🏆 Análise: Soft Voting (0.78049) - 2º MELHOR OVERALL!

**Ensemble com Soft Voting superou TF-IDF baseline!** Score de **0.78049** coloca como 2º melhor modelo geral.

### Por que funcionou?

1. **Erros descorrelacionados:** Cada modelo erra em exemplos diferentes
2. **Soft voting:** Média de probabilidades é mais robusta que hard voting
3. **Top 3 TF-IDF:** Combina os melhores modelos da categoria
4. **Mesma representação:** Todos usam TF-IDF, garantindo consistência

### Composição

```python
VotingClassifier([
    ('tfidf_svc', LinearSVC),      # 0.77885 (líder TF-IDF)
    ('tfidf_sgd', SGDClassifier),  # 0.75019 
    ('tfidf_lr', LogisticRegression),  # 0.72935
], voting='soft')
```

### Por que supera modelos individuais?

| Modelo | Score | Erro |
|--------|-------|------|
| LinearSVC | 0.77885 | Erra exemplos ambíguos |
| SGDClassifier | 0.75019 | Regularização diferente |
| LogReg | 0.72935 | Probabilidades calibradas |
| **Ensemble** | **0.78049** | Vota majoritariamente certo |

**Ganho:** +0.16% sobre o melhor modelo individual

---

## Análise: TF-IDF + W2V Voting (0.74667)

Combinação de TF-IDF com Word2Vec teve resultado misto.

### Por que não superou TF-IDF puro?

1. **Word2Vec adiciona ruído:** Pior nas classes críticas
2. **Representações incompatíveis:** TF-IDF sparse vs W2V dense
3. **Correlação de erros:** Ambos erram nos mesmos exemplos difíceis

### Lição aprendida

> **Ensemble só melhora se os modelos tiverem erros descorrelacionados.**
> Combinar TF-IDF + W2V é pior que TF-IDF + TF-IDF com classificadores diferentes.

---

## Análise: Stacking Meta-Learner (0.73852)

Stacking com meta-learner LogReg ficou abaixo do esperado.

### Por que não superou Soft Voting?

1. **Overfitting no meta-learner:** CV interno pode ter vazado
2. **Pouca diversidade:** Base learners muito similares
3. **Dataset pequeno:** Insuficiente para treinar meta-learner robusto

### Quando Stacking funciona melhor?

- Datasets maiores (>10k samples)
- Base learners muito diversos (ex: TF-IDF + Transformers + W2V)
- Meta-learner adequado (não-linear como XGBoost)

---

## 💡 Próximos Passos

### Ensemble com Transformers

Combinar BERTimbau + Focal (0.797) com TF-IDF Ensemble (0.780):

```python
# Proposta: Weighted Blend
final_pred = 0.6 * bertimbau_probs + 0.4 * tfidf_ensemble_probs
```

**Hipótese:** Pode superar 0.80 se erros forem complementares.

### Stacking com Transformers

Usar embeddings de BERTimbau como features para meta-learner:

```python
StackingClassifier(
    estimators=[bertimbau, linearsvc, biobertpt],
    final_estimator=XGBClassifier()
)
```

---

*Atualizado em: 24/02/2026*
