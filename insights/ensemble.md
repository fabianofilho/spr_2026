# Ensemble - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| - | TF-IDF + W2V voting | - | ⏳ Pendente |
| - | VotingClassifier soft | - | ⏳ Pendente |
| - | Stacking OOF | - | ⏳ Pendente |

---

## Estratégias

### 1. Voting (Soft)

Combinar predições probabilísticas de múltiplos modelos.

```python
VotingClassifier([
    ('tfidf_svc', tfidf_svc_pipeline),
    ('tfidf_sgd', tfidf_sgd_pipeline),
    ('tfidf_lr', tfidf_lr_pipeline),
], voting='soft')
```

**Candidatos (Top 3 TF-IDF):**
- LinearSVC (0.77885)
- SGDClassifier (0.75019)
- LogReg (0.72935)

### 2. Stacking

Meta-learner treinado nas predições OOF (out-of-fold).

```python
StackingClassifier(
    estimators=[...],
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'
)
```

### 3. Blending Manual

Média ponderada das probabilidades.

```python
# Pesos baseados em performance
weights = {
    'tfidf_svc': 0.5,
    'tfidf_sgd': 0.3,
    'tfidf_lr': 0.2,
}
final_prob = sum(w * pred for w, pred in zip(weights, preds))
```

---

## Hipóteses

### Pode funcionar se:
- Modelos têm **erros descorrelacionados**
- Combinação de TF-IDF + embeddings captura aspectos diferentes

### Pode falhar se:
- Todos os modelos erram nos mesmos exemplos
- Word2Vec adiciona mais ruído que sinal

---

## Análise (a ser preenchida após submissões)

*Aguardando resultados...*

---

*Atualizado em: 20/02/2026*
