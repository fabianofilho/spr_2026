# TODO - SPR 2026

##  Problema Atual: Modelos de Boosting MUITO LENTOS

**Observações:**
- TF-IDF + CatBoost: **1h+** no Kaggle
- TF-IDF + XGBoost: **7min+** e ainda rodando
- TF-IDF + Logistic Regression: **< 1min** 

**Causa:** TF-IDF gera matriz esparsa com 15.000+ features. Tree-based models (XGBoost, CatBoost, LightGBM) não são otimizados para matrizes esparsas de alta dimensionalidade.

---

## Prioridade Alta: Otimização de Boosting com TF-IDF

### Estratégia 1: Redução de Dimensionalidade
- [ ] **TruncatedSVD/LSA**: Reduzir TF-IDF de 15k  500-1000 componentes
- [ ] **Ajustar TF-IDF**: Reduzir `max_features` para 3000-5000
- [ ] **Char n-grams apenas**: (3,5) char ngrams são mais compactos

```python
# Exemplo de otimização
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(texts)

# Reduzir dimensionalidade
svd = TruncatedSVD(n_components=500, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)  # Agora 500 features densas
```

### Estratégia 2: Hiperparâmetros Mais Leves
- [ ] **XGBoost/LightGBM**: `n_estimators=100`, `max_depth=6`, `subsample=0.8`
- [ ] **CatBoost**: `iterations=100`, `depth=6`, `early_stopping_rounds=10`
- [ ] **Early Stopping**: Usar validation set para parar cedo

```python
# XGBoost otimizado
model = xgb.XGBClassifier(
    n_estimators=100,       # Menos árvores
    max_depth=6,            # Profundidade menor
    subsample=0.8,          # Subsample para velocidade
    colsample_bytree=0.5,   # Usar 50% das features por árvore
    n_jobs=-1,
    eval_metric='mlogloss'
)
```

### Estratégia 3: GPU Acceleration
- [ ] **XGBoost GPU**: `tree_method='gpu_hist'` (requer GPU no Kaggle)
- [ ] **LightGBM GPU**: `device='gpu'`
- [ ] **CatBoost GPU**: `task_type='GPU'`

```python
# XGBoost com GPU
model = xgb.XGBClassifier(
    tree_method='gpu_hist',  # Usar GPU
    gpu_id=0,
    n_estimators=200,
    max_depth=8
)
```

### Estratégia 4: Usar Modelos Lineares (RECOMENDADO)
- [x] **Logistic Regression**: Já funciona bem (0.72935), < 1min 
- [ ] **LinearSVC**: Rápido e bom para texto esparso
- [ ] **SGDClassifier**: Muito rápido, escala bem
- [ ] **RidgeClassifier**: Simples e eficaz

**Conclusão:** Para TF-IDF esparso, modelos lineares são mais adequados que boosting.

---

## Prioridade Média: Experimentos Word2Vec/FastText

### Embeddings + Boosting (MAIS RÁPIDO)
- [ ] **FastText + XGBoost**: Embedding denso (100-300 dims)  XGBoost é rápido
- [ ] **Word2Vec + TF-IDF Weighted + LightGBM**: Features densas  boosting viável
- [ ] **Mean+Max Pooling**: 200 features densas  qualquer classificador

**Por que funciona:** Embeddings geram vetores densos de baixa dimensão (~100-300), tree-based models funcionam bem com features densas.

```python
# Word2Vec (100 dims) + XGBoost = RÁPIDO
X_w2v = np.array([get_embedding(t) for t in texts])  # Shape: (n, 100)
model = xgb.XGBClassifier(n_estimators=200, max_depth=8)
model.fit(X_w2v, y)  # Treina em segundos!
```

---

## Prioridade Média: Transformers

- [ ] **BERTimbau-Large + Focal Loss**: GPU T4 x2, ~30-45min
- [ ] **mDeBERTa + Class Weights**: Alternativa mais leve
- [ ] **BioBERTpt**: Pré-treinado em textos médicos PT-BR
- [ ] **LoRA**: Fine-tuning eficiente, menor tempo/memória

---

## Prioridade Baixa: Ensembles

- [ ] **Voting**: Combinar TF-IDF + Word2Vec + Transformer
- [ ] **Stacking**: OOF predictions  Meta-learner
- [ ] **Blending**: Média ponderada por F1

---

## Resumo de Ações Imediatas

1.  **PAUSAR** testes de boosting com TF-IDF (muito lento)
2.  **MANTER** Logistic Regression para TF-IDF (rápido + bom score)
3.  **TESTAR** FastText/Word2Vec com boosting (features densas = rápido)
4.  **TESTAR** TF-IDF + SVD (500 dims) + XGBoost (se quiser boosting)
5.  **TESTAR** Transformers em GPU (paralelo aos outros testes)

---

## Notebooks a Criar/Atualizar

| Notebook | Status | Prioridade |
|----------|--------|------------|
| `submit_tfidf_svd_xgboost.ipynb` | A criar | Alta |
| `submit_tfidf_sgd.ipynb` | A criar | Alta |
| `submit_tfidf_linearsvc.ipynb` | A criar | Média |
| `submit_word2vec_xgboost.ipynb` |  Criado | Alta |
| `submit_fasttext.ipynb` |  Criado | Alta |
