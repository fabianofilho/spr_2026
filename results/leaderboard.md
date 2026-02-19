# Resultados - Leaderboard

**Melhor Score:** 0.72935 (TF-IDF + Logistic Regression)

---

## 📊 Resultados por Categoria

### 1. TF-IDF (Bag of Words)

| Data       | Classificador       | Score   | Status       | Notebook                         |
|------------|---------------------|---------|--------------|----------------------------------|
| 2025-02-19 | Logistic Regression | 0.72935 | ✅ Submetido | tfidf/submit_tfidf.ipynb         |
| 2025-02-19 | LightGBM            | 0.70273 | ✅ Submetido | tfidf/submit_tfidf_lgbm.ipynb    |
| -          | XGBoost             | -       | ⏳ Pendente  | tfidf/submit_tfidf_xgboost.ipynb |
| -          | CatBoost            | -       | ⏳ Pendente  | tfidf/submit_tfidf_catboost.ipynb|
| -          | TabPFN              | -       | ⏳ Pendente  | tfidf/submit_tfidf_tabpfn.ipynb  |
| -          | LinearSVC           | -       | ⏳ A criar   | -                                |
| -          | SGDClassifier       | -       | ⏳ A criar   | -                                |
| -          | NaiveBayes          | -       | ⏳ A criar   | -                                |
| -          | RandomForest        | -       | ⏳ A criar   | -                                |

**Experimentos pendentes TF-IDF:**
- [ ] Testar diferentes `max_features` (5k, 10k, 20k, 30k)
- [ ] Testar `ngram_range` (1,1), (1,2), (1,3), (2,3)
- [ ] Char n-grams vs Word n-grams
- [ ] TF-IDF + SVD (redução de dimensionalidade)
- [ ] Ensemble de classificadores lineares

---

### 2. Word2Vec / Embeddings Estáticos

| Data       | Classificador       | Score   | Status       | Notebook                        |
|------------|---------------------|---------|--------------|----------------------------------|
| -          | Logistic Regression | -       | ⏳ Pendente  | word2vec/submit_word2vec.ipynb  |
| -          | LightGBM            | -       | ⏳ A criar   | -                                |

**Experimentos pendentes Word2Vec:**
- [ ] Testar diferentes `vector_size` (100, 200, 300)
- [ ] Testar diferentes `window` (3, 5, 7, 10)
- [ ] Usar pesos TF-IDF na média dos embeddings
- [ ] Testar FastText (subword)
- [ ] GloVe pré-treinado português

---

### 3. Transformers Fine-tuned

| Data       | Modelo              | Score   | Status       | Notebook                                     |
|------------|---------------------|---------|--------------|----------------------------------------------|
| -          | BERTimbau-base      | -       | ⏳ Pendente  | transformers/submit_bertimbau.ipynb          |
| -          | BERTimbau-large     | -       | ⏳ A criar   | -                                            |
| -          | DistilBERT          | -       | ⏳ Pendente  | transformers/submit_distilbert.ipynb         |
| -          | mBERT               | -       | ⏳ A criar   | -                                            |
| -          | DeBERTa-v3          | -       | ⏳ Pendente  | transformers/submit_deberta.ipynb            |
| -          | XLM-RoBERTa         | -       | ⏳ A criar   | -                                            |
| -          | Custom Transformer  | -       | ⏳ Pendente  | transformers/submit_custom_transformer.ipynb |

**Experimentos pendentes Transformers:**
- [ ] Learning rate tuning (1e-5, 2e-5, 3e-5, 5e-5)
- [ ] Batch size (8, 16, 32)
- [ ] Epochs (3, 5, 8, 10)
- [ ] Max length (128, 256, 512)
- [ ] Pooling strategy (CLS, Mean, Max)
- [ ] Focal Loss para desbalanceamento
- [ ] Layer-wise Learning Rate Decay

---

### 4. Sentence Transformers (Embeddings → ML)

| Data       | Modelo + Classificador | Score   | Status       | Notebook                                   |
|------------|------------------------|---------|--------------|--------------------------------------------|
| -          | SBERT + Logistic       | -       | ⏳ Pendente  | sentence_transformers/submit_sbert.ipynb   |
| -          | SBERT + LightGBM       | -       | ⏳ A criar   | -                                          |
| -          | SBERT + XGBoost        | -       | ⏳ A criar   | -                                          |
| -          | SBERT + CatBoost       | -       | ⏳ A criar   | -                                          |
| -          | SBERT + TabPFN         | -       | ⏳ A criar   | -                                          |

**Modelos SBERT a testar:**
- [ ] `neuralmind/bert-base-portuguese-cased` (sentence embeddings)
- [ ] `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- [ ] `sentence-transformers/all-mpnet-base-v2`
- [ ] `sentence-transformers/LaBSE`

---

### 5. Ensemble

| Data       | Método              | Score   | Status       | Notebook                        |
|------------|---------------------|---------|--------------|----------------------------------|
| -          | Voting (Soft)       | -       | ⏳ Pendente  | ensemble/submit_ensemble.ipynb  |
| -          | Stacking            | -       | ⏳ A criar   | -                                |
| -          | Blending            | -       | ⏳ A criar   | -                                |

**Experimentos pendentes Ensemble:**
- [ ] Voting: TF-IDF + SBERT + BERT
- [ ] Stacking com meta-learner (Logistic Regression)
- [ ] Weighted average baseado em OOF F1

---

## 📈 Resumo de Performance

| Categoria              | Melhor Score | Modelo                |
|------------------------|--------------|------------------------|
| TF-IDF                 | 0.72935      | Logistic Regression    |
| Word2Vec               | -            | -                      |
| Transformers           | -            | -                      |
| Sentence Transformers  | -            | -                      |
| Ensemble               | -            | -                      |

---

## 🔍 Insights

1. **Logistic Regression > LightGBM** em TF-IDF sugere que modelos lineares funcionam bem para texto esparso
2. Priorizar: LinearSVC, SGDClassifier, NaiveBayes antes de tree-based
3. Transformers podem melhorar se dados forem suficientes para fine-tuning
