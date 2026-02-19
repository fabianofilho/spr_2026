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

| Data       | Modelo + Classificador     | Score   | Status       | Notebook                                 |
|------------|----------------------------|---------|--------------|------------------------------------------|
| -          | W2V + LightGBM             | -       | ⏳ Pendente  | word2vec/submit_word2vec.ipynb           |
| -          | W2V + SVM (RBF)            | -       | ⏳ Pendente  | word2vec/submit_word2vec_svm.ipynb       |
| -          | W2V + XGBoost              | -       | ⏳ Pendente  | word2vec/submit_word2vec_xgboost.ipynb   |
| -          | FastText + LogReg          | -       | ⏳ Pendente  | word2vec/submit_fasttext.ipynb           |
| -          | W2V + Max/Mean Pool        | -       | ⏳ Pendente  | word2vec/submit_word2vec_maxpool.ipynb   |
| -          | W2V + TF-IDF Weighted      | -       | ⏳ Pendente  | word2vec/submit_word2vec_tfidf_weighted.ipynb |
| -          | W2V NILC (pre-trained)     | -       | ⏳ Pendente  | word2vec/submit_word2vec_nilc.ipynb      |

**Experimentos pendentes Word2Vec:**
- [ ] Testar diferentes `vector_size` (100, 200, 300)
- [ ] Testar diferentes `window` (3, 5, 7, 10)
- [x] Usar pesos TF-IDF na média dos embeddings
- [x] Testar FastText (subword)
- [ ] GloVe pré-treinado português

---

### 3. Transformers Fine-tuned

| Data       | Modelo                    | Score   | Status       | Notebook                                          |
|------------|---------------------------|---------|--------------|---------------------------------------------------|
| -          | BERTimbau-base            | -       | ⏳ Pendente  | transformers/submit_bertimbau.ipynb               |
| -          | BERTimbau-large + Focal   | -       | ⏳ Pendente  | transformers/submit_bertimbau_large_focal.ipynb   |
| -          | BERTimbau + LoRA          | -       | ⏳ Pendente  | transformers/submit_bertimbau_lora.ipynb          |
| -          | DistilBERT                | -       | ⏳ Pendente  | transformers/submit_distilbert.ipynb              |
| -          | mDeBERTa + ClassWeights   | -       | ⏳ Pendente  | transformers/submit_mdeberta_classweights.ipynb   |
| -          | DeBERTa-v3                | -       | ⏳ Pendente  | transformers/submit_deberta.ipynb                 |
| -          | XLM-RoBERTa + MeanPool    | -       | ⏳ Pendente  | transformers/submit_xlmroberta_meanpool.ipynb     |
| -          | BioBERTpt                 | -       | ⏳ Pendente  | transformers/submit_biobertpt.ipynb               |
| -          | Custom Transformer        | -       | ⏳ Pendente  | transformers/submit_custom_transformer.ipynb      |

**Experimentos pendentes Transformers:**
- [x] Focal Loss para desbalanceamento
- [x] Class Weights
- [x] Mean Pooling
- [x] LoRA (PEFT)
- [ ] Learning rate tuning (1e-5, 2e-5, 3e-5, 5e-5)
- [ ] Batch size (8, 16, 32)
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

| Data       | Método              | Score   | Status       | Notebook                              |
|------------|---------------------|---------|--------------|---------------------------------------|
| -          | Voting (Soft)       | -       | ⏳ Pendente  | ensemble/submit_ensemble.ipynb        |
| -          | Voting Classifier   | -       | ⏳ Pendente  | ensemble/submit_ensemble_voting.ipynb |
| -          | Stacking            | -       | ⏳ Pendente  | ensemble/submit_stacking.ipynb        |

**Experimentos pendentes Ensemble:**
- [x] Voting: TF-IDF + SBERT + BERT
- [x] Stacking com meta-learner (Logistic Regression)
- [ ] Weighted average baseado em OOF F1
- [ ] Blending

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
