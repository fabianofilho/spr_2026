# SPR 2026 - TODO

## Leaderboard (Public Score)

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 1 | TF-IDF + SGDClassifier | **0.75019** | ✅ Submetido |
| 2 | TF-IDF + Logistic Regression | 0.72935 | ✅ Submetido |
| 3 | TF-IDF + LightGBM | 0.70273 | ✅ Submetido |
| 4 | TF-IDF + SVD + XGBoost | 0.66897 | ✅ Submetido |
| 5 | TF-IDF + CatBoost | 0.48202 | ✅ Submetido |

---

## Estrutura de Modelos

### 1. TF-IDF (8 notebooks) ✅
- [x] LogisticRegression baseline → 0.72935
- [x] LightGBM + SVD → 0.70273
- [x] XGBoost
- [x] CatBoost → 0.48202
- [x] LinearSVC
- [x] SGDClassifier → **0.75019** 🏆
- [x] TabPFN ⚠️ (requer internet)
- [x] SVD + XGBoost → 0.66897

### 2. Word2Vec (7 notebooks) ✅
- [x] Word2Vec + LightGBM
- [x] Word2Vec + SVM
- [x] Word2Vec + XGBoost
- [x] FastText + LogReg
- [x] MaxPool (Mean+Max 200D)
- [x] TF-IDF Weighted
- [x] NILC pretrained

### 3. Transformers (9 notebooks) ✅
- [x] BERTimbau base
- [x] BERTimbau large + Focal Loss
- [x] BERTimbau + LoRA
- [x] BioBERTpt
- [x] mDeBERTa-v3
- [x] mDeBERTa + class weights
- [x] DistilBERT
- [x] XLM-RoBERTa + Mean Pool
- [x] Custom Transformer

### 4. Sentence Transformers (1 notebook) ✅
- [x] SBERT + LightGBM

### 5. Ensemble (3 notebooks) ✅
- [x] TF-IDF + W2V voting
- [x] VotingClassifier soft
- [x] Stacking OOF

## Workflows (Excalidraw) ✅
- [x] 1_tfidf_pipeline.excalidraw
- [x] 2_word2vec_pipeline.excalidraw
- [x] 3_transformers_pipeline.excalidraw
- [x] 4_sentence_transformers_pipeline.excalidraw
- [x] 5_ensemble_pipeline.excalidraw