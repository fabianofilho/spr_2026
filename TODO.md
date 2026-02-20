# SPR 2026 - TODO

## Leaderboard (Public Score)

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 1 | TF-IDF + LinearSVC | **0.77885** | ✅ Submetido |
| 2 | TF-IDF + SGDClassifier | 0.75019 | ✅ Submetido |
| 3 | TF-IDF + Logistic Regression | 0.72935 | ✅ Submetido |
| 4 | TF-IDF + LightGBM | 0.70273 | ✅ Submetido |
| 5 | TF-IDF + XGBoost | 0.69482 | ✅ Submetido |
| 6 | TF-IDF + SVD + XGBoost | 0.66897 | ✅ Submetido |
| 7 | TF-IDF + CatBoost | 0.48202 | ✅ Submetido |
| 8 | TF-IDF + TabPFN v0.1.9 | 0.39074 | ✅ Submetido |

---

## Estrutura de Modelos

### 1. TF-IDF (9 notebooks)
- [x] LinearSVC → **0.77885** 🏆 `submit/tfidf/submit_tfidf_linearsvc.ipynb`
- [x] SGDClassifier → 0.75019 `submit/tfidf/submit_tfidf_sgd.ipynb`
- [x] LogisticRegression baseline → 0.72935 `submit/tfidf/submit_tfidf.ipynb`
- [x] LightGBM + SVD → 0.70273 `submit/tfidf/submit_tfidf_lgbm.ipynb`
- [x] XGBoost → 0.69482 `submit/tfidf/submit_tfidf_xgboost.ipynb`
- [x] SVD + XGBoost → 0.66897 `submit/tfidf/submit_tfidf_svd_xgboost.ipynb`
- [x] CatBoost → 0.48202 `submit/tfidf/submit_tfidf_catboost.ipynb`
- [x] TabPFN v0.1.9 → 0.39074 `submit/tfidf/submit_tfidf_tabpfn_v1.ipynb`
- ~~TabPFN-2.5~~ ❌ Não funciona offline (requer pip install com internet)

### 2. Word2Vec (7 notebooks)
- [ ] Word2Vec + LightGBM → `submit/word2vec/submit_word2vec.ipynb`
- [ ] Word2Vec + SVM → `submit/word2vec/submit_word2vec_svm.ipynb`
- [ ] Word2Vec + XGBoost → `submit/word2vec/submit_word2vec_xgboost.ipynb`
- [ ] FastText + LogReg → `submit/word2vec/submit_fasttext.ipynb`
- [ ] MaxPool (Mean+Max 200D) → `submit/word2vec/submit_word2vec_maxpool.ipynb`
- [ ] TF-IDF Weighted → `submit/word2vec/submit_word2vec_tfidf_weighted.ipynb`
- [ ] NILC pretrained → `submit/word2vec/submit_word2vec_nilc.ipynb`

### 3. Transformers (9 notebooks)
- [ ] BERTimbau base → `submit/transformers/submit_bertimbau.ipynb`
- [ ] BERTimbau large + Focal Loss → `submit/transformers/submit_bertimbau_large_focal.ipynb`
- [ ] BERTimbau + LoRA → `submit/transformers/submit_bertimbau_lora.ipynb`
- [ ] BioBERTpt → `submit/transformers/submit_biobertpt.ipynb`
- [ ] mDeBERTa-v3 → `submit/transformers/submit_deberta.ipynb`
- [ ] mDeBERTa + class weights → `submit/transformers/submit_mdeberta_classweights.ipynb`
- [ ] DistilBERT → `submit/transformers/submit_distilbert.ipynb`
- [ ] XLM-RoBERTa + Mean Pool → `submit/transformers/submit_xlmroberta_meanpool.ipynb`
- [ ] Custom Transformer → `submit/transformers/submit_custom_transformer.ipynb`

### 4. Sentence Transformers (1 notebook)
- [ ] SBERT + LightGBM → `submit/sentence_transformers/submit_sbert.ipynb`

### 5. Ensemble (3 notebooks)
- [ ] TF-IDF + W2V voting → `submit/ensemble/submit_ensemble.ipynb`
- [ ] VotingClassifier soft → `submit/ensemble/submit_ensemble_voting.ipynb`
- [ ] Stacking OOF → `submit/ensemble/submit_stacking.ipynb`

### 6. LLMs (Zero-Shot) - NOVO
- [ ] Qwen3 1.7B → `submit/llm/submit_qwen3_1.7b.ipynb`
- [ ] Gemma 3 4B → `submit/llm/submit_gemma3_4b.ipynb`
- [ ] Qwen3 4B → `submit/llm/submit_qwen3_4b.ipynb`
- [ ] Llama 3.2 3B → `submit/llm/submit_llama3_3b.ipynb`

### 7. Pré-Treinamento (Datasets Externos) - NOVO
> Ver `submit/pretrain/datasets.md` para lista completa de datasets disponíveis

- [ ] BERTimbau + MLM médico → `submit/pretrain/submit_bertimbau_pretrain.ipynb`
- [ ] mDeBERTa + Domain Adaptation → `submit/pretrain/submit_deberta_pretrain.ipynb`
- [ ] Sentence-BERT + Contrastive → `submit/pretrain/submit_sbert_pretrain.ipynb`

**Datasets disponíveis:**
- Medical Transcriptions (`tboyle10/medicaltranscriptions`)
- PubMed 200k RCT (`owaiskhan9654/pubmed-200k-rct`)
- CBIS-DDSM BI-RADS (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`)
- Medical Specialty Classification (`chaitanyakck/medical-text`)

### 8. Dados Tratados (Pós-Processamento) - NOVO
> Ver `submit/treated/process.md` para pipeline completo

**Pipeline de processamento:**
- [ ] Análise exploratória (distribuição, padrões)
- [ ] Remoção de stop words (customizada para área médica)
- [ ] Lematização (spaCy pt_core_news_lg)
- [ ] Filtros e extração de termos BI-RADS
- [ ] Feature engineering

**Experimentos:**
- [ ] treated_v1 (stop words + lowercase) → `submit/treated/submit_treated_v1.ipynb`
- [ ] treated_v2 (stop words + lematização) → `submit/treated/submit_treated_v2.ipynb`
- [ ] treated_v3 (lematização + filtros BI-RADS) → `submit/treated/submit_treated_v3.ipynb`

## Workflows (Excalidraw) ✅
- [x] 1_tfidf_pipeline.excalidraw
- [x] 2_word2vec_pipeline.excalidraw
- [x] 3_transformers_pipeline.excalidraw
- [x] 4_sentence_transformers_pipeline.excalidraw
- [x] 5_ensemble_pipeline.excalidraw