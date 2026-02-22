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
| 7 | Word2Vec + XGBoost | 0.66385 | ✅ Submetido |
| 8 | Word2Vec + Max Pooling | 0.58009 | ✅ Submetido |
| 9 | Word2Vec + SVM | 0.57456 | ✅ Submetido |
| 10 | FastText + LogReg | 0.56783 | ✅ Submetido |
| 11 | Word2Vec NILC | 0.56727 | ✅ Submetido |
| 12 | Word2Vec + LightGBM | 0.56096 | ✅ Submetido |
| 13 | **BERT Multilingual** | 0.56095 | ✅ Submetido |
| 14 | Word2Vec + TF-IDF Weighted | 0.52215 | ✅ Submetido |
| 15 | TF-IDF + CatBoost | 0.48202 | ✅ Submetido |
| 16 | TF-IDF + TabPFN v0.1.9 | 0.39074 | ✅ Submetido |

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
- [x] Word2Vec + XGBoost → 0.66385 `submit/word2vec/submit_word2vec_xgboost.ipynb`
- [x] Word2Vec + Max Pooling → 0.58009 `submit/word2vec/submit_word2vec_maxpool.ipynb`
- [x] Word2Vec + SVM → 0.57456 `submit/word2vec/submit_word2vec_svm.ipynb`
- [x] FastText + LogReg → 0.56783 `submit/word2vec/submit_fasttext.ipynb`
- [x] Word2Vec NILC → 0.56727 `submit/word2vec/submit_word2vec_nilc.ipynb`
- [x] Word2Vec + LightGBM → 0.56096 `submit/word2vec/submit_word2vec.ipynb`
- [x] Word2Vec + TF-IDF Weighted → 0.52215 `submit/word2vec/submit_word2vec_tfidf_weighted.ipynb`

### 3. Transformers (10 notebooks)

> **⚠️ IMPORTANTE:** A maioria dos modelos **NÃO está** no Kaggle Models. Requer upload manual como Dataset.

#### Como fazer upload de modelos HuggingFace:
```bash
# 1. Clone o modelo localmente
git clone https://huggingface.co/neuralmind/bert-base-portuguese-cased

# 2. Kaggle → Datasets → New Dataset → Upload a pasta
# 3. Add Data → Your Work → selecionar o dataset
```

| Notebook | Modelo HuggingFace | Kaggle Dataset |
|----------|-------------------|----------------|
| submit_bertimbau.ipynb | `neuralmind/bert-base-portuguese-cased` | ✅ `fernandosr85/bertimbau-portuguese` |
| submit_bertimbau_large_focal.ipynb | `neuralmind/bert-large-portuguese-cased` | ❌ Upload manual |
| submit_bertimbau_lora_offline.ipynb | `neuralmind/bert-large-portuguese-cased` | ❌ Upload manual |
| submit_biobertpt.ipynb | `pucpr/biobertpt-all` | ❌ Upload (ou usar `Bio_ClinicalBERT`) |

| submit_deberta.ipynb | `microsoft/mdeberta-v3-base` | ✅ `jonathanchan/mdeberta_v3_base` |
| submit_mdeberta_classweights.ipynb | `microsoft/mdeberta-v3-base` | ✅ `jonathanchan/mdeberta_v3_base` |
| submit_distilbert.ipynb | `distilbert-base-multilingual-cased` | ✅ Verificar no Kaggle |
| submit_xlmroberta_meanpool.ipynb | `xlm-roberta-large` | ✅ Verificar no Kaggle |
| submit_modernbert.ipynb | `answerdotai/ModernBERT-base` | ✅ `chesteryuan/modernbert-base` |
| submit_bert_multilingual.ipynb | `google-bert/bert-base-multilingual-cased` | ✅ Kaggle Models |
| submit_custom_transformer.ipynb | Tokenizer only | ❌ Upload manual |

#### dia 1
- [x] **BERT Multilingual** → 0.56095 `submit/transformers/submit_bert_multilingual.ipynb` ✅ Kaggle Models
- [ ] BERTimbau base → `submit/transformers/submit_bertimbau.ipynb`
- [ ] BERTimbau large + Focal Loss → `submit/transformers/submit_bertimbau_large_focal.ipynb`
- [ ] BERTimbau + LoRA (offline) → `submit/transformers/submit_bertimbau_lora_offline.ipynb` 🆕 ✅ 100% offline
- [ ] BioBERTpt → `submit/transformers/submit_biobertpt.ipynb`
- [ ] mDeBERTa-v3 → `submit/transformers/submit_deberta.ipynb`

#### dia 2
- [ ] mDeBERTa + class weights → `submit/transformers/submit_mdeberta_classweights.ipynb`
- [ ] DistilBERT → `submit/transformers/submit_distilbert.ipynb`
- [ ] XLM-RoBERTa + Mean Pool → `submit/transformers/submit_xlmroberta_meanpool.ipynb`
- [ ] **ModernBERT base** → `submit/transformers/submit_modernbert.ipynb` 🆕
- [ ] Custom Transformer → `submit/transformers/submit_custom_transformer.ipynb`

### 4. Sentence Transformers (1 notebook)

> **Kaggle Input:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

- [ ] SBERT + LightGBM → `submit/sentence_transformers/submit_sbert.ipynb`

### 5. Ensemble (3 notebooks)
- [ ] TF-IDF + W2V voting → `submit/ensemble/submit_ensemble.ipynb`
- [ ] VotingClassifier soft → `submit/ensemble/submit_ensemble_voting.ipynb`
- [ ] Stacking OOF → `submit/ensemble/submit_stacking.ipynb`

### 6. LLMs (Zero-Shot) - NOVO

> **⚠️ IMPORTANTE:** Modelos devem ser adicionados como **Input** no Kaggle

| Notebook | Kaggle Input |
|----------|--------------|
| submit_qwen3_1.7b.ipynb | `QwenLM/Qwen3` → Variação `1.7B` |
| submit_gemma3_4b.ipynb | `google/gemma-3` → Variação `4b` |
| submit_qwen3_4b.ipynb | `QwenLM/Qwen3` → Variação `4B` |
| submit_llama3_3b.ipynb | `meta-llama/Llama-3.2` → Variação `3B` |

- [ ] Qwen3 1.7B → `submit/llm/submit_qwen3_1.7b.ipynb`
- [ ] Gemma 3 4B → `submit/llm/submit_gemma3_4b.ipynb`
- [ ] Qwen3 4B → `submit/llm/submit_qwen3_4b.ipynb`
- [ ] Llama 3.2 3B → `submit/llm/submit_llama3_3b.ipynb`

### 7. Pré-Treinamento (Datasets Externos) - NOVO
> Ver `tests/pretrain/datasets.md` para lista completa de datasets disponíveis

- [ ] BERTimbau + MLM médico → `tests/pretrain/submit_bertimbau_pretrain.ipynb`
- [ ] mDeBERTa + Domain Adaptation → `tests/pretrain/submit_deberta_pretrain.ipynb`
- [ ] Sentence-BERT + Contrastive → `tests/pretrain/submit_sbert_pretrain.ipynb`

**Datasets disponíveis:**
- Medical Transcriptions (`tboyle10/medicaltranscriptions`)
- PubMed 200k RCT (`owaiskhan9654/pubmed-200k-rct`)
- CBIS-DDSM BI-RADS (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`)
- Medical Specialty Classification (`chaitanyakck/medical-text`)

### 8. Dados Tratados (Pós-Processamento) - NOVO
> Ver `tests/treated/process.md` para pipeline completo

**Pipeline de processamento:**
- [ ] Análise exploratória (distribuição, padrões)
- [ ] Remoção de stop words (customizada para área médica)
- [ ] Lematização (spaCy pt_core_news_lg)
- [ ] Filtros e extração de termos BI-RADS
- [ ] Feature engineering

**Experimentos:**
- [ ] treated_v1 (stop words + lowercase) → `tests/treated/submit_treated_v1.ipynb`
- [ ] treated_v2 (stop words + lematização) → `tests/treated/submit_treated_v2.ipynb`
- [ ] treated_v3 (lematização + filtros BI-RADS) → `tests/treated/submit_treated_v3.ipynb`

### 9. Data Augmentation - NOVO
> Ver `tests/augmented/strategies.md` para técnicas disponíveis

**Status:** ⏳ Aguardando resultados dos modelos base

**Técnicas planejadas:**
- [ ] EDA (Easy Data Augmentation): synonym/swap/delete
- [ ] SMOTE oversampling para classes 5 e 6
- [ ] MLM augmentation com BERTimbau
- [ ] Back-translation PT→EN→PT
- [ ] Pseudo-labeling com datasets externos

**Experimentos (após identificar melhores modelos):**
- [ ] augmented_linearsvc → `tests/augmented/submit_augmented_linearsvc.ipynb`
- [ ] augmented_bertimbau → `tests/augmented/submit_augmented_bertimbau.ipynb`
- [ ] augmented_ensemble → `tests/augmented/submit_augmented_ensemble.ipynb`

## Workflows (Excalidraw) ✅
- [x] 1_tfidf_pipeline.excalidraw
- [x] 2_word2vec_pipeline.excalidraw
- [x] 3_transformers_pipeline.excalidraw
- [x] 4_sentence_transformers_pipeline.excalidraw
- [x] 5_ensemble_pipeline.excalidraw