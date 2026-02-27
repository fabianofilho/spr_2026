# SPR 2026 - TODO

## Leaderboard (Public Score)

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 🏆 | **BERTimbau + Focal Loss** | **0.79696** | ✅ Submetido |
| 2 | **Ensemble Soft Voting** | **0.78049** | ✅ Submetido |
| 3 | TF-IDF + LinearSVC | 0.77885 | ✅ Submetido |
| 4 | **Custom Transformer Encoder** | **0.77272** | ✅ Submetido |
| 5 | **SGDClassifier v3** 🚀 | **0.77036** | ✅ Submetido (26/02) |
| 6 | LinearSVC v3 | 0.75966 | ✅ Submetido (26/02) |
| 7 | TF-IDF + SGDClassifier | 0.75019 | ✅ Submetido |
| 8 | **Ensemble TF-IDF + W2V** | **0.74667** | ✅ Submetido |
| 9 | **Stacking Meta-Learner** | **0.73852** | ✅ Submetido |
| 10 | TF-IDF + Logistic Regression | 0.72935 | ✅ Submetido |
| 11 | **BioBERTpt** | 0.72480 | ✅ Submetido |
| 12 | LogisticRegression v3 | 0.71303 | ✅ Submetido (26/02) |
| 13 | TF-IDF + LightGBM | 0.70273 | ✅ Submetido |
| 14 | TF-IDF + XGBoost | 0.69482 | ✅ Submetido |
| 15 | **XLM-RoBERTa + Mean Pooling** | 0.68767 | ✅ Submetido |
| 16 | **ModernBERT** | 0.68578 | ✅ Submetido |
| 17 | TF-IDF + SVD + XGBoost | 0.66897 | ✅ Submetido |
| 18 | Word2Vec + XGBoost | 0.66385 | ✅ Submetido |
| 19 | **BERTimbau base** | 0.64319 | ✅ Submetido |
| 20 | Word2Vec + Max Pooling | 0.58009 | ✅ Submetido |
| 21 | Word2Vec + SVM | 0.57456 | ✅ Submetido |
| 22 | FastText + LogReg | 0.56783 | ✅ Submetido |
| 23 | Word2Vec NILC | 0.56727 | ✅ Submetido |
| 24 | Word2Vec + LightGBM | 0.56096 | ✅ Submetido |
| 25 | BERT Multilingual | 0.56095 | ✅ Submetido |
| 26 | **DistilBERT Multilingual** | 0.55229 | ✅ Submetido |
| 27 | Word2Vec + TF-IDF Weighted | 0.52215 | ✅ Submetido |
| 28 | **SBERT + LightGBM** | 0.48376 | ✅ Submetido |
| 29 | TF-IDF + CatBoost | 0.48202 | ✅ Submetido |
| 30 | TF-IDF + TabPFN v0.1.9 | 0.39074 | ✅ Submetido |
| ❌ | BERTimbau + LoRA (Offline) | 0.13261 | ⚠️ Falhou |
| ❌ | Qwen3 1.7B One-Shot | 0.13261 | ⚠️ Falhou (26/02) |
| ❌ | mDeBERTa-v3 | 0.01008 | ⚠️ Bug fp16 |
| ❌ | mDeBERTa-v3 + Class Weights | 0.01008 | ⚠️ Bug fp16 |

### Resubmissões (v2/v3)

| Modelo | Score Original | Score Resubmit | Delta | Status |
|--------|----------------|----------------|-------|--------|
| BERTimbau + Focal Loss v2 | 0.79696 | **0.79505** | -0.2% | ✅ OK |
| **SGDClassifier v3** | 0.75019 | **0.77036** | **+2.7%** | 🚀 Melhoria |
| LinearSVC v3 | 0.77885 | 0.75966 | -2.5% | ⚠️ Regressão |
| BERTimbau + Focal Loss v3 | 0.79696 | 0.72625 | -8.9% | ⚠️ Regressão |
| LogisticRegression v3 | 0.72935 | 0.71303 | -2.2% | ⚠️ Regressão |
| Ensemble Soft Voting v2 | 0.78049 | 0.76387 | -2.1% | ⚠️ Regressão |
| Custom Transformer v2 | 0.77272 | 0.41721 | -46% | ❌ Falhou |
| BioBERTpt + Focal Loss v2 | 0.72480 | 0.26099 | -64% | ❌ Falhou |
| Qwen3 1.7B One-Shot | - | 0.13261 | - | ❌ Falhou |
| LightGBM v3 | 0.70273 | ⏳ Pendente | - | 📅 Reagendado 27/02 |

> **Lição:** SGDClassifier v3 foi a única melhoria! RandomizedSearch funcionou bem. Restantes v3 regredram.
> ⚠️ **Nota:** SGDClassifier v3 foi submetido 2x por engano (duplicado).

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

> **✅ TODOS OS TRANSFORMERS JÁ FORAM SUBMETIDOS** - Veja resultados no leaderboard acima

#### Notebooks de Download Disponíveis (`models/`)

**`models/bert/`**
| Notebook | Modelo HuggingFace | Tamanho | Status |
|----------|-------------------|---------|--------|
| `download_bertimbau.ipynb` | `neuralmind/bert-base-portuguese-cased` | ~440 MB | ✅ Usado |
| `download_bertimbau_large.ipynb` | `neuralmind/bert-large-portuguese-cased` | ~1.3 GB | ✅ Usado |
| `download_biobertpt.ipynb` | `pucpr/biobertpt-all` | ~440 MB | ✅ Usado |
| `download_distilbert.ipynb` | `distilbert-base-multilingual-cased` | ~540 MB | ✅ Usado |
| `download_modernbert.ipynb` | `answerdotai/ModernBERT-base` | ~580 MB | ✅ Usado |

**`models/deberta/`**
| Notebook | Modelo HuggingFace | Tamanho | Status |
|----------|-------------------|---------|--------|
| `download_mdeberta.ipynb` | `microsoft/mdeberta-v3-base` | ~560 MB | ✅ Usado |

**`models/roberta/`**
| Notebook | Modelo HuggingFace | Tamanho | Status |
|----------|-------------------|---------|--------|
| `download_xlmroberta.ipynb` | `xlm-roberta-base` | ~1.1 GB | ✅ Usado |

**`models/sbert/`**
| Notebook | Modelo HuggingFace | Tamanho | Status |
|----------|-------------------|---------|--------|
| `download_sbert.ipynb` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | ~470 MB | ⏳ Pendente |

#### Modelos Disponíveis no Kaggle (sem download)

| Modelo | Kaggle Path | Como adicionar |
|--------|-------------|----------------|
| BERT Multilingual | `bert-base-multilingual-cased` | Add Input → Models |
| ModernBERT | `modernbert` (answer-ai/base) | Add Input → Models |
| BERTimbau | `bertimbau-ptbr-complete` | Add Input → Models |
| mDeBERTa | `mdeberta_v3_base` (Jonathan Chan) | Add Input → Datasets |

#### dia 1
- [x] **BERT Multilingual** → 0.56095 `submit/transformers/submit_bert_multilingual.ipynb` ✅ Kaggle Models
- [x] **ModernBERT base** → 0.68578 `submit/transformers/submit_modernbert.ipynb` 🆕
- [x] **BERTimbau base** → 0.64319 `submit/transformers/submit_bertimbau.ipynb`
- [x] **DistilBERT Multilingual** → 0.55229 `submit/transformers/submit_distilbert.ipynb`
- [x] mDeBERTa + class weights → 0.01008 ⚠️ BUG `submit/transformers/submit_mdeberta_classweights.ipynb`

#### dia 2
- [x] **BERTimbau + Focal Loss** → **0.79696** 🏆 `submit/transformers/submit_bertimbau_large_focal.ipynb`
- [x] BERTimbau + LoRA (offline) → 0.13261 ❌ Falhou `submit/transformers/submit_bertimbau_lora.ipynb`
- [x] mDeBERTa-v3 → 0.01008 ⚠️ Bug fp16 `submit/transformers/submit_mdeberta.ipynb`
- [x] **XLM-RoBERTa + Mean Pooling** → 0.68767 `submit/transformers/submit_xlmroberta_meanpool.ipynb`
- [x] **BioBERTpt** → 0.72480 `submit/transformers/submit_biobertpt.ipynb`


### 4. Sentence Transformers (2 notebooks)

| Notebook | Modelo | Download | Status |
|----------|--------|----------|--------|
| `submit_sbert.ipynb` | `paraphrase-multilingual-MiniLM-L12-v2` | `models/sbert/download_sbert.ipynb` | ✅ Submetido |
| `submit_custom_transformer.ipynb` | Tokenizer from scratch | Não precisa | ✅ Submetido |

- [x] SBERT + LightGBM → 0.48376 `submit/sentence_transformers/submit_sbert.ipynb`
- [x] Custom Transformer Encoder → **0.77272** 🔥 `submit/transformers/submit_custom_transformer.ipynb`

### 5. Ensemble (3 notebooks)

> **✅ TODOS OS ENSEMBLES JÁ FORAM SUBMETIDOS** - Ensemble Soft Voting é o 2º melhor modelo!

- [x] VotingClassifier soft → **0.78049** 🥈 `submit/ensemble/submit_ensemble_voting.ipynb`
- [x] TF-IDF + W2V voting → 0.74667 `submit/ensemble/submit_ensemble.ipynb`
- [x] Stacking Meta-Learner → 0.73852 `submit/ensemble/submit_stacking.ipynb`
- [ ] **Super Ensemble v1** → `resubmit/2026-02-27/resubmit_super_ensemble_v1.ipynb` 📅 **Agendado 27/02**
  - BERTimbau + Focal Loss (0.45) + LinearSVC (0.25) + SGD v3 (0.20) + LogReg (0.10)
  - Weighted Soft Voting + Threshold Tuning

### 6. LLMs (Zero-Shot / One-Shot)

> **Modelos disponíveis no Kaggle Models** - Add Input → Models

| Notebook | Kaggle Models Path | Variação | VRAM | Download? |
|----------|--------------------|----------|------|----------|
| `submit_qwen3_1.7b.ipynb` | `QwenLM/Qwen3` | `1.7B` | ~4 GB | ✅ Kaggle Models |
| `submit_qwen3_4b.ipynb` | `QwenLM/Qwen3` | `4B` | ~8 GB | ✅ Kaggle Models |
| `submit_gemma3_4b.ipynb` | `google/gemma-3` | `4b` | ~8 GB | ✅ Kaggle Models |
| `submit_llama3_3b.ipynb` | `meta-llama/Llama-3.2` | `3B` | ~6 GB | ✅ Kaggle Models |
| `submit_mistral_7b.ipynb` | `MistralAI/Mistral-7B` | `7B-Instruct` | ~14 GB | ✅ Kaggle Models |
| `submit_phi3.5_mini.ipynb` | `Microsoft/Phi-3.5` | `mini-instruct` | ~8 GB | ✅ Kaggle Models |
| `submit_qwen2.5_7b.ipynb` | `QwenLM/Qwen2.5` | `7B-Instruct` | ~14 GB | ✅ Kaggle Models |

- [ ] Qwen3 1.7B → `submit/llm/submit_qwen3_1.7b.ipynb` ✅ **Pronto para rodar**
- [ ] Gemma 3 4B → `submit/llm/submit_gemma3_4b.ipynb` ✅ **Pronto para rodar**
- [ ] Qwen3 4B → `submit/llm/submit_qwen3_4b.ipynb` ✅ **Pronto para rodar**
- [ ] Llama 3.2 3B → `submit/llm/submit_llama3_3b.ipynb` ✅ **Pronto para rodar**
- [ ] Mistral 7B One-Shot → `resubmit/2026-02-27/resubmit_mistral_oneshot.ipynb` 📅 **Agendado 27/02**
- [ ] Phi-3.5 Mini One-Shot → `resubmit/2026-02-27/resubmit_phi35_oneshot.ipynb` 📅 **Agendado 27/02**
- [ ] Qwen 2.5 7B One-Shot → `resubmit/2026-02-27/resubmit_qwen25_oneshot.ipynb` 📅 **Agendado 27/02**

### 7. Pré-Treinamento (Datasets Externos)
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

### 10. NER (Named Entity Recognition) - NOVO
> Ver `tests/ner/README.md` para estratégia completa

**Motivação:** "ausência de microcalcificações" ≈ "microcalcificações pleomórficas" no espaço vetorial, mas são opostos para classificação BI-RADS.

**Pipeline:**
1. Extrair entidades (Nódulo, Microcalcificação, Arquitetura, Negação, BI-RADS)
2. Gerar embeddings **separados** por categoria
3. Concatenar + PCA (reduzir dimensionalidade)
4. XGBoost ordinal + regras determinísticas (BI-RADS 0 e 6)

**Experimentos:**
- [ ] NER + Regex + LinearSVC → `tests/ner/submit_ner_regex.ipynb`
- [ ] NER + BERTimbau embeddings → `tests/ner/submit_ner_bertimbau.ipynb`
- [ ] NER + XGBoost ordinal → `tests/ner/submit_ner_xgboost_ordinal.ipynb`
- [ ] NER + Regras determinísticas → `tests/ner/submit_ner_rules.ipynb`

## Workflows (Excalidraw) ✅
- [x] 1_tfidf_pipeline.excalidraw
- [x] 2_word2vec_pipeline.excalidraw
- [x] 3_transformers_pipeline.excalidraw
- [x] 4_sentence_transformers_pipeline.excalidraw
- [x] 5_ensemble_pipeline.excalidraw

## Depois de rodar
- Mapear um passo a passo para rodar os melhores modelos melhorados nos topicos dos tests/
    - Preprocess
    - Review
    - Treated
    - Augmented