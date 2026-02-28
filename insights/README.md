# Insights - Análise Metodológica

> Consolidação de 40+ submissões (Fev/2026)

## Leaderboard Completo

### Top Performers (> 0.75)

| Rank | Modelo | Score | Categoria |
|------|--------|-------|-----------|
| 🏆 | BERTimbau + Focal Loss | **0.79696** | Transformers |
| 2 | Super Ensemble v1 | 0.78729 | Ensemble |
| 3 | Ensemble Soft Voting | 0.78049 | Ensemble |
| 4 | TF-IDF + LinearSVC | 0.77885 | TF-IDF |
| 5 | Custom Transformer | 0.77272 | Custom |
| 6 | LinearSVC v4 | 0.77244 | TF-IDF |
| 7 | SGDClassifier v3 🚀 | 0.77036 | TF-IDF |

### Mid Performers (0.65 - 0.75)

| Modelo | Score | Categoria |
|--------|-------|-----------|
| Ensemble v3 | 0.76567 | Ensemble |
| SGDClassifier v4 | 0.76503 | TF-IDF |
| LinearSVC v3 | 0.75966 | TF-IDF |
| TF-IDF + SGDClassifier | 0.75019 | TF-IDF |
| Ensemble TF-IDF + W2V | 0.74667 | Ensemble |
| Stacking Meta-Learner | 0.73852 | Ensemble |
| TF-IDF + LogReg | 0.72935 | TF-IDF |
| BioBERTpt | 0.72480 | Transformers |
| LogReg v3 | 0.71303 | TF-IDF |
| TF-IDF + LightGBM | 0.70273 | TF-IDF |
| TF-IDF + XGBoost | 0.69482 | TF-IDF |
| XLM-RoBERTa | 0.68767 | Transformers |
| ModernBERT | 0.68578 | Transformers |
| SVD + XGBoost | 0.66897 | TF-IDF |
| LightGBM v3 | 0.66454 | TF-IDF |
| Word2Vec + XGBoost | 0.66385 | Word2Vec |

### Low Performers (< 0.65)

| Modelo | Score | Categoria |
|--------|-------|-----------|
| BERTimbau base | 0.64319 | Transformers |
| Word2Vec + Max Pooling | 0.58009 | Word2Vec |
| Word2Vec + SVM | 0.57456 | Word2Vec |
| FastText + LogReg | 0.56783 | Word2Vec |
| Word2Vec NILC | 0.56727 | Word2Vec |
| Word2Vec + LightGBM | 0.56096 | Word2Vec |
| BERT Multilingual | 0.56095 | Transformers |
| DistilBERT | 0.55229 | Transformers |
| Word2Vec TF-IDF Weighted | 0.52215 | Word2Vec |
| SBERT + LightGBM | 0.48376 | SBERT |
| TF-IDF + CatBoost | 0.48202 | TF-IDF |
| TF-IDF + TabPFN | 0.39074 | TF-IDF |

### Falhas (< 0.30)

| Modelo | Score | Motivo |
|--------|-------|--------|
| BioBERTpt + Focal v2 | 0.26099 | Focal mal calibrada |
| Custom Transformer v2 | 0.41721 | Tokenizer quebrado |
| BERTimbau + LoRA | 0.13261 | Offline não funciona |
| Qwen3 Zero-Shot | 0.13261 | LLM não entende |
| Qwen3 One-Shot | 0.13261 | Mesmo problema |
| mDeBERTa | 0.01008 | Bug fp16 |

---

## Resubmissões (v2/v3/v4)

| Modelo | Original | Resubmit | Delta | Status |
|--------|----------|----------|-------|--------|
| Super Ensemble v1 | - | **0.78729** | - | 🔥 Novo 2º! |
| SGDClassifier v3 | 0.75019 | **0.77036** | **+2.7%** | 🚀 Única melhoria |
| BERTimbau + Focal v2 | 0.79696 | 0.79505 | -0.2% | ✅ OK |
| LinearSVC v4 | 0.77885 | 0.77244 | -0.8% | ⚠️ |
| Ensemble v3 | 0.78049 | 0.76567 | -1.9% | ⚠️ |
| SGDClassifier v4 | 0.77036 | 0.76503 | -0.7% | ⚠️ |
| LinearSVC v3 | 0.77885 | 0.75966 | -2.5% | ⚠️ |
| BERTimbau + Focal v3 | 0.79696 | 0.72625 | -8.9% | ⚠️ |
| LogReg v3 | 0.72935 | 0.71303 | -2.2% | ⚠️ |
| Custom Transformer v2 | 0.77272 | 0.41721 | -46% | ❌ |
| BioBERTpt + Focal v2 | 0.72480 | 0.26099 | -64% | ❌ |

**Taxa de sucesso:** 2/11 melhoraram (18%)

---

## Fatores de Sucesso

### 1. Tratamento de Classes Desbalanceadas

| Técnica | Impacto |
|---------|---------|
| Focal Loss (γ=2) | +15% em classes raras |
| class_weight='balanced' | Baseline sólido |
| Soft Voting | Suaviza erros |

### 2. Modelos em Português

| Modelo | Língua | Score |
|--------|--------|-------|
| BERTimbau | PT | **0.797** |
| BioBERTpt | PT | 0.725 |
| ModernBERT | EN | 0.686 |
| BERT Multilingual | Multi | 0.561 |

### 3. Preservação de Termos Técnicos

TF-IDF e BERTimbau preservam termos como "BIRADS", "calcificação", "nódulo espiculado".
Word2Vec (média) dilui essa informação.

---

## Arquivos de Insight por Categoria

| Arquivo | Melhor Score | Conclusão |
|---------|--------------|-----------|
| [transformers.md](transformers.md) | 0.79696 | BERTimbau + Focal Loss domina |
| [ensemble.md](ensemble.md) | 0.78049 | Soft Voting > Stacking |
| [tfidf.md](tfidf.md) | 0.77885 | Baseline difícil de bater |
| [sentence_transformers.md](sentence_transformers.md) | 0.77272 | Custom encoder surpreende |
| [word2vec.md](word2vec.md) | 0.66385 | Não recomendado |

---

*Atualizado: 28/02/2026*
