# Insights - Análise Metodológica

> Consolidação de 55+ submissões (Fev-Mar/2026)

---

## 🎯 DICAS RÁPIDAS (Copie e Cole!)

### ✅ FAÇA ISSO

```python
# 1. USE MAX_LEN=512 (não 192!)
MAX_LEN = 512  # +2.4% vs MAX_LEN=192

# 2. USE 5-FOLD ENSEMBLE
N_FOLDS = 5
# Média das predições dos 5 folds

# 3. USE FOCAL LOSS
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# 4. USE THRESHOLD TUNING POR CLASSE
thresholds = [1.25, 0.40, 1.55, 0.90, 1.0, 1.0, 1.0]
adjusted = probs * thresholds
preds = adjusted.argmax(axis=1)

# 5. USE BERTimbau LARGE (não base!)
model = "neuralmind/bert-large-portuguese-cased"

# 6. USE RAW DATA (sem tratamento de texto)
# Não normalize, não remova stopwords!
```

### ❌ NÃO FAÇA ISSO

| Técnica | Por que falha | Score |
|---------|---------------|-------|
| LoRA offline | Kaggle não suporta | 0.13 |
| LLMs zero/one-shot | Não entendem BI-RADS | 0.13 |
| SMOTE | Overfitting | -5% |
| Tratar texto | Remove contexto útil | -2% |
| Seed Ensemble (>3) | Variância aumenta | -8% |
| BERTimbau base | Capacidade insuficiente | 0.64 |
| mDeBERTa | Bug fp16 | 0.01 |

### 🔥 PRÓXIMO EXPERIMENTO (Prioridade)

```python
# Combinar as 3 melhores técnicas:
MAX_LEN = 512              # ✅ Melhor score
N_FOLDS = 5                # ✅ Estabilidade
thresholds = otimizar()    # ✅ +3% potencial
label_smoothing = 0.1      # ✅ Regularização
```

**Meta:** Superar 0.85+

---

## Leaderboard Completo

### Top Performers (> 0.75)

| Rank | Modelo | Score | Categoria |
|------|--------|-------|-----------|
| 🏆 | **BERTimbau 5-Fold + MAX_LEN=512** | **0.84027** | Transformers |
| 🥈 | BERTimbau Threshold v3 | 0.81301 | Transformers |
| 3 | BERTimbau Raw Data v9 | 0.81213 | Transformers |
| 4 | BERTimbau v5 + Label Smoothing | 0.81100 | Transformers |
| 5 | BERTimbau 5-Fold v11 | 0.80950 | Transformers |
| 6 | BERTimbau MAX_LEN=512 v2 | 0.80509 | Transformers |
| 7 | BERTimbau v4 (Threshold Tuning) | 0.82073 | Transformers |
| 8 | BERTimbau + Focal Loss | 0.79696 | Transformers |
| 9 | Super Ensemble v1 | 0.78729 | Ensemble |
| 10 | Super Ensemble v3 + Threshold | 0.78660 | Ensemble |
| 11 | Ensemble Soft Voting | 0.78049 | Ensemble |
| 12 | TF-IDF + LinearSVC | 0.77885 | TF-IDF |
| 13 | BERTimbau v5 Ensemble Threshold | 0.77385 | Transformers |
| Ensemble TF-IDF + W2V | 0.74667 | Ensemble |
| Stacking Meta-Learner | 0.73852 | Ensemble |
| TF-IDF + LogReg | 0.72935 | TF-IDF |
| BioBERTpt | 0.72480 | Transformers |
| BERTimbau v5 Seed Ensemble | 0.72135 | Transformers |
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
| LinearSVC v5 | 0.77244 | 0.76298 | -1.2% | ⚠️ |
| LinearSVC v3 | 0.77885 | 0.75966 | -2.5% | ⚠️ |
| SGDClassifier v5 | 0.76503 | 0.74827 | -2.2% | ⚠️ |
| BERTimbau + Focal v3 | 0.79696 | 0.72625 | -8.9% | ⚠️ |
| LogReg v3 | 0.72935 | 0.71303 | -2.2% | ⚠️ |
| Custom Transformer v2 | 0.77272 | 0.41721 | -46% | ❌ |
| BioBERTpt + Focal v2 | 0.72480 | 0.26099 | -64% | ❌ |

**Taxa de sucesso:** 2/13 melhoraram (15%)

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
