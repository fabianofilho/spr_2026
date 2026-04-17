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

### 🔥 LICAO 2026-04-17: AWP + Optuna = 0.77969 (REGRESSAO -6pp)

**Resultado contraintuitivo.** A combinacao "de ponta" falhou feio. Scores abaixo:

| Tentativa | Score | Delta vs winner |
|-----------|-------|-----------------|
| Winner historico (BERTimbau 5-fold MAX_LEN=512) | **0.84027** | baseline |
| Baseline TF-IDF + LinearSVC | 0.77885 | -6.1 pp |
| **AWP + Optuna 300 trials (2026-04-17)** | **0.77969** | **-6.1 pp** |

**Causas identificadas (ordem de impacto):**

1. **Faltou `build_input_text` (maior culpado):** o notebook AWP usou texto raw (`train_df['report'].fillna('')`). O winner extrai secoes Indicacao/Achados/Comparativa. Diferenca estimada: **-3 a -5 pp**.
2. **Optuna overfitou OOF:** 8 dimensoes x 300 trials sobre ~3000 amostras. O ganho no OOF nao generalizou. **-1 a -2 pp**.
3. **Regularizacao tripla:** AWP + Label Smoothing + Focal Loss juntos suprimem o sinal em dataset pequeno. **-0.5 a -1 pp**.
4. **AWP_LR=1e-1 excessivo:** literatura usa 1e-4 a 1e-3. **-0.5 a -1 pp**.

### 🎯 NOVA ESTRATEGIA: Voltar ao basico

Parar de inventar combinacoes. **Primeiro reproduzir, depois inovar.**

#### Plano de 4 submissoes

| # | Notebook | Tecnica | Meta |
|---|----------|---------|------|
| 1 | `winner_reproduce` | Copia exata do winner (thresholds fixos, sem Optuna) | ~0.84 (controle) |
| 2 | `winner_pseudo_label` | Winner + pseudo-labeling (conf >0.95) | +0.5 a +1 pp |
| 3 | `winner_msdropout` | Winner + Multi-Sample Dropout (5x) | +0.2 a +0.5 pp |
| 4 | `winner_ensemble` | Media das 3 anteriores | +0.3 a +0.8 pp |

**Principios desta rodada:**

- ✅ USAR `build_input_text` sempre (extracao de secoes)
- ✅ Thresholds fixos do winner: `[0.95, 1.6, 1.35, 1.0, 0.4, 1.15, 0.1]`, T=0.958
- ✅ Focal Loss γ=2.0, α=0.25 (sem label smoothing)
- ❌ SEM AWP (nao ajudou aqui)
- ❌ SEM Optuna (overfitou)
- ❌ SEM label smoothing combinado com focal
- ❌ SEM texto raw

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
| [transformers.md](transformers.md) | 0.84027 | BERTimbau Large 5-fold MAX_LEN=512 domina |
| [ensemble.md](ensemble.md) | 0.78729 | Super Ensemble v1 melhor |
| [tfidf.md](tfidf.md) | 0.77885 | Baseline forte |
| [sentence_transformers.md](sentence_transformers.md) | 0.77272 | Custom encoder surpreende |
| [word2vec.md](word2vec.md) | 0.66385 | Nao recomendado |

### Rodada 2026-04-17

| Notebook | Tecnica Nova | Score | Observacao |
|----------|-------------|-------|------------|
| submit_bertimbau_large_awp_optuna | AWP + Optuna 300 trials | **0.77969** | ❌ -6pp vs winner. Texto raw + Optuna overfit |

---

*Atualizado: 17/04/2026*
