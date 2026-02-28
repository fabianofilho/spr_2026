# SPR 2026 Mammography Report Classification

Repositório para o desafio [SPR 2026 Mammography Report Classification](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification).

## Sobre o Desafio

**Code Competition** da SPR para classificação de relatórios de mamografia em categorias BI-RADS (0-6).

- **Métrica**: F1-Score Macro
- **Formato**: O teste só existe no runtime de avaliação do Kaggle

## Leaderboard Atual (Top 15)

| Rank | Modelo | Score |
|------|--------|-------|
| 🥇 | **BERTimbau + Focal Loss** | **0.79696** |
| 🥈 | BERTimbau + Focal Loss v2 | 0.79505 |
| 🥉 | **Super Ensemble v1** | **0.78729** |
| 4 | Ensemble Soft Voting | 0.78049 |
| 5 | TF-IDF + LinearSVC | 0.77885 |
| 6 | Custom Transformer Encoder | 0.77272 |
| 7 | LinearSVC v4 (Calibration+Platt) | 0.77244 |
| 8 | **SGDClassifier v3** 🚀 | **0.77036** |
| 9 | Ensemble v3 | 0.76567 |
| 10 | SGDClassifier v4 (RandomSearch) | 0.76503 |
| 11 | Ensemble Soft Voting v2 | 0.76387 |
| 12 | LinearSVC v3 | 0.75966 |
| 13 | TF-IDF + SGDClassifier | 0.75019 |
| 14 | Ensemble TF-IDF + W2V | 0.74667 |
| 15 | Stacking Meta-Learner | 0.73852 |

> Ver [TODO.md](TODO.md) para lista completa de 35+ submissões (incluindo resubmissões v2/v3).

## Estrutura do Repositório

```
spr_2026/
├── submit/                    # 📤 Notebooks prontos para submeter no Kaggle
│   ├── tfidf/                 # TF-IDF + modelos clássicos (12 notebooks)
│   ├── word2vec/              # Word2Vec/FastText embeddings
│   ├── transformers/          # BERTimbau, ModernBERT, mDeBERTa, etc.
│   ├── sentence_transformers/ # SBERT
│   ├── ensemble/              # Combinações de modelos
│   ├── llm/                   # LLMs zero-shot (Qwen, Gemma, Llama)
│   └── resubmit/              # Re-submissões com ajustes
├── submissions/               # 📁 Cópia dos notebooks já submetidos (local)
├── notebooks/                 # 📓 Notebooks de exploração e desenvolvimento
├── insights/                  # 📊 Análises de resultados por categoria
├── models/                    # 🤖 Scripts para download de modelos
├── tests/                     # 🧪 Experimentos
│   ├── augmented/             # Data augmentation
│   ├── pretrain/              # Pré-treinamento com dados externos
│   ├── treated/               # Dados pré-processados
│   └── review/                # Revisão de resultados
├── workflows/                 # 📐 Diagramas Excalidraw dos pipelines
├── TODO.md                    # ✅ Tracking de progresso e leaderboard
├── CLAUDE.md                  # 🤖 Instruções para o Copilot
└── requirements.txt           # 📦 Dependências
```

## Quick Start

### Kaggle Notebooks (Recomendado)

1. Faça upload do notebook da pasta `submit/`
2. Adicione os inputs necessários (modelos, datasets)
3. Execute - o notebook detecta automaticamente o ambiente Kaggle

### Google Colab

1. Configure os **Secrets** do Colab:
   - `KAGGLE_USERNAME`: seu username
   - `KAGGLE_KEY`: sua API key

2. Abra qualquer notebook da pasta `notebooks/` e execute

## Categorias de Modelos

| Categoria | Notebooks | Melhor Score | Status |
|-----------|-----------|--------------|--------|
| Transformers | 11 | **0.79696** | ✅ Completo |
| Ensemble | 3 | 0.78049 | ✅ Completo |
| TF-IDF | 12 | 0.77885 | ✅ Completo |
| Custom Transformer | 1 | 0.77272 | ✅ Completo |
| Word2Vec | 7 | 0.66385 | ✅ Completo |
| SBERT | 1 | 0.48376 | ✅ Completo |
| LLMs | 6 | - | ⏳ Pendente |

## Insights

Análises metodológicas por categoria em `insights/`:
- [TF-IDF](insights/tfidf.md) - Benchmark: 0.77885
- [Word2Vec](insights/word2vec.md) - Diagnóstico de falhas
- [Transformers](insights/transformers.md) - Análise de resultados
- [Sentence Transformers](insights/sentence_transformers.md)
- [Ensemble](insights/ensemble.md)

### Lições das Resubmissões (v2/v3/v4)

| Modelo | Original | Resubmit | Status |
|--------|----------|----------|--------|
| **Super Ensemble v1** | - | **0.78729** | 🔥 2º melhor! |
| BERTimbau + Focal v2 | 0.79696 | 0.79505 | ✅ OK |
| **SGDClassifier v3** | 0.75019 | **0.77036** | 🚀 +2.7% |
| LinearSVC v4 | 0.77885 | 0.77244 | ⚠️ -0.8% |
| Ensemble v3 | 0.78049 | 0.76567 | ⚠️ -1.9% |
| SGDClassifier v4 | 0.77036 | 0.76503 | ⚠️ -0.7% |
| LinearSVC v3 | 0.77885 | 0.75966 | ⚠️ -2.5% |
| LogisticRegression v3 | 0.72935 | 0.71303 | ⚠️ -2.2% |
| BERTimbau + Focal v3 | 0.79696 | 0.72625 | ⚠️ -8.9% |
| Ensemble Voting v2 | 0.78049 | 0.76387 | ⚠️ -2.1% |
| Custom Transformer v2 | 0.77272 | 0.41721 | ❌ -46% |
| BioBERTpt + Focal v2 | 0.72480 | 0.26099 | ❌ -64% |
| Qwen3 1.7B One-Shot | - | 0.13261 | ❌ Falhou |

> **Insight:** SGDClassifier v3 é a única melhoria! v4 com SMOTE regrediu. Super Ensemble v1 é 2º melhor!

## Dicas

1. **Melhor modelo até agora**: BERTimbau + Focal Loss (0.79696)
2. **Ensemble**: Soft Voting (0.78049) é o 2º melhor modelo!
3. **Custom Transformer**: Tokenizer from scratch (0.77272) supera maioria dos transformers
4. **Class weights**: Use para lidar com desbalanceamento
5. **Offline**: Sempre use `local_files_only=True` no Kaggle
