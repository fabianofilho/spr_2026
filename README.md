# SPR 2026 Mammography Report Classification

Repositório para o desafio [SPR 2026 Mammography Report Classification](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification).

## Sobre o Desafio

**Code Competition** da SPR para classificação de relatórios de mamografia em categorias BI-RADS (0-6).

- **Métrica**: F1-Score Macro
- **Formato**: O teste só existe no runtime de avaliação do Kaggle

## Leaderboard Atual (Top 5)

| Rank | Modelo | Score |
|------|--------|-------|
| 🥇 | TF-IDF + LinearSVC | **0.77885** |
| 🥈 | TF-IDF + SGDClassifier | 0.75019 |
| 🥉 | TF-IDF + Logistic Regression | 0.72935 |
| 4 | TF-IDF + LightGBM | 0.70273 |
| 5 | TF-IDF + XGBoost | 0.69482 |
| 6 | **ModernBERT** | 0.68578 |

> Ver [TODO.md](TODO.md) para lista completa de 18 submissões.

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
| TF-IDF | 12 | **0.77885** | ✅ Completo |
| Word2Vec | 7 | 0.66385 | ✅ Completo |
| Transformers | 11 | 0.68578 | 🔄 Em progresso |
| SBERT | 1 | - | ⏳ Pendente |
| Ensemble | 3 | - | ⏳ Pendente |
| LLMs | 6 | - | ⏳ Pendente |

## Insights

Análises metodológicas por categoria em `insights/`:
- [TF-IDF](insights/tfidf.md) - Benchmark: 0.77885
- [Word2Vec](insights/word2vec.md) - Diagnóstico de falhas
- [Transformers](insights/transformers.md) - Análise de resultados
- [Sentence Transformers](insights/sentence_transformers.md)
- [Ensemble](insights/ensemble.md)

## Dicas

1. **Melhor modelo até agora**: TF-IDF + LinearSVC (baseline forte!)
2. **Transformers**: ModernBERT (0.68578) superou BERTimbau (0.64319)
3. **Class weights**: Use para lidar com desbalanceamento
4. **Offline**: Sempre use `local_files_only=True` no Kaggle
