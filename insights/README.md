# Insights - Análise Metodológica

Esta pasta contém análises metodológicas dos resultados de cada categoria de modelos.

## Objetivo

Identificar **por que** certos modelos performam melhor que outros, analisando:
- Características do dataset
- Adequação da representação textual
- Hiperparâmetros e configurações
- Trade-offs entre abordagens

## Arquivos

| Arquivo | Categoria | Melhor Score |
|---------|-----------|--------------|
| [tfidf.md](tfidf.md) | TF-IDF | 0.77885 |
| [word2vec.md](word2vec.md) | Word2Vec | 0.66385 |
| [transformers.md](transformers.md) | Transformers | - |
| [sentence_transformers.md](sentence_transformers.md) | SBERT | - |
| [ensemble.md](ensemble.md) | Ensemble | - |

## Resumo Executivo

### Por que TF-IDF domina?

TF-IDF captura **termos específicos** do domínio médico (BI-RADS, calcificações, nódulos) que são altamente discriminativos. Em textos curtos e técnicos, a presença/ausência de palavras-chave é mais importante que semântica.

### Por que Word2Vec decepciona?

Word2Vec aprende representações **semânticas gerais**, mas:
1. Perde a especificidade léxica (TF-IDF preserva)
2. Média de embeddings dilui informação discriminativa
3. Vocabulário médico pode estar sub-representado em modelos pré-treinados

---

*Atualizado em: 20/02/2026*
