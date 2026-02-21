# Sentence Transformers - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| - | SBERT + LightGBM | - | ⏳ Pendente |

---

## Hipóteses

### 1. Vantagens sobre Word2Vec

- **Embeddings de sentença:** Não precisa fazer média de palavras
- **Treinamento contrastivo:** Otimizado para similaridade semântica
- **Modelos multilíngues:** E5, BGE, etc.

### 2. Vantagens sobre Fine-tuning

- **Mais rápido:** Só extrai embeddings e treina classificador
- **Menos overfitting:** Não ajusta pesos do transformer
- **Mais estável:** Menos hiperparâmetros

### 3. Riscos

- **Pode sofrer mesmo problema do Word2Vec:** Representação densa perde especificidade léxica
- **Modelos multilíngues:** Podem ser inferiores a modelos PT-BR específicos

---

## Configurações a Testar

```python
# Modelos candidatos
models = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
]

# Pipeline
1. Extrair embeddings (768D ou 1024D)
2. Redução opcional (PCA, UMAP)
3. Classificador (LightGBM, XGBoost, SVM)
```

---

## Análise (a ser preenchida após submissões)

*Aguardando resultados...*

---

*Atualizado em: 20/02/2026*
