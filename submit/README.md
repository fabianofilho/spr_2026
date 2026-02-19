# SPR 2026 - Notebooks de Submissão (Offline)

Notebooks prontos para submissão no Kaggle **sem acesso à internet**.

## 📁 Estrutura por Categoria

```
submit/
├── tfidf/                    # TF-IDF + Classificadores ML
│   ├── submit_tfidf.ipynb           # Logistic Regression ✅ 0.72935
│   ├── submit_tfidf_lgbm.ipynb      # LightGBM ✅ 0.70273
│   ├── submit_tfidf_xgboost.ipynb   # XGBoost ⏳
│   ├── submit_tfidf_catboost.ipynb  # CatBoost ⏳
│   └── submit_tfidf_tabpfn.ipynb    # TabPFN ⏳
├── word2vec/                 # Word2Vec Embeddings
│   └── submit_word2vec.ipynb        # Word2Vec + ML ⏳
├── transformers/             # Fine-tuned Transformers
│   ├── submit_bertimbau.ipynb       # BERTimbau ⏳
│   ├── submit_distilbert.ipynb      # DistilBERT ⏳
│   └── submit_deberta.ipynb         # DeBERTa ⏳
├── sentence_transformers/    # SBERT Embeddings + ML
│   └── submit_sbert.ipynb           # SBERT + Logistic ⏳
└── ensemble/                 # Combinações
    └── submit_ensemble.ipynb        # Voting/Stacking ⏳
```

**Legenda:** ✅ Submetido | ⏳ Pendente

---

## Notebooks por Categoria

### 1. TF-IDF (`tfidf/`)

| Notebook | Classificador | Score | Requer Dataset Externo? |
|----------|---------------|-------|-------------------------|
| `submit_tfidf.ipynb` | Logistic Regression | 0.72935 | Não |
| `submit_tfidf_lgbm.ipynb` | LightGBM | 0.70273 | Não |
| `submit_tfidf_xgboost.ipynb` | XGBoost | - | Não |
| `submit_tfidf_catboost.ipynb` | CatBoost | - | Não |
| `submit_tfidf_tabpfn.ipynb` | TabPFN | - | Sim (TabPFN v2 Weights) |

### 2. Word2Vec (`word2vec/`)

| Notebook | Classificador | Score | Requer Dataset Externo? |
|----------|---------------|-------|-------------------------|
| `submit_word2vec.ipynb` | LightGBM | - | Não |

### 3. Transformers (`transformers/`)

| Notebook | Modelo | Score | Dataset Externo |
|----------|--------|-------|-----------------|
| `submit_bertimbau.ipynb` | BERTimbau | - | `neuralmind/bert-base-portuguese-cased` |
| `submit_distilbert.ipynb` | DistilBERT | - | `distilbert-base-multilingual-cased` |
| `submit_deberta.ipynb` | mDeBERTa | - | `microsoft/mdeberta-v3-base` |

### 4. Sentence Transformers (`sentence_transformers/`)

| Notebook | Modelo | Score | Dataset Externo |
|----------|--------|-------|-----------------|
| `submit_sbert.ipynb` | SBERT + Logistic | - | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

### 5. Ensemble (`ensemble/`)

| Notebook | Método | Score | Requer Dataset Externo? |
|----------|--------|-------|-------------------------|
| `submit_ensemble.ipynb` | Voting | - | Depende dos modelos |

---

## Como Submeter no Kaggle

### Passo 1: Criar Notebook no Kaggle
1. Vá em [kaggle.com/competitions/spr-2026-mammography-report-classification](https://kaggle.com/competitions/spr-2026-mammography-report-classification)
2. Clique em "Code" → "New Notebook"
3. Copie o conteúdo do notebook desejado

### Passo 2: Adicionar Dados
1. Clique em "Add Data"
2. Adicione o dataset da competição: `spr-2026-mammography-report-classification`

### Passo 3: Adicionar Modelos (se necessário)
Para notebooks que requerem modelos pré-treinados:

| Notebook | Modelo a adicionar |
|----------|-------------------|
| `submit_bertimbau.ipynb` | `neuralmind/bert-base-portuguese-cased` |
| `submit_distilbert.ipynb` | `distilbert-base-multilingual-cased` |
| `submit_deberta.ipynb` | `microsoft/mdeberta-v3-base` |
| `submit_sbert.ipynb` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

**Como adicionar:**
- Clique em "Add Data" → "Models" → Pesquise pelo nome do modelo

### Passo 4: Desativar Internet
1. Vá em "Settings" (ícone de engrenagem)
2. Em "Internet" selecione **OFF**
3. Salve as configurações

### Passo 5: Executar e Submeter
1. Execute todas as células ("Run All")
2. Verifique que `submission.csv` foi criado
3. Clique em "Submit to Competition"

## Notas

- **GPU recomendada** para notebooks com transformers (BERT, DeBERTa, etc.)
- Notebooks sem dependência de modelos externos são mais rápidos e confiáveis
- O ensemble combina TF-IDF e Word2Vec sem precisar de modelos externos

## Estrutura de Saída

Todos os notebooks geram um arquivo `submission.csv` com:
- `ID`: identificador do caso de teste
- `target`: classe BI-RADS predita (0-6)
