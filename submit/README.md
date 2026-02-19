# SPR 2026 - Notebooks de Submissão (Offline)

Notebooks prontos para submissão no Kaggle **sem acesso à internet**.

## Notebooks Disponíveis

| Notebook | Estratégia | Requer Dataset Externo? |
|----------|------------|-------------------------|
| `submit_tfidf.ipynb` | TF-IDF + Logistic Regression | Não |
| `submit_tfidf_lgbm.ipynb` | TF-IDF + LightGBM | Não |
| `submit_word2vec.ipynb` | Word2Vec + LightGBM | Não |
| `submit_ensemble.ipynb` | Ensemble (TF-IDF + W2V) | Não |
| `submit_bertimbau.ipynb` | BERTimbau (fine-tuning) | Sim |
| `submit_distilbert.ipynb` | DistilBERT Multilingual | Sim |
| `submit_deberta.ipynb` | mDeBERTa V3 | Sim |
| `submit_sbert.ipynb` | Sentence Transformers | Sim |

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
