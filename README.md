# SPR 2026 Mammography Report Classification

Repositório para o desafio [SPR 2026 Mammography Report Classification](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification).

## Sobre o Desafio

**Code Competition** da SPR para classificação de relatórios de mamografia em categorias BI-RADS (0-6).

- **Métrica**: F1-Score Macro
- **Formato**: O teste só existe no runtime de avaliação do Kaggle

## Quick Start

### Opção 1: Google Colab

1. Configure os **Secrets** do Colab:
   - `KAGGLE_USERNAME`: seu username
   - `KAGGLE_KEY`: sua API key

2. Abra qualquer notebook e execute - os dados são baixados automaticamente

### Opção 2: Kaggle Notebooks

1. Faça upload do notebook desejado
2. Execute - o notebook detecta automaticamente o ambiente Kaggle

## Notebooks

| # | Notebook | Modelo | Complexidade | Tempo GPU |
|---|----------|--------|--------------|-----------|
| 01 | EDA | Análise Exploratória | - | 1 min |
| 02 | TF-IDF | TF-IDF + LightGBM/SVM | * | 5 min |
| 03 | Word2Vec | Word2Vec + Classifiers | * | 10 min |
| 04 | BERTimbau | BERT Português | *** | 45 min |
| 05 | FLAN-T5 | FLAN-T5 Small/Base | *** | 1h |
| 06 | DistilBERT | DistilBERT/mBERT/XLM-R | ** | 30 min |
| 07 | DeBERTa | mDeBERTa V3 | **** | 1.5h |
| 08 | SBERT | Sentence Transformers | ** | 15 min |
| 09 | Ensemble | Combinação | * | 1 min |
| 10 | Custom Transformer | Encoder + Extra Self-Attention | *** | 40 min |

## Estrutura

```
spr_2026/
├── notebooks/           # Notebooks Code Competition ready
├── src/
│   ├── configs/         # Configurações YAML
│   ├── models/          # Implementações
│   └── utils/           # Utilitários
├── data/                # Dados (git ignored)
└── submissions/         # Submissões geradas
```

## Code Competition

Os notebooks seguem o padrão da competição:

```python
# Detecção de ambiente
IS_KAGGLE = os.path.exists('/kaggle/input')
IS_COLAB = 'google.colab' in sys.modules

# Paths
if IS_KAGGLE:
    DATA_DIR = '/kaggle/input/spr-2026-mammography-report-classification'
elif IS_COLAB:
    # Download via Kaggle API
    DATA_DIR = 'data'

# Load test (disponível apenas em avaliação)
test_path = os.path.join(DATA_DIR, 'test.csv')
if os.path.exists(test_path):
    test = pd.read_csv(test_path)
else:
    test = None
```

## Dicas

1. **Modelos recomendados**: BERTimbau e DeBERTa para português
2. **Ensemble**: Combine modelos de diferentes famílias
3. **Class weights**: Use para lidar com desbalanceamento
4. **Early stopping**: Evite overfitting
