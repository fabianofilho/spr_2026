# Pré-Treinamento com Datasets Externos

## Objetivo
Melhorar a performance dos modelos usando pré-treinamento em datasets médicos/radiológicos antes do fine-tuning no dataset da competição.

## Estrutura

```
pretrain/
├── datasets.md          # Lista de datasets disponíveis no Kaggle
├── README.md             # Este arquivo
└── submit_*.ipynb        # Notebooks de pré-treinamento (a criar)
```

## Workflow

1. **Selecionar modelo base** (ex: BERTimbau, mDeBERTa)
2. **Escolher dataset de pré-treino** (ver `datasets.md`)
3. **Criar notebook** de pré-treinamento
4. **Salvar pesos** como dataset privado no Kaggle
5. **Fine-tune** no dataset da competição usando os pesos pré-treinados

## Notebooks Planejados

| Notebook | Modelo | Dataset Pré-treino | Status |
|----------|--------|-------------------|--------|
| `submit_bertimbau_pretrain.ipynb` | BERTimbau | Medical Transcriptions | ⏳ |
| `submit_deberta_pretrain.ipynb` | mDeBERTa | PubMed 200k | ⏳ |
| `submit_sbert_pretrain.ipynb` | Sentence-BERT | Medical QA | ⏳ |

## Como Usar Pesos Pré-treinados no Kaggle

1. Treinar modelo e salvar em `/kaggle/working/model_weights/`
2. Fazer "Save & Run All"
3. Na aba Output, clicar "New Dataset"
4. No notebook de fine-tuning: Add Input → Seu Dataset

## Configuração no Notebook

```python
# Carregar pesos pré-treinados
PRETRAIN_PATH = '/kaggle/input/meu-modelo-pretrain/model_weights'
model = AutoModelForSequenceClassification.from_pretrained(PRETRAIN_PATH)
```
