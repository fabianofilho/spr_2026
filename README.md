# SPR 2026 Mammography Report Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.35+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Repositório para o desafio [SPR 2026 Mammography Report Classification](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification) do Kaggle.

## 📋 Sobre o Desafio

A Sociedade Paulista de Radiologia (SPR) está organizando um Desafio de Inteligência Artificial para Câncer de Mama. O objetivo é criar uma solução que prediz a **categoria BI-RADS** a partir da seção de achados de relatórios de mamografia.

### O que é BI-RADS?

O sistema BI-RADS (Breast Imaging Reporting and Data System) foi desenvolvido pelo American College of Radiology para padronizar relatórios de mamografia:

| Categoria | Descrição | Ação Recomendada |
|-----------|-----------|------------------|
| 0 | Incompleto - necessita avaliação adicional | Recall para exames adicionais |
| 1 | Negativo | Rastreamento de rotina |
| 2 | Achado benigno | Rastreamento de rotina |
| 3 | Provavelmente benigno | Seguimento em curto intervalo |
| 4 | Anormalidade suspeita | Considerar biópsia |
| 5 | Altamente sugestivo de malignidade | Ação apropriada |
| 6 | Malignidade comprovada por biópsia | Tratamento cirúrgico/oncológico |

### Métricas

- **Métrica de avaliação**: F1-Score Macro
- **Formato de submissão**: `ID,target`

## 🏗️ Estrutura do Projeto

```
spr_2026/
├── data/                       # Dados do desafio
│   ├── train.csv              # Dados de treino
│   └── test.csv               # Dados de teste (apenas no Kaggle)
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_eda.ipynb           # Análise Exploratória
│   └── 02_bert_baseline.ipynb # Baseline com BERT
│
├── src/                        # Código fonte
│   ├── configs/               # Arquivos de configuração
│   │   ├── base_config.yaml   # Configurações base
│   │   ├── bert_config.yaml   # Config para BERT
│   │   ├── t5_config.yaml     # Config para T5
│   │   ├── gemma_config.yaml  # Config para GEMMA
│   │   └── deberta_config.yaml# Config para DeBERTa
│   │
│   ├── models/                # Implementações dos modelos
│   │   ├── bert/              # BERT, BERTimbau, XLM-RoBERTa
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   └── inference.py
│   │   ├── t5/                # T5, mT5, Flan-T5
│   │   │   └── train.py
│   │   ├── gemma/             # GEMMA 3 com LoRA/QLoRA
│   │   │   └── train.py
│   │   └── deberta/           # DeBERTa v3
│   │       └── train.py
│   │
│   └── utils/                 # Utilitários
│       ├── common.py          # Funções gerais
│       ├── data_utils.py      # Dataset e data loading
│       └── metrics.py         # Métricas de avaliação
│
├── experiments/               # Outputs dos experimentos
├── submissions/               # Arquivos de submissão
├── requirements.txt           # Dependências
└── README.md
```

## 🚀 Instalação

```bash
# Clonar repositório
git clone https://github.com/fabianofilho/spr_2026.git
cd spr_2026

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## 📦 Download dos Dados

```bash
# Fazer login no Kaggle (precisa ter ~/.kaggle/kaggle.json configurado)
kaggle competitions download -c spr-2026-mammography-report-classification

# Extrair dados
unzip spr-2026-mammography-report-classification.zip -d data/
```

## 🎯 Modelos Disponíveis

### 1. BERT / BERTimbau

Modelo encoder baseado em BERT, ideal para classificação de texto em português.

```bash
# Treinar BERT
python src/models/bert/train.py --config src/configs/bert_config.yaml

# Treinar apenas fold específico
python src/models/bert/train.py --fold 0

# Inferência (ensemble)
python src/models/bert/inference.py --experiment-dir experiments/bert_xxx --test-path data/test.csv
```

**Modelos recomendados:**
- `neuralmind/bert-base-portuguese-cased` (BERTimbau)
- `neuralmind/bert-large-portuguese-cased`
- `xlm-roberta-base`

### 2. T5 / mT5

Modelo sequence-to-sequence que trata classificação como geração de texto.

```bash
python src/models/t5/train.py --config src/configs/t5_config.yaml
```

**Modelos recomendados:**
- `unicamp-dl/ptt5-base-portuguese-vocab` (PTT5)
- `google/flan-t5-base`
- `google/mt5-base`

### 3. GEMMA 3 (com LoRA/QLoRA)

LLM recente do Google com fine-tuning eficiente via LoRA.

```bash
python src/models/gemma/train.py --config src/configs/gemma_config.yaml
```

**Modelos recomendados:**
- `google/gemma-3-4b-it`
- `google/gemma-3-1b-it`

**Requisitos:**
- GPU com pelo menos 16GB VRAM (com QLoRA 4-bit)
- `bitsandbytes` para quantização

### 4. DeBERTa

Modelo da Microsoft com atenção disentangled, excelente para classificação.

```bash
python src/models/deberta/train.py --config src/configs/deberta_config.yaml
```

**Modelos recomendados:**
- `microsoft/deberta-v3-base`
- `microsoft/mdeberta-v3-base` (multilingual)

## 📊 Notebooks

| Notebook | Descrição |
|----------|-----------|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Análise exploratória dos dados |
| [02_bert_baseline.ipynb](notebooks/02_bert_baseline.ipynb) | Baseline com BERT para submissão no Kaggle |

## ⚙️ Configurações

Edite os arquivos em `src/configs/` para ajustar hiperparâmetros:

```yaml
# src/configs/base_config.yaml
training:
  seed: 42
  num_folds: 5
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 5
  
model:
  name: "neuralmind/bert-base-portuguese-cased"
  max_length: 512
```

## 📈 Estratégias Recomendadas

1. **Tratamento de Desbalanceamento**:
   - Class weights na loss function
   - Focal Loss
   - Oversampling minoritário

2. **Ensemble**:
   - Média de probabilidades entre folds
   - Ensemble de diferentes arquiteturas (BERT + DeBERTa + T5)

3. **Data Augmentation**:
   - Back-translation
   - Synonym replacement

4. **Otimização**:
   - Learning rate warmup
   - Early stopping
   - Gradient accumulation para batch sizes maiores

## 🏆 Submissão no Kaggle

Esta é uma **Code Competition**. Seu notebook deve:

1. Carregar e treinar usando `train.csv`
2. Carregar o test set oculto em runtime
3. Gerar predições
4. Salvar `submission.csv`

```python
# Template para carregar test set
import os
import pandas as pd

test_path = "/kaggle/input/spr-2026-mammography-report-classification/test.csv"

if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
else:
    raise FileNotFoundError("test.csv only exists in evaluation runtime")
```

## 📝 Licença

Este projeto é distribuído sob a licença MIT.

## 🙏 Agradecimentos

- Sociedade Paulista de Radiologia (SPR)
- Instituições doadoras de dados: AC Camargo, Hapvida, Unifesp
- Organizadores: Dr. Felipe Kitamura, Dr. Eduardo Farina e equipe

## 📚 Referências

- [BI-RADS - American College of Radiology](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS)
- [BERTimbau - Portuguese BERT](https://github.com/neuralmind-ai/portuguese-bert)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

**Bom experimento! 🚀**
