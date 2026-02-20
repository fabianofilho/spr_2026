# Datasets para Pré-Treinamento

## Objetivo
Usar datasets externos do Kaggle para pré-treinar modelos antes de fine-tuning no dataset da competição SPR 2026 (classificação de laudos mamográficos em categorias BI-RADS).

## Como usar no Kaggle
1. Add Input → **Datasets** → pesquisar o dataset
2. O dataset fica disponível em `/kaggle/input/{dataset-slug}/`

---

## Datasets Recomendados

### 🏥 Radiologia / Mamografia

| Dataset | Slug Kaggle | Descrição | Tamanho |
|---------|-------------|-----------|---------|
| CBIS-DDSM | `awsaf49/cbis-ddsm-breast-cancer-image-dataset` | Mammography dataset com labels BI-RADS | ~3GB |
| VinDr-Mammo | `maedemaftouni/vindr-mammo` | Laudos mamográficos em inglês | ~1GB |
| Breast Cancer Wisconsin | `uciml/breast-cancer-wisconsin-data` | Classificação binário (benigno/maligno) | 24KB |
| INbreast | `ramanathansp20/inbreast-dataset` | Mamografias com BI-RADS | ~2GB |

### 📝 NLP Médico (Inglês)

| Dataset | Slug Kaggle | Descrição | Tamanho |
|---------|-------------|-----------|---------|
| Medical Transcriptions | `tboyle10/medicaltranscriptions` | 5000+ transcrições médicas | 20MB |
| Medical NER | `finalepoch/medical-ner` | Named Entity Recognition médico | 5MB |
| PubMed 200k RCT | `owaiskhan9654/pubmed-200k-rct` | Abstracts de artigos médicos | ~100MB |
| Medical QA | `jpmiller/layoutlm` | Perguntas e respostas médicas | 10MB |
| MIMIC-III Clinical Notes | `drscarlat/mimic3` | Notas clínicas (requer aprovação) | ~4GB |
| Medical Specialty Classification | `chaitanyakck/medical-text` | Classificação de especialidade médica | 5MB |
| Symptom2Disease | `itachi9604/disease-symptom-description-dataset` | Sintomas → Diagnóstico | 1MB |
| Drug Review | `jessicali9530/kuc-hackathon-winter-2018` | Reviews de medicamentos | 50MB |

### 🇧🇷 Português

| Dataset | Slug Kaggle | Descrição | Tamanho |
|---------|-------------|-----------|---------|
| Portuguese Medical Corpus | `a pesquisar` | Corpus médico em PT-BR | - |
| Wikipedia PT | `ltcmdrdata/wikipedia-ptbr` | Para domain adaptation | ~2GB |
| Brazilian News | `diogocaliman/brazilian-news` | Notícias em PT-BR | 100MB |
| B2W Reviews | `olistbr/brazilian-ecommerce` | Reviews em PT-BR | 50MB |

### 🎯 Classificação de Texto (Transfer Learning)

| Dataset | Slug Kaggle | Descrição | Tamanho |
|---------|-------------|-----------|---------|
| IMDB Reviews | `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` | Sentimento (bom para base) | 60MB |
| AG News | `amananandrai/ag-news-classification-dataset` | Classificação de notícias | 30MB |
| Yelp Reviews | `yelp-dataset/yelp-dataset` | Reviews multi-classe | 5GB |
| Amazon Reviews | `bittlingmayer/amazonreviews` | Reviews multi-classe | 600MB |

---

## Estratégias de Pré-Treinamento

### 1. MLM (Masked Language Modeling)
- Usar corpus médico/radiológico para continuar pré-treino de BERT/BERTimbau
- Datasets: Medical Transcriptions, PubMed, MIMIC-III

### 2. Domain Adaptation
- Fine-tune em tarefa similar antes do dataset alvo
- Datasets: Medical Specialty Classification, Breast Cancer Wisconsin

### 3. Multi-Task Learning
- Treinar em múltiplas tarefas médicas simultaneamente
- Datasets: Medical NER + Classification + QA

### 4. Data Augmentation
- Usar dados similares para aumentar o treino
- Datasets: VinDr-Mammo, CBIS-DDSM

---

## Modelos a Testar (aguardando seleção)

| Modelo | Tipo | Status |
|--------|------|--------|
| BERTimbau + MLM médico | Transformer | ⏳ Aguardando |
| mDeBERTa + Domain Adaptation | Transformer | ⏳ Aguardando |
| Sentence-BERT + Contrastive | Sentence Transformer | ⏳ Aguardando |
| LLM Fine-tuned | LLM | ⏳ Aguardando |

---

## Notas

- ⚠️ Verificar licenças antes de usar
- ⚠️ Alguns datasets requerem aprovação (MIMIC-III)
- ⚠️ Datasets grandes podem estourar memória no Kaggle (16GB RAM)
- 💡 Preferir datasets em português ou multilíngues quando possível
- 💡 Começar com datasets menores para validar a abordagem
