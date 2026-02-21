# Data Augmentation - Estratégias para Retreino

## Objetivo
Aplicar data augmentation nos **melhores modelos** para aumentar a quantidade de dados de treino e melhorar generalização.

---

## ⏳ Status: Aguardando Resultados

> **Próximo passo:** Rodar todos os modelos base e identificar os top performers antes de aplicar augmentation.

### Melhores Modelos Atuais (Baseline)
| Rank | Modelo | Score | Categoria |
|------|--------|-------|-----------|
| 1 | TF-IDF + LinearSVC | **0.77885** | TF-IDF |
| 2 | TF-IDF + SGDClassifier | 0.75019 | TF-IDF |
| 3 | TF-IDF + LogReg | 0.72935 | TF-IDF |
| - | Transformers | ⏳ Pendente | - |
| - | Ensemble | ⏳ Pendente | - |

### Candidatos para Augmentation
Após rodar todos os modelos:
- [ ] Top 3 TF-IDF (já identificados)
- [ ] Melhor Transformer (BERTimbau, mDeBERTa, etc.)
- [ ] Melhor Ensemble

---

## Técnicas de Data Augmentation para NLP

### 1. Augmentation Textual (Sem Modelo)

| Técnica | Descrição | Complexidade | Offline |
|---------|-----------|--------------|---------|
| **Synonym Replacement** | Trocar palavras por sinônimos | Baixa | ✅ |
| **Random Insertion** | Inserir sinônimos aleatórios | Baixa | ✅ |
| **Random Swap** | Trocar posição de palavras | Baixa | ✅ |
| **Random Deletion** | Deletar palavras aleatórias | Baixa | ✅ |
| **EDA (Easy Data Augmentation)** | Combina as 4 técnicas acima | Baixa | ✅ |

```python
# Exemplo: EDA simples
def eda_augment(text, alpha=0.1):
    words = text.split()
    n = max(1, int(alpha * len(words)))
    
    # Random swap
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    
    return ' '.join(words)
```

### 2. Back Translation

| Técnica | Descrição | Complexidade | Offline |
|---------|-----------|--------------|---------|
| **PT → EN → PT** | Traduzir ida e volta | Média | ❌ |
| **PT → ES → PT** | Via espanhol (mais similar) | Média | ❌ |
| **Multi-pivot** | Múltiplos idiomas | Alta | ❌ |

> ⚠️ Requer modelo de tradução (MarianMT) ou API

### 3. Paraphrase Generation

| Técnica | Descrição | Complexidade | Offline |
|---------|-----------|--------------|---------|
| **T5 Paraphrase** | Gerar paráfrases com T5 | Alta | ⚠️ |
| **Pegasus** | Modelo especializado | Alta | ⚠️ |
| **ChatGPT/LLM** | Via API | Alta | ❌ |

### 4. Augmentation com Modelos de Linguagem

| Técnica | Descrição | Complexidade | Offline |
|---------|-----------|--------------|---------|
| **MLM Replacement** | BERT preenche [MASK] | Média | ✅ |
| **Contextual Word Embedding** | Substituição contextual | Média | ✅ |

```python
# Exemplo: MLM Augmentation
def mlm_augment(text, model, tokenizer, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    masked = [t if random.random() > mask_prob else '[MASK]' for t in tokens]
    # BERT prevê as posições mascaradas
    # Substituir [MASK] pelas previsões
```

### 5. Oversampling de Classes Minoritárias

| Técnica | Descrição | Complexidade | Offline |
|---------|-----------|--------------|---------|
| **Random Oversampling** | Duplicar exemplos | Baixa | ✅ |
| **SMOTE** | Interpolar embeddings | Média | ✅ |
| **ADASYN** | SMOTE adaptativo | Média | ✅ |

```python
# Para TF-IDF
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_tfidf, y)
```

### 6. Mixup / Cutout para Embeddings

| Técnica | Descrição | Complexidade | Offline |
|---------|-----------|--------------|---------|
| **Embedding Mixup** | Interpolar embeddings | Média | ✅ |
| **Manifold Mixup** | Mixup em camadas ocultas | Alta | ✅ |
| **Cutout** | Zerar partes do embedding | Baixa | ✅ |

---

## Datasets Externos para Augmentation

### Opção 1: Usar dados similares como augmentation

| Dataset | Uso | Kaggle Input |
|---------|-----|--------------|
| Medical Transcriptions | Textos médicos em inglês (traduzir) | `tboyle10/medicaltranscriptions` |
| VinDr-Mammo | Laudos mamográficos EN | `maedemaftouni/vindr-mammo` |
| CBIS-DDSM | Labels BI-RADS (metadata) | `awsaf49/cbis-ddsm-breast-cancer-image-dataset` |

### Opção 2: Pseudo-labeling

1. Treinar modelo no dataset da competição
2. Predizer em dataset externo relacionado
3. Usar predições confiantes (prob > 0.9) como dados extras

---

## Plano de Experimentos

### Fase 1: Identificar Melhores Modelos ⏳
- [ ] Rodar todos notebooks de transformers
- [ ] Rodar notebooks de ensemble
- [ ] Atualizar leaderboard

### Fase 2: Augmentation Simples (Offline)
- [ ] EDA (synonym/swap/delete) nos dados de treino
- [ ] Oversampling SMOTE para classes 5 e 6
- [ ] MLM augmentation com BERTimbau

### Fase 3: Augmentation Avançada
- [ ] Back-translation PT→EN→PT
- [ ] Pseudo-labeling com datasets externos
- [ ] Mixup em embeddings

### Fase 4: Retreino
- [ ] `submit_augmented_linearsvc.ipynb` (baseline + EDA)
- [ ] `submit_augmented_bertimbau.ipynb` (BERT + MLM aug)
- [ ] `submit_augmented_ensemble.ipynb` (melhor ensemble + aug)

---

## Referências

- [EDA: Easy Data Augmentation](https://arxiv.org/abs/1901.11196)
- [Back-Translation for NLP](https://arxiv.org/abs/1511.06709)
- [UDA: Unsupervised Data Augmentation](https://arxiv.org/abs/1904.12848)
- [Mixup for NLP](https://arxiv.org/abs/2004.12239)

---

*Atualizado em: 21/02/2026*
