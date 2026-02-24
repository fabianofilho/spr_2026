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
|---------|-----------|---------------|
| [transformers.md](transformers.md) | Transformers | **0.79696** 🏆 |
| [ensemble.md](ensemble.md) | Ensemble | 0.78049 |
| [tfidf.md](tfidf.md) | TF-IDF | 0.77885 |
| [sentence_transformers.md](sentence_transformers.md) | SBERT/Custom | 0.77272 |
| [word2vec.md](word2vec.md) | Word2Vec | 0.66385 |

---

## 🏆 Top 4 Modelos - O Que Funciona

| Rank | Modelo | Score | Categoria |
|------|--------|-------|----------|
| 1 | BERTimbau + Focal Loss | **0.79696** | Transformers |
| 2 | Ensemble Soft Voting | 0.78049 | Ensemble |
| 3 | TF-IDF + LinearSVC | 0.77885 | TF-IDF |
| 4 | Custom Transformer Encoder | 0.77272 | Custom |

---

## 🔑 Fatores de Sucesso Identificados

### 1. **Tratamento de Desbalanceamento de Classes**

O dataset tem classes minoritárias (BI-RADS 5 e 6) que são críticas para o F1-Macro.

| Modelo | Técnica | Impacto |
|--------|---------|----------|
| BERTimbau + Focal | Focal Loss (γ=2) | +15% nas classes raras |
| LinearSVC | class_weight='balanced' | Baseline sólido |
| Ensemble | Voting suaviza erros | Reduz falsos negativos |

**Insight:** Focal Loss > Class Weights > Nada

### 2. **Preservação de Termos Específicos**

Termos como "BIRADS", "calcificação", "nódulo espiculado" são altamente discriminativos.

| Abordagem | Preserva Termos? | Score |
|-----------|-----------------|-------|
| TF-IDF | ✅ Sim (exata) | 0.778 |
| BERTimbau | ✅ Sim (contextual) | 0.797 |
| Word2Vec (média) | ❌ Dilui | 0.664 |

**Insight:** Métodos que preservam lexicalidade vencem.

### 3. **Transfer Learning em Português**

| Modelo | Língua | Score |
|--------|--------|-------|
| BERTimbau (PT) | Português | **0.797** |
| BioBERTpt (PT) | Português | 0.725 |
| ModernBERT (EN) | Inglês | 0.686 |
| BERT Multilingual | Multi | 0.561 |

**Insight:** Modelos nativos PT > Multilingual > Inglês

### 4. **Diversidade no Ensemble**

Soft Voting combina modelos com erros descorrelacionados:

```
Ensemble Soft Voting (0.78049):
├─ TF-IDF + LinearSVC (0.778) → Captura termos exatos
├─ TF-IDF + SGD (0.750) → Regularização diferente
└─ TF-IDF + LogReg (0.729) → Probabilidades calibradas
```

**Insight:** Voting > Modelo único quando modelos são diversos.

### 5. **Arquitetura Custom vs Pré-treinado**

Custom Transformer Encoder (0.77272) é competitivo sem pré-treino!

| Vantagem | Descrição |
|----------|----------|
| Tokenizer específico | Vocabulário do domínio médico |
| Sem overhead | Modelo menor e mais rápido |
| Sem transfer gap | Treinado direto no task |

**Insight:** Custom from scratch pode superar modelos multilingual.

---

## ❌ O Que Não Funciona

### 1. Word2Vec Média
- Média de embeddings dilui informação discriminativa
- Vocabulário médico sub-representado
- Score: 0.56-0.66 (30% abaixo do baseline)

### 2. Modelos Multilingual
- BERT Multilingual (0.561) e DistilBERT (0.552) decepcionam
- Tokenização genérica perde termos médicos PT

### 3. LoRA Offline
- BERTimbau + LoRA: 0.132 (FALHA)
- Adapters não salvam corretamente offline
- Full fine-tuning é mais confiável

### 4. mDeBERTa + fp16
- Bug de mixed precision no Kaggle
- Score: 0.01 (quase aleatório)

---

## 💡 Recomendações

### Para melhorar ainda mais:

1. **Ensemble com BERTimbau + Focal**
   - Adicionar ao Soft Voting pode superar 0.80
   
2. **Focal Loss em outros modelos**
   - Aplicar em BioBERTpt, Custom Transformer
   
3. **Data Augmentation**
   - EDA para classes 5 e 6 (minoritárias)
   - SMOTE no espaço de embeddings
   
4. **Pré-treino domínio**
   - MLM com datasets médicos PT
   - PubMed PT, Medical Transcriptions

---

## Resumo Executivo

### Por que BERTimbau + Focal Loss lidera?

1. **Língua nativa:** Treinado em português, entende nuances
2. **Focal Loss:** Foca em exemplos difíceis/raros
3. **Fine-tuning:** Transfer learning efetivo
4. **Contexto:** Captura "sem sinais de malignidade" vs "com sinais"

### Por que Ensemble é vice-campeão?

Combinação de TF-IDF + SVC/SGD/LogReg tem erros descorrelacionados.
Soft voting suaviza predições e melhora robustez.

### Por que TF-IDF ainda é top 3?

Termos médicos são altamente discriminativos. "BIRADS 4" sozinho
classifica corretamente. TF-IDF captura isso diretamente.

---

*Atualizado em: 24/02/2026*
