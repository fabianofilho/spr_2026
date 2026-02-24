# Sentence Transformers - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 🔥 | **Custom Transformer Encoder** | **0.77272** | ✅ Submetido |
| 2 | SBERT + LightGBM | 0.48376 | ✅ Submetido |

---

## 🔥 Análise: Custom Transformer Encoder (0.77272) - TOP 4!

**Surprise hit!** Um transformer from scratch competiu com modelos pré-treinados!

### Por que funcionou tão bem?

1. **Tokenizer específico:** Vocabulário construído do dataset médico
2. **Sem transfer gap:** Treinado diretamente no task
3. **Arquitetura otimizada:** Modelo menor, mais eficiente
4. **Embeddings limpos:** Sem ruído de domínios irrelevantes

### Comparação com pré-treinados

| Modelo | Score | Pré-treino |
|--------|-------|------------|
| Custom Transformer | **0.77272** | ❌ Nenhum |
| BioBERTpt | 0.72480 | ✅ Médico PT |
| ModernBERT | 0.68578 | ✅ Geral EN |
| BERT Multilingual | 0.56095 | ✅ Multi |

**Insight:** Para datasets pequenos e específicos, treinar from scratch pode superar transfer learning!

### Configuração

```python
# Arquitetura
vocab_size = ~5000  # Vocabulário do dataset
embedding_dim = 256
num_heads = 4
num_layers = 2
max_len = 256

# Treinamento
epochs = 50
batch_size = 32
lr = 1e-3
```

### Por que superou modelos multilingual?

1. **Vocabulário ajustado:** "BIRADS" é um token único, não "BI" + "##RADS"
2. **Sem overhead:** Modelo 50x menor que BERT
3. **Rápido para iterar:** Treina em minutos, não horas

---

## ⚠️ Análise: SBERT + LightGBM (0.48376) - DECEPÇÃO

**SBERT multilingual falhou.** Score de 0.48376 é pior que Word2Vec.

### Por que falhou?

1. **Embeddings genéricos:** Otimizado para similaridade, não classificação
2. **Vocabulário multilingual:** Perde especificidade de termos médicos PT
3. **Representação densa:** Mesmo problema do Word2Vec

### Comparação com Word2Vec

| Modelo | Score | Representação |
|--------|-------|---------------|
| Word2Vec + XGBoost | 0.66385 | Média 300D |
| SBERT + LightGBM | 0.48376 | Sentença 384D |

**Insight:** SBERT multilingual é pior que Word2Vec para este task!

### Lições aprendidas

- SBERT funciona para similaridade semântica, não classificação
- Embeddings pré-treinados multilingual perdem especificidade
- Para classificação, TF-IDF ou custom transformer são melhores

---

## 💡 Próximos Passos

### Para melhorar SBERT:
1. Usar modelo PT-BR específico (se existir)
2. Adicionar TF-IDF features híbridas
3. Fine-tuning contrastivo no dataset

### Para melhorar Custom Transformer:
1. Adicionar Focal Loss (como BERTimbau)
2. Aumentar profundidade (4-6 layers)
3. Data augmentation (EDA)

---

*Atualizado em: 24/02/2026*
