# Validation - Testes Offline com Dataset de Treino

Notebooks para **validar arquiteturas ANTES de submeter**, usando apenas o dataset de treino dividido em treino/validação.

---

## 🎯 Objetivo

1. **Testar modelos promissores** que falharam ou nunca foram testados corretamente
2. **Comparar configurações** (Focal Loss, class weights, pooling, etc.)
3. **Gerar insights** para documentar em `insights/`
4. **Economizar submissões** (5 por dia são limitadas)

---

## 📁 Notebooks

| Notebook | Modelo | Prioridade | Status |
|----------|--------|------------|--------|
| `val_mdeberta.ipynb` | mDeBERTa-v3 FP32 fix | ⭐⭐⭐ ALTA | Nunca testado corretamente |
| `val_xlmroberta.ipynb` | XLM-RoBERTa | ⭐⭐ MÉDIA | Testar pooling strategies |
| `val_biobertpt.ipynb` | BioBERTpt + Focal | ⭐⭐ MÉDIA | Domínio médico + Focal |
| `val_bertimbau_large.ipynb` | BERTimbau Large | ⭐⭐ MÉDIA | Maior capacidade |
| `val_modernbert.ipynb` | ModernBERT | ⭐ BAIXA | Flash Attention + RoPE |
| `val_distilbert.ipynb` | DistilBERT | ⭐ BAIXA | Rápido, baseline |
| `val_sentence_transformers.ipynb` | SBERT | ⭐ BAIXA | Embeddings para ensemble |
| `val_llm_zeroshot.ipynb` | Qwen/MedGemma | ⭐ BAIXA | Zero/Few-shot |

---

## 📊 Metodologia

### Split de Dados
```python
# 80/20 split estratificado
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['report'].tolist(),
    train_df['target'].tolist(),
    test_size=0.2,
    stratify=train_df['target'],
    random_state=42
)
```

### Métricas Coletadas
- **F1-Macro** (métrica da competição)
- **F1 por classe** (identificar classes problemáticas)
- **Confusion Matrix** (entender erros)
- **Tempo de treino** (viabilidade no Kaggle)

### Configurações Testadas por Modelo
1. **Loss Functions:** CrossEntropy, Focal Loss (γ=1,2,3), Label Smoothing
2. **Class Weights:** None, Balanced, Custom
3. **Learning Rates:** 1e-5, 2e-5, 3e-5, 5e-5
4. **Batch Sizes:** 8, 16, 32
5. **Max Length:** 128, 256, 512
6. **Pooling:** CLS, Mean, Max (quando aplicável)

---

## 🏆 Baseline de Referência

| Modelo | Val F1-Macro | Test Score |
|--------|--------------|------------|
| TF-IDF + LinearSVC | ~0.78 | 0.77885 |
| BERTimbau Focal v4 | ~0.82 | **0.82073** |

**Objetivo:** Encontrar configurações com Val F1 > 0.82 para superar o campeão atual.

---

## 📋 Checklist por Notebook

- [ ] Carregar modelo `local_files_only=True`
- [ ] Testar múltiplas configurações
- [ ] Salvar métricas em tabela
- [ ] Gerar confusion matrix
- [ ] Identificar melhor configuração
- [ ] Documentar insights em `insights/`

---

## 🔗 Links

- **Insights:** [insights/](../insights/)
- **Submissões:** [resubmit/](../resubmit/)
- **Modelos:** [models/](../models/)

---

*Criado em: 01/03/2026*
