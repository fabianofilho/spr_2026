# NEXT.md - Próximos Passos

> Baseado nos insights de 32 submissões (incluindo 5 resubmissões)

## 📊 Resumo do Estado Atual

| Métrica | Valor |
|---------|-------|
| Melhor Score | **0.79696** (BERTimbau + Focal Loss) |
| 2º Melhor | 0.79505 (BERTimbau + Focal Loss v2) |
| Total Submissões | 32 |
| Submissões com Falha | 5 (LoRA, mDeBERTa, BioBERTpt v2, Custom v2) |

---

## ✅ O Que Funciona (Não Mexer)

### Modelos Estáveis
1. **BERTimbau + Focal Loss** → 0.79696 (CAMPEÃO)
2. **BERTimbau + Focal Loss v2** → 0.79505 (99.8% do original)
3. **Ensemble Soft Voting** → 0.78049
4. **TF-IDF + LinearSVC** → 0.77885

### Técnicas Comprovadas
- **Focal Loss** com γ=2 funciona bem para classes desbalanceadas
- **Class weights** como fallback para modelos clássicos
- **Soft Voting** entre modelos TF-IDF diversos
- **BERTimbau** supera modelos multilingual

---

## ❌ O Que Não Funciona (Evitar)

### Modelos que Falharam
| Modelo | Score | Motivo |
|--------|-------|--------|
| BioBERTpt + Focal v2 | 0.26099 | Focal Loss mal calibrada |
| Custom Transformer v2 | 0.41721 | Alterações quebraram tokenizer |
| BERTimbau + LoRA | 0.13261 | Offline não funciona |
| mDeBERTa | 0.01008 | Bug fp16 no Kaggle |

### Anti-patterns
- ⚠️ **Muitas alterações de uma vez** → Quebraram 3 de 5 resubmissões
- ⚠️ **LoRA offline** → Não funciona no Kaggle
- ⚠️ **Modelos multilingual** → ~30% piores que PT nativo
- ⚠️ **Word2Vec média** → Dilui informação discriminativa

---

## 🎯 Próximos Experimentos Prioritários

### 1. Ensemble com BERTimbau (ALTA PRIORIDADE)

**Hipótese:** Combinar BERTimbau + Focal com TF-IDF Ensemble pode superar 0.80

```python
# Proposta: Weighted Blend
final_pred = 0.6 * bertimbau_probs + 0.4 * tfidf_ensemble_probs
```

**Notebook:** `submit/ensemble/submit_ensemble_bertimbau_tfidf.ipynb`

**Cuidados:**
- Usar mesma seed do BERTimbau original
- Não alterar hiperparâmetros do Focal Loss
- Testar pesos: 0.5/0.5, 0.6/0.4, 0.7/0.3

---

### 2. Data Augmentation para Classes 5 e 6 (MÉDIA PRIORIDADE)

**Hipótese:** Aumentar samples das classes minoritárias pode melhorar F1-Macro

**Técnicas a testar:**
- [ ] EDA (Easy Data Augmentation) - synonym replacement
- [ ] Back-translation PT→EN→PT
- [ ] SMOTE no espaço de embeddings do BERTimbau

**Notebook:** `tests/augmented/submit_augmented_bertimbau.ipynb`

**Cuidados:**
- Augmentar APENAS treino, não validação
- Monitorar overfitting nas classes aumentadas
- Usar augmentation conservadora (max 2x samples)

---

### 3. Focal Loss em Outros Modelos (MÉDIA PRIORIDADE)

**Hipótese:** Focal Loss pode melhorar modelos abaixo de 0.78

**Candidatos:**
| Modelo | Score Atual | Score Esperado |
|--------|-------------|----------------|
| BioBERTpt | 0.72480 | ~0.76+ |
| XLM-RoBERTa | 0.68767 | ~0.72+ |
| ModernBERT | 0.68578 | ~0.72+ |

**Cuidados:**
- Copiar EXATAMENTE a configuração do BERTimbau + Focal
- γ=2, sem alterações
- Uma alteração por vez

---

### 4. Pre-training com MLM Médico (BAIXA PRIORIDADE)

**Hipótese:** Continuar pré-treino do BERTimbau com textos médicos PT

**Datasets disponíveis:**
- Medical Transcriptions
- PubMed 200k RCT
- CBIS-DDSM BI-RADS

**Notebook:** `tests/pretrain/submit_bertimbau_pretrain.ipynb`

**Cuidados:**
- Requer muito compute (2-4h no Kaggle P100)
- Risco de catastrophic forgetting
- Testar com learning rate menor (1e-5)

---

### 5. NER + Regras Determinísticas (BAIXA PRIORIDADE)

**Motivação:** Alguns BI-RADS (0 e 6) têm padrões textuais óbvios

**Pipeline:**
1. Regex para extrair "BIRADS 0", "BIRADS 6" explícitos
2. NER para identificar "negação" + "achado"
3. Fallback para modelo ML

**Notebook:** `tests/ner/submit_ner_rules.ipynb`

---

## 📋 Checklist de Boas Práticas

### Antes de Submeter
- [ ] Testar localmente com CV (5-fold)
- [ ] Verificar seed fixa para reprodutibilidade
- [ ] Validar `local_files_only=True` para modelos HuggingFace
- [ ] Confirmar que não há alterações vs versão que funcionou

### Durante Resubmissão
- [ ] UMA alteração por vez
- [ ] Copiar notebook original, não editar
- [ ] Documentar exatamente o que mudou
- [ ] Comparar scores antes de submeter

### Debugging de Falhas
- [ ] Verificar logs do Kaggle para errors
- [ ] Testar com inference local
- [ ] Comparar tokenização de inputs
- [ ] Verificar shapes dos outputs

---

## 📈 Roadmap Sugerido

| Semana | Foco | Objetivo |
|--------|------|----------|
| 1 | Ensemble BERTimbau + TF-IDF | Superar 0.80 |
| 2 | Data Augmentation | +0.5-1% no F1-Macro |
| 3 | Focal Loss em outros modelos | Diversificar top 5 |
| 4 | Fine-tuning & Otimização | Consolidar ganhos |

---

## 🔑 Conclusões Principais

1. **BERTimbau + Focal Loss é o padrão ouro** - Não alterar sem necessidade
2. **Resubmissões são arriscadas** - 3 de 5 falharam (60%)
3. **Modelos PT nativos > Multilingual** - Sempre priorizar
4. **Focal Loss > Class Weights** - Para classes desbalanceadas
5. **Ensemble simples funciona** - Soft Voting supera Stacking
6. **TF-IDF ainda é forte** - Baseline difícil de bater

---

*Criado em: 25/02/2026*
*Baseado em: 32 submissões no Kaggle SPR 2026*
