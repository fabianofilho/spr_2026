# NEXT.md - Próximos Passos

> Baseado nos insights de 35+ submissões (incluindo resubmissões v2/v3)

## 📊 Resumo do Estado Atual

| Métrica | Valor |
|---------|-------|
| Melhor Score | **0.79696** (BERTimbau + Focal Loss) |
| 2º Melhor | **0.78729** (Super Ensemble v1) |
| 🚀 **Única Melhoria v3** | **0.77036** (SGDClassifier v3, +2.7%) |
| Total Submissões | 40+ |
| Submissões v4 testadas | 3 (todas regrediram) |
| Submissões com Falha | 7+ (LoRA, mDeBERTa, BioBERTpt v2, Custom v2, Qwen3, LLMs) |

---

## ✅ O Que Funciona (Replicar)

### Modelos que Mantiveram/Melhoraram
1. **BERTimbau + Focal Loss** → 0.79696 (CAMPEÃO)
2. **Super Ensemble v1** → 0.78729 (BERTimbau + TFIDF blend)
3. **Ensemble Soft Voting** → 0.78049 (baseline estável)
4. **TF-IDF + LinearSVC** → 0.77885 (baseline estável)
5. 🚀 **SGDClassifier v3** → **0.77036** (ÚNICO QUE MELHOROU! +2.7%)

### Técnicas Comprovadas
- **Focal Loss** com γ=2 funciona bem para classes desbalanceadas
- **RandomizedSearchCV** com 20+ iter → SGD v3 melhorou 2.7%!
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
| Qwen3 Zero-Shot | 0.13261 | LLM não entende contexto médico |
| Qwen3 One-Shot | 0.13261 | Mesmo com exemplo, não funciona |
| mDeBERTa | 0.01008 | Bug fp16 no Kaggle |

### Resubmissões que Regrediram
| Modelo | Baseline | Resubmit | Delta |
|--------|----------|----------|-------|
| LinearSVC v4 | 0.77885 | 0.77244 | -0.8% |
| Ensemble v3 | 0.78049 | 0.76567 | -1.9% |
| SGDClassifier v4 | 0.77036 | 0.76503 | -0.7% |
| LinearSVC v3 | 0.77885 | 0.75966 | -2.5% |
| LogisticRegression v3 | 0.72935 | 0.71303 | -2.2% |
| BERTimbau + Focal v3 | 0.79696 | 0.72625 | -8.9% |

### Anti-patterns
- ⚠️ **Muitas alterações de uma vez** → Quebraram 3 de 5 resubmissões
- ⚠️ **LoRA offline** → Não funciona no Kaggle
- ⚠️ **Modelos multilingual** → ~30% piores que PT nativo
- ⚠️ **LLMs zero/one-shot** → Não funcionam para este problema
- ⚠️ **RandomSearch em LinearSVC/LogReg** → Regrediu, não melhorou
- ⚠️ **SMOTE com classes desbalanceadas** → v4 regrediram vs v3
- ⚠️ **Iterar sobre modelo que já melhorou** → SGD v4 piorou vs v3

---

## 🎯 Próximos Experimentos Prioritários

### 1. ❌ SGDClassifier v4: Testado - Regrediu

**Resultado:** 0.76503 vs 0.77036 (v3) → -0.7%

**Lição:** RandomSearch + SMOTE não melhorou. v3 já era ótimo.

---

### 2. ❌ LinearSVC v4: Testado - Regrediu

**Resultado:** 0.77244 vs 0.77885 (original) → -0.8%

**Lição:** Calibração + Platt não ajudou. Baseline é melhor.

---

### 3. ❌ Ensemble v3: Testado - Regrediu

**Resultado:** 0.76567 vs 0.78049 (original) → -1.9%

**Lição:** Ajustes de pesos não melhoraram.

---

### 4. BERTimbau v4: Threshold Tuning Apenas (ALTA PRIORIDADE)

**Hipótese:** Ajustar thresholds na inferência pode melhorar F1-Macro

```python
# NÃO MEXER NO MODELO - apenas pós-processamento
thresholds = {
    0: 0.50, 1: 0.50, 2: 0.50, 
    3: 0.50, 4: 0.50, 
    5: 0.30,  # Mais sensível para classe minoritária
    6: 0.25   # Muito mais sensível
}
```

**Cuidados:**
- Usar modelo EXATAMENTE como está
- Apenas ajustar thresholds na predição final

---

### 4. Focal Loss em Transformers (MÉDIA PRIORIDADE)

**Hipótese:** Copiar config EXATA do BERTimbau em outros transformers

**Candidatos:**
| Modelo | Score Atual | Score Esperado |
|--------|-------------|----------------|
| XLM-RoBERTa | 0.68767 | ~0.74+ |
| ModernBERT | 0.68578 | ~0.74+ |

**Cuidados:**
- COPIAR EXATAMENTE config do BERTimbau, não inventar
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
*Atualizado em: 28/02/2026*
*Baseado em: 40+ submissões no Kaggle SPR 2026*
