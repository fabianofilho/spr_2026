# NEXT.md - Próximos Passos

> Baseado em 40+ submissões (Fev/2026)

## Estado Atual

| Métrica | Valor |
|---------|-------|
| Melhor Score | **0.79696** (BERTimbau + Focal Loss) |
| 2º Melhor | 0.78729 (Super Ensemble v1) |
| Única Melhoria v3 | 0.77036 (SGDClassifier v3, +2.7%) |
| Total Submissões | 40+ |

---

## 🎯 Estratégias Prioritárias

### 1. Threshold Tuning (ALTA PRIORIDADE)

**Hipótese:** Ajustar thresholds de decisão por classe pode melhorar F1-Macro em classes minoritárias.

```python
# Não mexer no modelo - apenas pós-processamento
thresholds = {
    0: 0.50, 1: 0.50, 2: 0.50, 
    3: 0.50, 4: 0.50, 
    5: 0.30,  # Mais sensível para classe minoritária
    6: 0.25   # Muito mais sensível
}
```

**Risco:** Baixo (não altera modelo treinado)

---

### 2. Ensemble v4 - Otimização de Pesos (MÉDIA PRIORIDADE)

**Hipótese:** Otimizar pesos do Super Ensemble com validação cruzada.

**Composição atual (v1 = 0.78729):**
- BERTimbau + Focal: 0.45
- LinearSVC: 0.25
- SGD v3: 0.20
- LogReg: 0.10

**Experimento:** Usar optuna para encontrar pesos ótimos.

**Risco:** Médio (overfitting nos pesos)

---

### 3. Focal Loss em Outros Transformers (BAIXA PRIORIDADE)

**Hipótese:** Replicar sucesso do BERTimbau em XLM-RoBERTa e ModernBERT.

| Modelo | Score Atual | Score Esperado |
|--------|-------------|----------------|
| XLM-RoBERTa | 0.68767 | ~0.74+ |
| ModernBERT | 0.68578 | ~0.74+ |

**Risco:** Médio (pode não funcionar igual)

---

## ❌ O Que Evitar

| Técnica | Por que falhou |
|---------|----------------|
| LoRA offline | Não funciona no Kaggle |
| LLMs zero/one-shot | Não entendem contexto médico |
| SMOTE | v4/v5 regrediram |
| Tratamento de texto | v5 com normalização piorou -2% |
| Muitas alterações | 3/5 resubmissões falharam |
| Iterar sobre sucesso | SGD v4/v5 pioraram vs v3 |

---

## 📋 Checklist de Submissão

### Antes de Submeter
- [ ] Testar localmente com CV (5-fold)
- [ ] Seed fixa para reprodutibilidade
- [ ] `local_files_only=True` em modelos HF
- [ ] UMA alteração por vez

### Durante Resubmissão
- [ ] Copiar notebook original (não editar)
- [ ] Documentar exatamente o que mudou
- [ ] Se der bom score, PARAR de iterar

---

## 🔑 Princípios Guia

1. **Conservadorismo:** Mudanças pequenas e incrementais
2. **Documentação:** Registrar toda alteração
3. **Validação:** CV antes de submeter
4. **Simplicidade:** Soft Voting > Stacking complexo
5. **Especialização:** Modelos PT > Multilingual
