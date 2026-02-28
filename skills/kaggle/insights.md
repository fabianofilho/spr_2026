# Skill: Insights Workflow

## 🎯 Quando Ativar

- Após usuário reportar score público
- Para documentar análise de resultados
- Para comparar modelos e estratégias

---

## 📁 Estrutura

```
insights/
├── README.md               # Índice geral
├── tfidf.md               # Análise TF-IDF
├── word2vec.md            # Análise Word2Vec
├── transformers.md        # Análise Transformers
├── sentence_transformers.md  # Análise SBERT
└── ensemble.md            # Análise Ensemble
```

---

## 📋 Workflow Completo

### 1. Usuário Reporta Score
```
"O score público foi 0.75019"
```

### 2. Agente Atualiza

- [ ] `TODO.md` → Atualizar Leaderboard
- [ ] `insights/{categoria}.md` → Análise do resultado
- [ ] Comparação com outros modelos
- [ ] Hipóteses sobre o que funcionou/não funcionou
- [ ] Commitar alterações

---

## 📝 O Que Documentar

### Em cada arquivo de insight:

1. **Ranking e Scores**
   - Tabela com todos os modelos testados
   - Score público de cada um

2. **Por que funcionou/não funcionou**
   - Análise metodológica
   - Fatores que influenciaram

3. **Comparação com Baseline**
   - TF-IDF + LinearSVC: **0.77885**
   - Diferença percentual

4. **Recomendações**
   - Próximos passos
   - O que tentar/evitar

---

## 📝 Template: Entrada de Insight

```markdown
## {Nome do Modelo}

**Score Público:** {score}
**Data:** {YYYY-MM-DD}
**Rank no Leaderboard:** {posição}

### Configuração
- Embedding: {tipo}
- Classificador: {modelo}
- Hiperparâmetros: {principais}

### Análise
{Por que esse resultado? O que funcionou/não funcionou?}

### Comparação com Baseline
- Baseline (LinearSVC): 0.77885
- Diferença: {+X.XX% ou -X.XX%}

### Próximos Passos
- {Recomendação 1}
- {Recomendação 2}
```

---

## 📊 Métricas Importantes

| Métrica | Descrição |
|---------|-----------|
| Score Público | F1-Macro no test set público |
| Score Privado | F1-Macro no test set privado (após competição) |
| Baseline | TF-IDF + LinearSVC = **0.77885** |

---

## ✅ Checklist de Insights

- [ ] Score adicionado ao TODO.md
- [ ] Entrada criada em insights/{categoria}.md
- [ ] Análise de por que funcionou/não funcionou
- [ ] Comparação com baseline documentada
- [ ] Recomendações para próximos passos
- [ ] Commit realizado
