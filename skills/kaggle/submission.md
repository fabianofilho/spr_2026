# Skill: Kaggle Submission Workflow

## 🎯 Quando Ativar

- Criar novo notebook para submissão
- Preparar notebook existente para Kaggle
- Organizar notebooks por categoria

---

## 📁 Estrutura de Pastas

| Pasta | Propósito | Versionamento |
|-------|-----------|---------------|
| `submit/` | Notebooks prontos para submeter | ✅ Git |
| `submissions/` | Cópia dos já submetidos | ❌ Local |

### Categorias em `submit/`:

```
submit/
├── tfidf/           # TF-IDF + modelos clássicos
├── word2vec/        # Word2Vec/FastText embeddings
├── transformers/    # BERTimbau, DeBERTa, etc.
├── sentence_transformers/  # SBERT
├── ensemble/        # Combinações de modelos
└── llm/             # Large Language Models
```

---

## 📋 Workflow Completo

### 1. Criar Notebook
```
Arquivo: submit/{categoria}/submit_{modelo}.ipynb
Exemplo: submit/tfidf/submit_tfidf_xgboost.ipynb
```

### 2. Usuário Submete no Kaggle
- Usa notebook de `submit/`
- Aguarda resultado

### 3. Usuário Reporta Score
- Informa score público ao agente

### 4. Agente Atualiza
- [ ] Copiar notebook para `submissions/`
- [ ] Atualizar `TODO.md` (Leaderboard)
- [ ] Marcar como `✅ Submetido`
- [ ] Commitar alterações

---

## 📝 Template: Atualização do TODO.md

```markdown
## Leaderboard (Public Score)

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 1 | TF-IDF + LinearSVC | **0.77885** | ✅ Submetido |
| NEW | TF-IDF + XGBoost | 0.XXXXX | ✅ Submetido |
```

---

## 📝 Nomenclatura de Notebooks

```
submit_{embedding}_{modelo}[_versao].ipynb
```

Exemplos:
- `submit_tfidf_linearsvc.ipynb`
- `submit_tfidf_xgboost_v2.ipynb`
- `submit_bertimbau_focal.ipynb`
- `submit_ensemble_voting.ipynb`

---

## ✅ Checklist de Submissão

### Antes de Submeter:
- [ ] Notebook funciona localmente
- [ ] Paths usando `/kaggle/input/`
- [ ] Internet desabilitada (offline)
- [ ] Modelos com `local_files_only=True`
- [ ] Output gera `submission.csv`

### Após Score Reportado:
- [ ] Copiar para `submissions/`
- [ ] Atualizar `TODO.md`
- [ ] Atualizar `insights/{categoria}.md`
- [ ] Commitar: `git add -A && git commit -m "score: {modelo} {score}"`
