# CLAUDE.md - Instruções para o Copilot

> 📁 **Skills disponíveis em [`skills/`](skills/README.md)** — Carregadas sob demanda, sem sobrecarregar a janela de contexto.

---

## ⚠️ Regras Obrigatórias

### 1. Sempre Commitar
```bash
git add -A && git commit -m "descrição breve"
```
→ [`skills/git/`](skills/git/README.md)

### 2. Modelos Offline
```python
local_files_only=True  # SEMPRE em from_pretrained()
```
→ [`skills/models/`](skills/models/huggingface.md)

---

## 📋 Skills por Domínio

| Domínio | Skill | Quando Ativar |
|---------|-------|---------------|
| **Git** | [git-workflow](skills/git/README.md) | Após qualquer modificação |
| **Kaggle** | [submission](skills/kaggle/submission.md) | Criar/submeter notebook |
| **Kaggle** | [insights](skills/kaggle/insights.md) | Após score reportado |
| **Kaggle** | [models](skills/kaggle/models.md) | Usar transformers offline |
| **Kaggle** | [finetuning](skills/kaggle/finetuning.md) | Treinar transformers (Focal Loss, Thresholds) |
| **Kaggle** | [llm-instruction](skills/kaggle/llm-instruction.md) | Usar LLMs com BI-RADS prompt |
| **Kaggle** | [resubmit](skills/kaggle/resubmit.md) | Organizar backlog de experimentos |
| **Modelos** | [huggingface](skills/models/huggingface.md) | Carregar modelos HF |
| **Modelos** | [upload](skills/models/upload.md) | Modelos não disponíveis |

---

## 📁 Estrutura do Projeto

```
submit/           # Notebooks para submeter (git)
submissions/      # Cópias submetidas (local)
insights/         # Análises por categoria
skills/           # Instruções sob demanda
```

**Categorias de modelos:** `tfidf/`, `word2vec/`, `transformers/`, `sentence_transformers/`, `ensemble/`, `llm/`

---

## 🔗 Referência Rápida

- **Baseline:** TF-IDF + LinearSVC = 0.77885
- **Melhor Score:** BERTimbau v4 = **0.82073**
- **Métrica:** F1-Macro
- **TrainingArguments:** usar `eval_strategy` (não `evaluation_strategy`)
- **Focal Loss:** γ=2.0, α=0.25 (obrigatório para transformers)
- **Threshold Tuning:** Otimizar por classe após treino
