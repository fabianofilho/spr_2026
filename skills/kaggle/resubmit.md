# Skill: Resubmit Workflow

## 🎯 Quando Ativar

- Organizar notebooks para resubmissão
- Criar backlog de experimentos
- Preparar pastas diárias de trabalho

---

## 📁 Estrutura de Pastas

```
resubmit/
├── README.md           # Índice e status
├── backlog/            # Notebooks pendentes (não organizados por dia)
│   └── *.ipynb
├── 2026-02-24/         # Notebooks de um dia específico
│   └── *.ipynb
├── 2026-02-25/
│   └── *.ipynb
└── ...
```

---

## 📋 Workflow de Resubmissão

### 1. Criar Notebook no Backlog
```bash
resubmit/backlog/resubmit_{modelo}_{variacao}.ipynb
```

Exemplos:
- `resubmit_bertimbau_focal_v3.ipynb`
- `resubmit_bertimbau_threshold_grid.ipynb`
- `resubmit_qwen_birads_instruction.ipynb`

### 2. Preparar Pasta do Dia (Priorização)
```bash
mkdir resubmit/YYYY-MM-DD
cp resubmit/backlog/notebook_prioritario.ipynb resubmit/YYYY-MM-DD/
```

### 3. Submeter no Kaggle
- Upload dos notebooks da pasta do dia
- Executar e aguardar score

### 4. Documentar Resultado
- Atualizar `TODO.md` com score
- Marcar notebook como `✅ Submetido` no backlog/README.md

---

## 📝 Nomenclatura de Notebooks

```
resubmit_{modelo}_{tecnica}[_versao].ipynb
```

| Componente | Exemplos |
|------------|----------|
| modelo | `bertimbau`, `biobertpt`, `qwen`, `medgemma` |
| tecnica | `focal`, `threshold`, `cv`, `ensemble`, `birads` |
| versao | `v2`, `v3`, `v4` |

Exemplos:
- `resubmit_bertimbau_focal_v3.ipynb`
- `resubmit_bertimbau_v5_threshold_grid.ipynb`
- `resubmit_qwen_birads_instruction.ipynb`
- `resubmit_medgemma_4b_it.ipynb`

---

## 📊 Priorização de Notebooks

### ALTA Prioridade (rodar primeiro):
- Variações do melhor modelo (BERTimbau v4+)
- Threshold tuning com novos valores
- Seed ensembles

### MÉDIA Prioridade:
- Novos modelos (MedGemma, BioBERTpt)
- Técnicas experimentais (Label Smoothing)

### BAIXA Prioridade:
- Modelos já testados sem sucesso
- Técnicas que regrediram (SMOTE)

---

## 🔧 Comandos Úteis

### Criar pasta do dia:
```bash
mkdir -p resubmit/$(date +%Y-%m-%d)
```

### Copiar notebooks prioritários:
```bash
cp resubmit/backlog/resubmit_bertimbau_*.ipynb resubmit/$(date +%Y-%m-%d)/
```

### Listar backlog:
```bash
ls -la resubmit/backlog/
```

### Contar notebooks por pasta:
```bash
find resubmit -name "*.ipynb" | wc -l
```

---

## 📝 Template: README do Backlog

```markdown
# Backlog de Resubmissões

## 🏆 Melhor Score Atual
- **BERTimbau v4**: 0.82073

## 📋 Status dos Notebooks

### ✅ Submetidos
| Notebook | Score | Data |
|----------|-------|------|
| bertimbau_focal_v3 | 0.82073 | 2026-02-28 |

### 🔄 Pendentes (Alta Prioridade)
- [ ] resubmit_bertimbau_ultimate_v6.ipynb
- [ ] resubmit_bertimbau_v5_threshold_grid.ipynb

### 📝 Backlog (Média Prioridade)
- [ ] resubmit_medgemma_birads.ipynb
- [ ] resubmit_qwen_birads_instruction.ipynb

### 🔬 Experimental (Baixa Prioridade)
- [ ] resubmit_biogpt_large.ipynb
- [ ] resubmit_clinicalbert_finetune.ipynb
```

---

## ⚠️ Boas Práticas

1. **Não deletar notebooks do backlog** - manter histórico
2. **Copiar, não mover** - backlog é referência
3. **Versionar notebooks** - v2, v3, v4 para iterações
4. **Documentar score no nome** - `_score_082073.ipynb` se necessário
5. **Commitar após cada sessão**

---

## 📋 Checklist Diário

### Início do dia:
- [ ] Criar pasta `resubmit/YYYY-MM-DD/`
- [ ] Copiar notebooks prioritários
- [ ] Verificar dependências (modelos disponíveis)

### Durante o dia:
- [ ] Submeter notebooks
- [ ] Anotar scores
- [ ] Iterar se necessário

### Final do dia:
- [ ] Atualizar TODO.md com scores
- [ ] Atualizar backlog/README.md
- [ ] Commitar: `git add -A && git commit -m "resubmit: resultados YYYY-MM-DD"`
