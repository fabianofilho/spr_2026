# Skills - Sistema de Habilidades do Agente

Este diretório contém **habilidades (skills)** que funcionam como instruções, scripts e recursos que permitem ao agente trabalhar com mais precisão e eficiência.

As skills são **ativadas sob demanda** quando o pedido coincide com a descrição de uma habilidade disponível.

---

## 📋 Índice de Habilidades

| Skill | Domínio | Descrição | Trigger |
|-------|---------|-----------|---------|
| [git-workflow](git/README.md) | Versionamento | Commits obrigatórios e padrões | Após qualquer modificação |
| [kaggle-submission](kaggle/submission.md) | Kaggle | Workflow de submissão de notebooks | Criar/submeter notebook |
| [kaggle-insights](kaggle/insights.md) | Kaggle | Documentação de resultados e análises | Após score reportado |
| [kaggle-models](kaggle/models.md) | Kaggle | Uso de modelos offline no Kaggle | Notebooks com transformers |
| [kaggle-finetuning](kaggle/finetuning.md) | Kaggle | Fine-tuning: Focal Loss, Thresholds, Ensembles | Treinar transformers |
| [kaggle-llm-instruction](kaggle/llm-instruction.md) | Kaggle | LLMs com prompt BI-RADS | Usar LLMs para classificação |
| [kaggle-resubmit](kaggle/resubmit.md) | Kaggle | Workflow de resubmissão e backlog | Organizar experimentos |
| [huggingface-local](models/huggingface.md) | Modelos | Configurações HuggingFace local/offline | Carregar modelos |
| [model-upload](models/upload.md) | Modelos | Upload manual para Kaggle Datasets | Modelos não disponíveis |

---

## 🔧 Como Usar

### Para o Agente (Claude):
1. Identificar qual skill se aplica ao pedido
2. Consultar os arquivos da skill para instruções detalhadas
3. Seguir o checklist e templates disponíveis
4. Executar scripts auxiliares quando disponíveis

### Para o Usuário:
- As skills são ativadas automaticamente pelo contexto
- Pode referenciar diretamente: *"usa a skill de submission"*
- Cada skill tem exemplos e templates prontos

---

## 📁 Estrutura

```
skills/
├── README.md           # Este arquivo (índice)
├── git/
│   ├── README.md       # Workflow de commits
│   └── hooks/          # Scripts git (futuro)
├── kaggle/
│   ├── submission.md   # Workflow de submissão
│   ├── insights.md     # Workflow de insights
│   ├── models.md       # Uso de modelos offline
│   ├── finetuning.md   # Fine-tuning: Focal Loss, Thresholds
│   ├── llm-instruction.md  # LLMs com BI-RADS prompt
│   ├── resubmit.md     # Workflow de resubmissão
│   └── templates/      # Templates de notebooks
└── models/
    ├── huggingface.md  # Configurações HF
    ├── upload.md       # Upload manual
    └── scripts/        # Scripts de download
```

---

## ✅ Regras Globais

1. **Sempre commitar** após modificações (ver [git-workflow](git/README.md))
2. **Documentar scores** no TODO.md e insights/ (ver [kaggle-insights](kaggle/insights.md))
3. **Usar `local_files_only=True`** em notebooks Kaggle (ver [huggingface-local](models/huggingface.md))
4. **Usar Focal Loss** para transformers (ver [kaggle-finetuning](kaggle/finetuning.md))
5. **Threshold Tuning** obrigatório após treino (ver [kaggle-finetuning](kaggle/finetuning.md))

---

## 🏆 Scores de Referência

| Modelo | Score | Técnicas |
|--------|-------|----------|
| TF-IDF + LinearSVC | 0.77885 | Baseline |
| **BERTimbau v4** | **0.82073** | Focal Loss + Threshold Tuning |
