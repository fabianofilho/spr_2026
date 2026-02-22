# Models - Notebooks de Download

Esta pasta contém notebooks para baixar modelos do HuggingFace e salvá-los como output no Kaggle para uso offline.

---

## 📋 Passo a Passo Completo

### Etapa 1: Upload do Notebook de Download

1. Acesse o Kaggle: https://www.kaggle.com/
2. Clique em **Create** → **New Notebook**
3. Clique em **File** → **Import Notebook**
4. Selecione o notebook de download desejado (ex: `download_bertimbau_large.ipynb`)
5. Dê um nome descritivo (ex: `download-bertimbau-large`)

### Etapa 2: Configurar o Notebook

1. No painel direito, clique em **Settings**
2. Configure:
   - **Internet**: `ON` ✅ (necessário para download)
   - **Accelerator**: `None` (não precisa de GPU)
   - **Environment**: `Always use latest environment`

### Etapa 3: Executar o Download

1. Clique em **Run All** (ou Ctrl+Shift+Enter)
2. Aguarde todas as células executarem (~5-15 min)
3. Verifique se não houve erros
4. A última célula deve mostrar "Teste OK!"

### Etapa 4: Salvar como Output

1. Clique em **Save Version** (canto superior direito)
2. Selecione **Save & Run All (Commit)**
3. Adicione uma descrição (ex: "BERTimbau Large model download")
4. Clique em **Save**
5. **IMPORTANTE**: Aguarde o notebook finalizar (pode levar 10-20 min)

### Etapa 5: Usar no Notebook de Submit

1. Abra o notebook de submit (ex: `submit_bertimbau_large_focal.ipynb`)
2. No painel direito, clique em **Add Input**
3. Vá em **Your Work** → **Notebooks**
4. Selecione o notebook de download que você salvou
5. O modelo estará disponível em `/kaggle/input/nome-do-notebook/`

### Etapa 6: Submeter

1. Configure o notebook de submit:
   - **Internet**: `OFF` ✅
   - **Accelerator**: `GPU T4 x2`
2. Clique em **Run All**
3. Após finalizar, clique em **Save Version** → **Save & Run All (Commit)**

---

## 🔄 Fluxo Visual

```
┌─────────────────────────────────────────────────────────────────┐
│                     KAGGLE WORKFLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. DOWNLOAD (Internet ON)                                      │
│   ┌──────────────────────────────┐                               │
│   │  download_bertimbau_large    │  ──► Run All ──► Save Version │
│   └──────────────────────────────┘                               │
│                    │                                             │
│                    ▼                                             │
│   ┌──────────────────────────────┐                               │
│   │  Output: bertimbau-large/    │  (modelo salvo)               │
│   │  ├── config.json             │                               │
│   │  ├── model.safetensors       │                               │
│   │  └── tokenizer.json          │                               │
│   └──────────────────────────────┘                               │
│                    │                                             │
│                    ▼                                             │
│   2. SUBMIT (Internet OFF)                                       │
│   ┌──────────────────────────────┐                               │
│   │  submit_bertimbau_focal      │                               │
│   │  + Add Input: download-...   │  ──► Run All ──► Submit       │
│   └──────────────────────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Dicas Importantes

1. **Não feche a aba** enquanto o Save Version estiver rodando
2. **Verifique o tamanho do output** - deve ser próximo ao esperado (ver tabela abaixo)
3. **Se der erro de timeout**, tente novamente - às vezes o HuggingFace está lento
4. **Mantenha os nomes simples** - evite caracteres especiais no nome do notebook

---

## Notebooks Disponíveis

| Notebook | Modelo | Tamanho | Notebooks que usam |
|----------|--------|---------|-------------------|
| `download_bertimbau.ipynb` | neuralmind/bert-base-portuguese-cased | ~440 MB | `submit_bertimbau.ipynb` |
| `download_bertimbau_large.ipynb` | neuralmind/bert-large-portuguese-cased | ~1.3 GB | `submit_bertimbau_large_focal.ipynb`, `submit_bertimbau_lora_offline.ipynb` |
| `download_biobertpt.ipynb` | pucpr/biobertpt-all | ~440 MB | `submit_biobertpt.ipynb` |
| `download_mdeberta.ipynb` | microsoft/mdeberta-v3-base | ~560 MB | `submit_deberta.ipynb`, `submit_mdeberta_classweights.ipynb` |
| `download_distilbert.ipynb` | distilbert-base-multilingual-cased | ~540 MB | `submit_distilbert.ipynb` |
| `download_xlmroberta.ipynb` | xlm-roberta-base | ~1.1 GB | `submit_xlmroberta_meanpool.ipynb` |
| `download_modernbert.ipynb` | answerdotai/ModernBERT-base | ~580 MB | `submit_modernbert.ipynb` |

## Modelos Disponíveis no Kaggle Models (não precisam de download)

Estes modelos já estão disponíveis diretamente no Kaggle:

| Modelo | Como adicionar |
|--------|---------------|
| BERT Multilingual | Add Input → Models → `bert-base-multilingual-cased` |
| ModernBERT | Add Input → Models → `modernbert` (answer-ai, variation: base) |
| BERTimbau | Add Input → Models → `bertimbau-ptbr-complete` (fabianofilho) |
| mDeBERTa | Add Input → Datasets → `mdeberta_v3_base` (Jonathan Chan) |

---

## 📌 Mapeamento: Submit → Download

| Notebook de Submit | Notebook de Download | Alternativa Kaggle |
|-------------------|---------------------|-------------------|
| `submit_bertimbau.ipynb` | `download_bertimbau.ipynb` | Models: `bertimbau-ptbr-complete` |
| `submit_bertimbau_large_focal.ipynb` | `download_bertimbau_large.ipynb` | - |
| `submit_bertimbau_lora_offline.ipynb` | `download_bertimbau.ipynb` ou `download_bertimbau_large.ipynb` | Models: `bertimbau-ptbr-complete` |
| `submit_biobertpt.ipynb` | `download_biobertpt.ipynb` | - |
| `submit_deberta.ipynb` | `download_mdeberta.ipynb` | Datasets: `mdeberta_v3_base` |
| `submit_mdeberta_classweights.ipynb` | `download_mdeberta.ipynb` | Datasets: `mdeberta_v3_base` |
| `submit_distilbert.ipynb` | `download_distilbert.ipynb` | Models: `distilbert` |
| `submit_xlmroberta_meanpool.ipynb` | `download_xlmroberta.ipynb` | Models: `xlm-roberta` |
| `submit_modernbert.ipynb` | `download_modernbert.ipynb` | Models: `modernbert` (answer-ai) |
| `submit_bert_multilingual.ipynb` | - | Models: `bert-base-multilingual-cased` |

---

## 🔧 Troubleshooting

### Erro: "Model not found"
- Verifique se o notebook de download foi salvo com sucesso
- Confirme que adicionou o output como Input no notebook de submit
- Rode `show_tree('/kaggle/input')` para ver a estrutura de pastas

### Erro: "local_files_only=True but file not found"
- O modelo não foi adicionado corretamente como Input
- Verifique se está usando `Add Input → Your Work → Notebooks`

### Erro de timeout no download
- Tente novamente em horário diferente (HuggingFace pode estar lento)
- Verifique se a internet está ON no notebook

### Output muito pequeno (< 100 MB)
- O download pode ter falhado silenciosamente
- Verifique os logs do notebook de download

---

## Estrutura de Pastas Baixadas

```
/kaggle/input/
├── download-bertimbau/
│   └── bertimbau-base/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
├── download-bertimbau-large/
│   └── bertimbau-large/
│       └── ...
└── ...
```

## Uso nos Notebooks de Submit

```python
# Busca automática do modelo (já implementado nos notebooks)
def find_model_path():
    base = '/kaggle/input'
    def search(d, depth=0):
        if depth > 10: return None
        for item in os.listdir(d):
            p = os.path.join(d, item)
            if os.path.isdir(p):
                if os.path.exists(os.path.join(p, 'config.json')): 
                    return p
                r = search(p, depth+1)
                if r: return r
        return None
    return search(base)

MODEL_PATH = find_model_path()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
```
