# Models - Notebooks de Download

Esta pasta contém notebooks para baixar modelos do HuggingFace e salvá-los como output no Kaggle para uso offline.

## Como Usar

1. Faça upload do notebook de download para o Kaggle
2. **Internet ON**, Accelerator **None**
3. Run All
4. **Save Version** → **Save & Run All (Commit)**
5. Aguarde finalizar (~5-15 min dependendo do modelo)
6. O output fica em: **Add Input → Your Work → nome do notebook**

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
