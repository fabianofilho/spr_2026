# Skill: Kaggle Models (Offline)

## 🎯 Quando Ativar

- Criar notebook com transformers no Kaggle
- Usar modelos HuggingFace offline
- Resolver erros de path de modelo

---

## 📋 Tipos de Input no Kaggle

| Tipo | Estrutura em `/kaggle/input/` | Exemplo |
|------|------------------------------|---------|
| **Kaggle Models** | `/kaggle/input/models/{autor}/{modelo}/...` | ModernBERT, Llama, Gemma |
| **Kaggle Datasets** | `/kaggle/input/{dataset-slug}/...` | Upload manual de modelos |

---

## 🔧 Código: Busca Automática de Modelo

**SEMPRE usar busca recursiva** para encontrar `config.json`:

```python
import os

def find_model_path():
    """Encontra automaticamente o path do modelo (busca recursiva até 6 níveis)."""
    base = '/kaggle/input'
    
    def has_config(path):
        return os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.json'))
    
    def search_dir(directory, depth=0, max_depth=6):
        if depth > max_depth:
            return None
        try:
            for item in os.listdir(directory):
                path = os.path.join(directory, item)
                if os.path.isdir(path):
                    if has_config(path):
                        return path
                    result = search_dir(path, depth + 1, max_depth)
                    if result:
                        return result
        except:
            pass
        return None
    
    return search_dir(base)

MODEL_PATH = find_model_path()
print(f"Modelo encontrado em: {MODEL_PATH}")
```

---

## ✅ Modelos Disponíveis no Kaggle Models

Modelos que funcionam via **Add Input → Models**:

| Modelo | Buscar | Autor | Variation | Path |
|--------|--------|-------|-----------|------|
| BERT Multilingual | `bert-base-multilingual-cased` | `muhammadukasha09` | `default` (V1) | `/kaggle/input/models/muhammadukasha09/bert-base-multilingual-cased/transformers/default/1/bert-base-multilingual-cased` |
| ModernBERT | `modernbert` | `answer-ai` | `base` (V2) | `/kaggle/input/models/answer-ai/modernbert/pytorch/base/2` |
| **BERTimbau Large** | `bertimbau-ptbr-complete` | `fabianofilho` | `pytorch/default/1` | `/kaggle/input/models/fabianofilho/bertimbau-ptbr-complete/pytorch/default/1` |

---

## ❌ Modelos NÃO Disponíveis

Requerem **upload manual** via Dataset:

| Modelo | HuggingFace Hub |
|--------|-----------------|
| BioBERTpt | `pucpr/biobertpt-all` |

→ Ver [upload.md](../models/upload.md) para instruções de upload manual.

---

## 🔧 Como Adicionar Modelo no Kaggle

### Via Kaggle Models (preferido):
1. No notebook Kaggle → **Add Input**
2. Aba **Models**
3. Buscar nome do modelo
4. Selecionar autor e versão
5. Clicar **Add**

### Via Dataset (upload manual):
1. Baixar modelo localmente
2. Upload como Dataset
3. Ver [upload.md](../models/upload.md)

---

## ⚠️ Erros Comuns

### Erro: `OSError: {path} is not a valid git identifier`
**Causa:** Tentando baixar da internet em modo offline
**Solução:** Usar `local_files_only=True`

### Erro: `FileNotFoundError: config.json`
**Causa:** Path incorreto do modelo
**Solução:** Usar função `find_model_path()` acima

### Erro: `No model found`
**Causa:** Modelo não adicionado como input
**Solução:** Add Input → Models/Datasets
