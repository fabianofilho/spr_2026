# BERTimbau PTBR - Kaggle Model

## Informações do Modelo

| Campo | Valor |
|-------|-------|
| **Model Name** | bertimbau ptbr |
| **Kaggle URL** | `kaggle.com/models/fabianofilho/bertimbau-ptbr` |
| **Framework** | PyTorch |
| **Variation** | default |
| **License** | MIT |
| **Visibility** | Private |

## Modelo Original (HuggingFace)

| Campo | Valor |
|-------|-------|
| **Nome** | `neuralmind/bert-base-portuguese-cased` |
| **Tipo** | BERT Base |
| **Parâmetros** | ~110M |
| **Hidden Size** | 768 |
| **Layers** | 12 |
| **Vocab Size** | 29,794 |
| **Max Length** | 512 tokens |

## Arquivos Incluídos

```
bert-base-portuguese-cased/
├── config.json              # Configuração do modelo
├── pytorch_model.bin        # Pesos do modelo (~438MB)
├── tokenizer_config.json    # Configuração do tokenizer
├── vocab.txt                # Vocabulário
├── special_tokens_map.json  # Tokens especiais
└── tokenizer.json           # Tokenizer (opcional)
```

## Como Usar no Kaggle

### 1. Adicionar como Input

No notebook de submissão:
- **Add Input** → **Models** → buscar `bertimbau ptbr`
- Ou: **Add Input** → **Your Work** → selecionar o modelo

### 2. Path no Kaggle

O modelo estará disponível em:
```
/kaggle/input/models/fabianofilho/bertimbau-ptbr/pytorch/default/1/bert-base-portuguese-cased/
```

### 3. Código de Carregamento

```python
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def find_model_path():
    """Encontra automaticamente o path do modelo."""
    base = '/kaggle/input'
    
    def search_dir(directory, depth=0, max_depth=6):
        if depth > max_depth:
            return None
        try:
            for item in os.listdir(directory):
                path = os.path.join(directory, item)
                if os.path.isdir(path):
                    if os.path.exists(os.path.join(path, 'config.json')):
                        return path
                    result = search_dir(path, depth + 1, max_depth)
                    if result:
                        return result
        except:
            pass
        return None
    
    return search_dir(base)

MODEL_PATH = find_model_path()
print(f"Modelo encontrado: {MODEL_PATH}")

# Carregar (sempre usar local_files_only=True para offline)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=7,
    local_files_only=True
)
```

## Sobre o BERTimbau

BERTimbau é um modelo BERT pré-treinado especificamente para português brasileiro. Foi desenvolvido pela [NeuralMind](https://github.com/neuralmind-ai/portuguese-bert) e treinado em um corpus massivo de textos em português.

**Vantagens para nossa competição:**
- Pré-treinado em português brasileiro (não multilingual genérico)
- Entende nuances do idioma português
- Melhor tokenização de palavras em português
- Base sólida para fine-tuning em domínio médico

## Referências

- [Paper: BERTimbau](https://link.springer.com/chapter/10.1007/978-3-030-61377-8_28)
- [HuggingFace Model](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [GitHub NeuralMind](https://github.com/neuralmind-ai/portuguese-bert)
