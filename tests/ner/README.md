# NER (Named Entity Recognition) - Estratégia de Tratamento

## Objetivo

Extrair entidades médicas específicas dos laudos de mamografia usando NER, gerar embeddings separados por categoria e usar como features para os modelos baseline.

---

## 💡 Motivação

### Problema com Embeddings Gerais

```
"ausência de microcalcificações" ≈ "microcalcificações pleomórficas"
```

No espaço vetorial geral, essas frases são **similares** (ambas contêm "microcalcificações").
Porém, semanticamente são **opostas** para classificação BI-RADS:
- "ausência de microcalcificações" → BI-RADS 1 ou 2 (benigno)
- "microcalcificações pleomórficas" → BI-RADS 4 ou 5 (suspeito)

### Solução: NER + Embeddings por Categoria

1. Extrair entidades específicas (nódulo, microcalcificação, etc.)
2. Gerar embeddings **separados** para cada categoria
3. Concatenar embeddings como features
4. Modelo final classifica baseado em contexto separado

---

## 🏷️ Categorias de Entidades (NER)

| Categoria | Exemplos | Relevância |
|-----------|----------|------------|
| **Nódulo** | "nódulo espiculado", "nódulo oval", "massa irregular" | Alta - indica suspeita |
| **Microcalcificação** | "calcificações pleomórficas", "microcalcificações agrupadas" | Alta - padrão discriminativo |
| **Arquitetura** | "distorção arquitetural", "assimetria focal" | Alta - alteração estrutural |
| **Densidade** | "mamas densas", "parênquima fibroglandular" | Média - contexto |
| **Negação** | "sem alterações", "ausência de", "não foram observados" | Crítica - inverte sentido |
| **BI-RADS** | "BIRADS 4", "categoria 2", "BI-RADS 0" | Alta - classificação explícita |

---

## 🔄 Pipeline Proposto

```
┌─────────────────────────────────────────────────────────────────┐
│                         LAUDO DE MAMA                           │
│   "Nódulo oval com margens circunscritas. Sem microcalcificações│
│    suspeitas. Parênquima mamário denso. BI-RADS 2."             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      1. NER EXTRACTION                          │
│   ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│   │ Nódulo:     │  │ Microcalcif.:   │  │ BI-RADS:         │   │
│   │ "oval com   │  │ "Sem micro-     │  │ "BI-RADS 2"      │   │
│   │  margens    │  │  calcificações  │  │                  │   │
│   │  circuns-   │  │  suspeitas"     │  │                  │   │
│   │  critas"    │  │                 │  │                  │   │
│   └─────────────┘  └─────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. EMBEDDINGS POR CATEGORIA                   │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│   │ Emb_nódulo │  │ Emb_micro  │  │ Emb_birads │  (768D cada)  │
│   │   [BERTimbau ou Custom]    │  │            │               │
│   └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3. CONCATENAR + REDUZIR (PCA)                   │
│   [Emb_nódulo | Emb_micro | Emb_arch | Emb_neg | Emb_birads]   │
│                    (768*5 = 3840D) → PCA → 512D                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. CLASSIFICADOR FINAL                        │
│   ┌────────────────────────────────────────────────────────┐   │
│   │  XGBoost (ordinal, não categórico)                      │   │
│   │  + Regras determinísticas para BI-RADS 0 e 6            │   │
│   └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        BI-RADS 0-6
```

---

## ⚙️ Implementação

### Fase 1: NER com Regex + spaCy

```python
# Regex patterns para cada categoria
patterns = {
    'nodulo': r'(nódulo|massa|lesão)\s+[\w\s]*(oval|irregular|espiculado|circunscrit|lobulad)',
    'microcalcificacao': r'(micro)?calcifica(ção|ções)[\w\s]*(pleomórfic|agrupadas|amorf|suspeitas)?',
    'arquitetura': r'(distorção|assimetria|alteração)\s+[\w\s]*(arquitetural|focal|global)',
    'negacao': r'(sem|ausência|não|nenhum|inexistem)[\w\s]*(alterações|sinais|evidência)',
    'birads': r'bi-?rads?\s*:?\s*(\d|zero|um|dois|três|quatro|cinco|seis)',
}
```

### Fase 2: spaCy NER Custom (opcional)

Treinar modelo spaCy NER com dataset anotado manualmente.

### Fase 3: Embeddings por Categoria

```python
from transformers import AutoModel, AutoTokenizer

def get_entity_embeddings(text, entities, model, tokenizer):
    embeddings = {}
    for category, entity_text in entities.items():
        if entity_text:
            inputs = tokenizer(entity_text, return_tensors='pt')
            outputs = model(**inputs)
            embeddings[category] = outputs.last_hidden_state.mean(dim=1)
        else:
            embeddings[category] = torch.zeros(768)  # Placeholder
    return embeddings
```

### Fase 4: PCA + Classificador

```python
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Redução de dimensionalidade
pca = PCA(n_components=512)
X_reduced = pca.fit_transform(X_concatenated)

# XGBoost como ordinal (BI-RADS tem ordem: 0 < 1 < 2 < 3 < 4 < 5 < 6)
clf = XGBClassifier(
    objective='multi:softmax',  # Mas tratar como ordinal
    num_class=7,
    reg_alpha=0.1,  # Regularização para evitar overfitting
    reg_lambda=1.0,
)
```

---

## 📌 Regras Determinísticas

### BI-RADS 0 (Inconclusivo)
```python
if 'avaliação adicional' in text.lower() or 'bi-rads 0' in text.lower():
    return 0
```

### BI-RADS 6 (Malignidade Conhecida)
```python
if 'malignidade conhecida' in text.lower() or 'bi-rads 6' in text.lower():
    return 6
if 'biópsia prévia positiva' in text.lower():
    return 6
```

---

## 📈 Hipóteses de Melhoria

| Técnica | Impacto Esperado | Risco |
|---------|------------------|-------|
| NER + embeddings categóricos | +3-5% F1 | Complexidade de implementação |
| PCA (512D) | Reduz overfitting | Pode perder informação |
| XGBoost ordinal | Melhor para BI-RADS | Precisa validar |
| Regras BI-RADS 0/6 | +2% nas classes raras | Baixo |

---

## 🧪 Experimentos Planejados

| Notebook | Descrição | Status |
|----------|-----------|--------|
| `submit_ner_regex.ipynb` | NER com regex + LinearSVC | ⏳ |
| `submit_ner_bertimbau.ipynb` | NER + BERTimbau embeddings | ⏳ |
| `submit_ner_xgboost_ordinal.ipynb` | NER + XGBoost ordinal | ⏳ |
| `submit_ner_rules.ipynb` | NER + regras determinísticas | ⏳ |

---

## 📚 Referências

- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
- [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [BI-RADS Atlas](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Bi-Rads)

---

*Criado em: 24/02/2026*
