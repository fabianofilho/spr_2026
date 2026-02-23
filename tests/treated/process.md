# Processamento de Dados - Modelos Tratados

## Objetivo
Aplicar técnicas de pré-processamento de texto para melhorar a performance dos melhores modelos baseline identificados.

---

## 💡 Insight [22/02/2026] - Eduardo Farina

> "Acho que tem que tratar os dados muito bem antes. Os laudos são muito parecidos. Tem o glossário do BI-RADS."

### Implicações
1. **Laudos muito parecidos** → O modelo precisa capturar **diferenças sutis** entre categorias
   - Pequenas variações de vocabulário podem ser decisivas
   - TF-IDF funciona bem (0.77885) porque captura frequências de termos específicos
   
2. **Glossário BI-RADS** → Usar vocabulário **padronizado** do BI-RADS como guia
   - Termos-chave por categoria já existem e são bem definidos
   - Possibilidade de criar features baseadas no glossário oficial
   - Normalizar sinônimos para termos canônicos do BI-RADS

3. **Tratamento cuidadoso** → Foco em:
   - Não perder informação discriminativa (evitar stemming agressivo)
   - Preservar termos técnicos exatos do BI-RADS
   - Extrair e destacar keywords que diferenciam categorias

### Ações Prioritárias
- [ ] Obter glossário oficial BI-RADS
- [ ] Mapear termos-chave por categoria (0-6)
- [ ] Criar dicionário de normalização (sinônimos → termo canônico)
- [ ] Feature engineering baseada no glossário

---

## ⚠️ Análise EDA - Observações Críticas

### Desbalanceamento de Classes
Os dados são **bem desbalanceados**:
- **Classes 5 e 6**: Poucos dados e são **críticas** (casos graves)
- **Classe 2**: Pode não estar no teste (possível "pegadinha" da competição)

### Estratégias para Desbalanceamento
- [ ] **Class Weights**: Penalizar mais erros em classes minoritárias
- [ ] **Oversampling**: SMOTE ou RandomOverSampler para classes 5 e 6
- [ ] **Undersampling**: Reduzir classe majoritária
- [ ] **Focal Loss**: Para modelos de deep learning
- [ ] **Threshold Tuning**: Ajustar thresholds de decisão por classe
- [ ] **Stratified K-Fold**: Garantir proporção das classes no CV

### Hipótese da Classe 2
> ⚠️ Suspeita: Classe 2 pode não existir no conjunto de teste
- Testar modelo sem classe 2 no treino
- Verificar se redistribuir classe 2 para outras melhora score
- Analisar se classe 2 tem padrão textual diferente

---

## Pipeline de Processamento

### 1. Análise Exploratória (EDA)
- [x] Distribuição das classes (target) → **DESBALANCEADO**
- [ ] Comprimento dos textos (min, max, média)
- [ ] Frequência de palavras
- [ ] Identificar padrões nos laudos
- [ ] Verificar textos duplicados
- [ ] Analisar vocabulário específico de mamografia

### 2. Limpeza de Texto
- [ ] Remover caracteres especiais
- [ ] Normalizar acentuação
- [ ] Converter para minúsculas
- [ ] Remover números (ou normalizar: `12mm` → `NUM`)
- [ ] Remover espaços extras
- [ ] Corrigir encoding (UTF-8)

### 3. Stop Words
- [ ] Remover stop words em português (NLTK/spaCy)
- [ ] **NÃO remover** termos médicos importantes
- [ ] Criar lista customizada de stop words
- [ ] Testar com/sem remoção (comparar performance)

### 4. Normalização de Vocabulário
- [ ] **Lematização** (spaCy pt_core_news_lg)
  - Ex: `calcificações` → `calcificação`
  - Ex: `assimétricos` → `assimétrico`
- [ ] **Stemming** (alternativa: RSLP Stemmer)
  - Mais agressivo, pode perder significado
- [ ] Expandir abreviações médicas
  - Ex: `BI-RADS` (manter)
  - Ex: `cm` → `centímetros` (ou manter)

### 5. Filtros e Regras
- [ ] Extrair termos BI-RADS explícitos (regex)
- [ ] Identificar palavras-chave de cada categoria
- [ ] Remover seções irrelevantes (cabeçalho, assinatura)
- [ ] Extrair features estruturadas:
  - Menção de "massa"
  - Menção de "calcificação"
  - Menção de "assimetria"
  - Menção de "achado benigno"

### 6. Feature Engineering
- [ ] N-grams (bigramas, trigramas)
- [ ] TF-IDF com vocabulário filtrado
- [ ] Embeddings com texto limpo
- [ ] Features numéricas extraídas do texto

---

## Experimentos Planejados

| Experimento | Processamento | Modelo Base | Status |
|-------------|---------------|-------------|--------|
| treated_v1 | Stop words + lowercase | LinearSVC | ⏳ |
| treated_v2 | Stop words + lematização | LinearSVC | ⏳ |
| treated_v3 | Lematização + filtros BI-RADS | LinearSVC | ⏳ |
| treated_v4 | Completo (todos os passos) | Top 3 modelos | ⏳ |

---

## Dependências Offline

Para funcionar no Kaggle (Internet OFF), adicionar datasets:

| Biblioteca | Dataset Kaggle |
|------------|----------------|
| spaCy pt_core_news_lg | `rtatman/spacy-pretrained-models` |
| NLTK stopwords | `nltkdata/stopwords` |
| NLTK punkt | `nltkdata/punkt` |

---

## Código Base

```python
import re
import spacy

# Carregar modelo spaCy (offline)
nlp = spacy.load('/kaggle/input/spacy-models/pt_core_news_lg')

# Stop words customizadas (manter termos médicos)
STOP_WORDS_REMOVE = {'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'em', 'para', 'com'}
STOP_WORDS_KEEP = {'não', 'sem', 'normal', 'benigno', 'maligno', 'suspeito'}

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remover caracteres especiais (manter hífen para BI-RADS)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lematização
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc 
              if token.text not in STOP_WORDS_REMOVE or token.text in STOP_WORDS_KEEP]
    
    return ' '.join(tokens)

# Extrair menções BI-RADS
def extract_birads(text):
    pattern = r'bi-?rads?\s*:?\s*(\d)'
    match = re.search(pattern, text.lower())
    return int(match.group(1)) if match else None
```

---

## Notas

- ⚠️ Sempre comparar com baseline sem tratamento
- ⚠️ Cuidado para não remover informação importante
- 💡 Textos médicos têm vocabulário específico
- 💡 BI-RADS pode estar explícito ou implícito no texto
- 💡 Fazer validação cruzada para evitar overfitting
