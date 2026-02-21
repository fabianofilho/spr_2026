# Word2Vec - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Gap vs TF-IDF Best |
|------|--------|-------|-------------------|
| 1 | Word2Vec + XGBoost | 0.66385 | -0.115 |
| 2 | Word2Vec + Max Pooling | 0.58009 | -0.199 |
| 3 | Word2Vec + SVM | 0.57456 | -0.204 |
| 4 | FastText + LogReg | 0.56783 | -0.211 |
| 5 | Word2Vec NILC | 0.56727 | -0.212 |
| 6 | Word2Vec + LightGBM | 0.56096 | -0.218 |
| 7 | Word2Vec + TF-IDF Weighted | 0.52215 | -0.257 |

**Melhor Word2Vec: 0.66385** vs **Melhor TF-IDF: 0.77885** = **Gap de 11.5 pontos**

---

## Diagnóstico: Por que Word2Vec falha neste dataset?

### 1. Perda de Especificidade Léxica

**Problema:** Word2Vec representa palavras em espaço contínuo, perdendo a distinção binária "tem/não tem" que TF-IDF captura.

```
Exemplo:
- Texto: "calcificações grosseiras benignas"
- TF-IDF: vetor esparso com pesos altos em "calcificações", "grosseiras", "benignas"
- Word2Vec: média dos embeddings → vetor denso genérico
```

Em textos médicos curtos, a **presença de termos específicos** (BI-RADS 2, calcificação, nódulo) é mais discriminativa que a semântica.

### 2. Diluição por Média

**Problema:** Ao fazer média dos embeddings de todas as palavras, informação discriminativa é diluída.

```python
# Típico pipeline Word2Vec
doc_embedding = np.mean([model[word] for word in words], axis=0)
```

Se um texto tem 50 palavras e apenas 3 são discriminativas, a média dilui essas 3 palavras em 97% de "ruído".

**Tentativas:**
- Max Pooling (0.58009): Melhorou marginalmente
- TF-IDF Weighted (0.52215): Piorou! Provavelmente porque pesos TF-IDF já capturam importância

### 3. Vocabulário Médico Sub-Representado

**Problema:** Modelos Word2Vec pré-treinados (Skip-gram ou CBOW) são treinados em textos genéricos.

- Palavras como "BI-RADS", "microcalcificações", "espiculado" podem ter embeddings pobres ou inexistentes
- Embeddings OOV (out-of-vocabulary) prejudicam severamente

**NILC (0.56727):** Mesmo com modelo pré-treinado em português, performance foi similar → vocabulário médico ainda insuficiente.

### 4. Textos Muito Curtos

**Problema:** Word2Vec funciona melhor com contexto longo. Textos curtos não fornecem contexto suficiente para média significativa.

---

## O que funcionou (relativamente)

### XGBoost (0.66385) - Melhor Word2Vec

XGBoost conseguiu extrair padrões não-lineares dos embeddings que classificadores lineares (SVM, LogReg) não capturam. Ainda assim, 11.5 pontos abaixo do TF-IDF + LinearSVC.

### Max Pooling (0.58009)

Capturar o máximo em vez da média preserva mais informação discriminativa, mas ainda insuficiente.

---

## Recomendações Futuras

### 1. Não investir mais em Word2Vec puro
O gap é muito grande para justificar otimizações marginais.

### 2. Testar Doc2Vec / Paragraph Vectors
Aprende embedding diretamente do documento, sem média.

### 3. Word2Vec como feature adicional para ensemble
Combinado com TF-IDF pode agregar valor marginal.

### 4. FastText subword
FastText com subword pode capturar melhor termos médicos OOV.

---

## Conclusão

Word2Vec não é adequado para este problema porque:

1. **Textos curtos e técnicos** → presença léxica > semântica
2. **Vocabulário especializado** → embeddings genéricos insuficientes
3. **Média dilui informação** → perda de especificidade

**Veredicto:** Abandonar Word2Vec como abordagem primária. Foco em TF-IDF + modelos lineares ou Transformers.

---

*Atualizado em: 20/02/2026*
