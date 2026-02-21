# TF-IDF - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Observação |
|------|--------|-------|------------|
| 1 | LinearSVC | **0.77885** | 🏆 Melhor geral |
| 2 | SGDClassifier | 0.75019 | -0.029 |
| 3 | Logistic Regression | 0.72935 | -0.050 |
| 4 | LightGBM + SVD | 0.70273 | -0.076 |
| 5 | XGBoost | 0.69482 | -0.084 |
| 6 | SVD + XGBoost | 0.66897 | -0.110 |
| 7 | CatBoost | 0.48202 | -0.297 |
| 8 | TabPFN v0.1.9 | 0.39074 | -0.388 |

---

## Por que TF-IDF funciona tão bem?

### 1. Textos Curtos e Técnicos

O dataset contém laudos médicos com características:
- **Textos curtos:** poucas sentenças
- **Vocabulário técnico restrito:** termos BI-RADS, anatomia, patologia
- **Alta discriminatividade léxica:** presença de "benigno" vs "maligno" é decisiva

TF-IDF captura exatamente isso: **quais palavras aparecem e com que frequência**.

### 2. Esparsidade é Feature, não Bug

Em textos curtos técnicos, a matriz esparsa do TF-IDF é vantajosa:
- Cada termo médico vira uma "feature binária" implícita
- Modelos lineares (SVM, LogReg) exploram isso diretamente
- Não há "ruído semântico" de embeddings densos

### 3. Classificadores Lineares Dominam

| Tipo | Melhor Score | Observação |
|------|--------------|------------|
| Linear (SVC, SGD, LogReg) | 0.77885 | ✅ Domina |
| Ensemble (LGBM, XGB) | 0.70273 | ❌ Perde 7.6 pontos |
| Boosting + SVD | 0.66897 | ❌ SVD piora |

**Por quê?** 
- TF-IDF já é linearmente separável neste problema
- Boosting adiciona complexidade desnecessária
- SVD (redução de dimensionalidade) descarta informação útil

---

## Análise por Modelo

### LinearSVC (0.77885) 🏆

**Por que é o melhor:**
- SVM com kernel linear é otimizado para dados esparsos de alta dimensão
- Margem máxima funciona bem quando classes são linearmente separáveis
- Regularização L2 implícita previne overfitting

**Configuração provável ideal:**
```python
LinearSVC(C=1.0, max_iter=10000, class_weight='balanced')
```

### SGDClassifier (0.75019)

**Gap de 2.9 pontos:**
- SGD é aproximação estocástica do SVM
- Menos estável, mais sensível a learning rate
- Útil para datasets maiores, mas aqui perde para LinearSVC

### Logistic Regression (0.72935)

**Gap de 5 pontos:**
- Boa baseline, interpretável
- Menos robusto que SVM para margens apertadas

### Modelos de Boosting (LGBM, XGBoost) 

**Por que perdem 7-8 pontos:**
1. Boosting é ótimo para features densas, não esparsas
2. Árvores cortam features TF-IDF de forma subótima
3. Overfitting mais provável em alta dimensionalidade

### CatBoost (0.48202) e TabPFN (0.39074)

**Falhas severas:**
- CatBoost: não otimizado para texto esparso
- TabPFN: não roda bem offline, pode ter tido problemas de execução

---

## Próximos Passos

### 1. Otimização do LinearSVC
- Grid search em C: [0.1, 1, 10]
- Testar `class_weight` ajustado
- Experimentar `loss='squared_hinge'`

### 2. Ensemble com TF-IDF
- Voting: LinearSVC + SGD + LogReg
- Stacking com meta-learner simples

### 3. TF-IDF + Transformers
- Usar TF-IDF como feature adicional em fine-tuning

### 4. N-grams e Preprocessing
- Testar bi-grams e tri-grams
- Remover stopwords médicas específicas
- Lematização com spaCy

---

## Conclusão

TF-IDF + LinearSVC é o **baseline imbatível** para este problema porque:

1. **Dados são linearmente separáveis** em espaço TF-IDF
2. **Termos médicos específicos** são altamente discriminativos
3. **Complexidade adicional** (boosting, embeddings) prejudica

**Veredicto:** Focar em otimizações marginais do LinearSVC e testar Transformers como alternativa semântica.

---

*Atualizado em: 20/02/2026*
