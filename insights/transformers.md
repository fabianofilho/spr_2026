# Transformers - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 🏆 | **BERTimbau 5-Fold + MAX_LEN=512** | **0.84027** | ✅ MELHOR |
| 🥈 | BERTimbau Threshold v3 | 0.81301 | ✅ 2º MELHOR |
| ❌ | **BERTimbau Large + AWP + Optuna (2026-04-17)** | **0.77969** | ⚠️ Regressao -6pp |
| 3 | BERTimbau Raw Data v9 | 0.81213 | ✅ Submetido |
| 4 | BERTimbau v5 + Label Smoothing | 0.81100 | ✅ Submetido |
| 5 | BERTimbau 5-Fold v11 | 0.80950 | ✅ Submetido |
| 6 | BERTimbau MAX_LEN=512 v2 | 0.80509 | ✅ Submetido |
| 7 | BERTimbau v4 + Threshold Tuning | 0.82073 | ✅ Submetido |
| 8 | **BERTimbau + Focal Loss** | **0.79696** | ✅ Submetido |
| 9 | BERTimbau v5 Ensemble Threshold | 0.77385 | ✅ Submetido |
| 10 | BERTimbau v5 (Gamma Search γ=2.5) | 0.75574 | ❌ Abaixo best |
| 11 | BERTimbau v5 (LR Search) | 0.75508 | ❌ Abaixo best |
| 12 | **BioBERTpt** | **0.72480** | ✅ Submetido |
| 13 | BERTimbau v5 Seed Ensemble | 0.72135 | ❌ Abaixo best |
| 14 | BERTimbau v5 (Class) | 0.69238 | ❌ Regressão |
| 15 | **XLM-RoBERTa + Mean Pool** | **0.68767** | ✅ Submetido |
| 16 | ModernBERT base | 0.68578 | ✅ Submetido |
| ❌ | BERTimbau + LoRA (Offline) | 0.13261 | ⚠️ Falhou |
| ❌ | mDeBERTa + class weights | 0.01008 | ⚠️ BUG |

> **🏆 BERTimbau 5-Fold + MAX_LEN=512 = 0.84027** - Novo recorde! +2% acima do v4!

---

## 🔥 Análise: MAX_LEN=512 (0.84027) - NOVO MELHOR SCORE!

**MAX_LEN=512 superou todas as outras técnicas!** Score de **0.84027** supera o v4 (0.82073) em **+2.4%**.

### Por que funcionou?

1. **MAX_LEN=512:** Relatórios médicos podem ter informações importantes no final
2. **5-Fold Ensemble:** Média de 5 modelos reduz variância
3. **model-v4:** Pesos mais recentes e melhor calibrados
4. **Calibração com thresholds:** Ajuste fino por classe

### Comparação de MAX_LEN

| MAX_LEN | Score | Delta |
|---------|-------|-------|
| 512 | **0.84027** | +2.4% |
| 192 | 0.81213 | baseline |

**Insight:** Usar mais contexto ajuda significativamente!

---

## 🏆 Análise: BERTimbau + Focal Loss (0.79696)

**Primeiro transformer a superar TF-IDF!** BERTimbau com Focal Loss alcançou **0.79696**, superando o baseline TF-IDF (0.77885) em **+2.3%**.

### Por que funcionou?

1. **Focal Loss:** γ=2 foca nos exemplos difíceis, melhorando classes minoritárias
2. **BERTimbau base:** Especializado em português, vocabulário adequado
3. **Fine-tuning adequado:** Parâmetros bem calibrados para o dataset
4. **Sem overfitting:** Focal Loss tem efeito regularizador implícito

### Por que superou TF-IDF?

- **Contexto semântico:** Entende relações complexas como "sem sinais de malignidade"
- **Transfer learning:** Conhecimento prévio de português médico
- **Focal Loss:** Resolve o problema de desbalanceamento de classes

### Comparação com baseline TF-IDF

- BERTimbau + Focal: **0.79696** (+2.3% acima)
- TF-IDF baseline: 0.77885
- **Breakthrough!** Primeiro transformer a vencer.

---

## ⚠️ Análise: BERTimbau + LoRA (0.13261) - FALHA

**LoRA offline falhou completamente.** Score de 0.13261 indica predições quase aleatórias.

### Por que falhou?

1. **Offline execution:** Kaggle offline pode ter problemas com bibliotecas PEFT
2. **Rank muito baixo:** LoRA com r=4 ou r=8 pode ser insuficiente para este task
3. **Adapter não treinou:** Possível problema no salvamento/carregamento dos pesos
4. **Incompatibilidade:** Versão do PEFT pode não ser compatível com ambiente Kaggle

### Lições aprendidas

- Full fine-tuning funciona melhor que LoRA para datasets pequenos
- Testar sempre online antes de submeter offline
- LoRA economiza memória mas pode perder performance

---

## Análise: ModernBERT (0.68578)

**Segundo melhor transformer.** ModernBERT ficou **12% abaixo** do TF-IDF baseline (0.77885).

### Por que funcionou melhor?

1. **Arquitetura moderna:** RoPE embeddings, Flash Attention 2, GeGLU
2. **Contexto longo:** Suporta até 8192 tokens nativamente
3. **Treinamento eficiente:** Mais dados, melhor curriculum
4. **Tokenização robusta:** Melhor handling de tokens subword

### Comparação com baseline TF-IDF

- ModernBERT: **0.68578** (88% do baseline)
- Gap ainda significativo, mas promissor para ensembles

---

## ✅ Análise: BioBERTpt (0.72480) - 2º MELHOR TRANSFORMER

**BioBERTpt superou ModernBERT!** Score de **0.72480** demonstra a importância do domínio médico.

### Por que funcionou bem?

1. **Pré-treino médico:** Treinado em textos biomédicos PT-BR (PubMed, artigos)
2. **Vocabulário especializado:** Tokeniza "calcificação", "BIRADS" corretamente
3. **Transfer learning específico:** Conhecimento de domínio médico
4. **Português nativo:** Entende nuances da língua

### Comparação com outros modelos

| Modelo | Score | Domínio | Língua |
|--------|-------|---------|--------|
| BioBERTpt | **0.72480** | Médico | PT |
| ModernBERT | 0.68578 | Geral | EN |
| BERTimbau | 0.64319 | Geral | PT |

**Insight:** Domínio específico > Língua específica > Geral multilingual

### Por que não superou BERTimbau + Focal?

- Sem Focal Loss: Class weights normais não são tão efetivos
- BERTimbau Large + Focal teve mais capacidade

---

## ✅ Análise: XLM-RoBERTa + Mean Pooling (0.68767)

**XLM-RoBERTa competitivo** com ModernBERT. Mean pooling melhorou a representação.

### Por que funcionou?

1. **Mean Pooling:** Agrega todos os tokens, não só [CLS]
2. **Pré-treino massivo:** 100 línguas, dados abundantes
3. **RoBERTa optimizations:** Treinamento mais longo, sem NSP

### Comparação Mean vs CLS pooling

| Pooling | Score | Motivo |
|---------|-------|--------|
| Mean | **0.68767** | Captura toda a sequência |
| CLS | ~0.65 | Perde informação de tokens finais |

**Insight:** Mean pooling é superior para classificação de textos médicos.

---

## Análise: BERTimbau (0.64319)

**Resultado intermediário.** BERTimbau ficou abaixo do ModernBERT apesar de ser especializado em português.

### Por que não superou ModernBERT?

1. **Arquitetura mais antiga:** BERT original (2018) vs ModernBERT (2024)
2. **Sem otimizações modernas:** Sem Flash Attention, RoPE
3. **Pré-treinamento limitado:** Menos dados que ModernBERT
4. **Contexto máximo:** 512 tokens vs 8192 do ModernBERT

### Pontos positivos

- Modelo PT-BR ainda é **14% melhor** que BERT Multilingual (0.56095)
- Base sólida para fine-tuning adicional

---

## Análise: DistilBERT Multilingual (0.55229)

**Resultado esperado.** DistilBERT ficou próximo ao BERT Multilingual (0.56095), apenas 1.5% abaixo.

### Por que ficou similar ao BERT Multilingual?

1. **Destilação preservou conhecimento:** DistilBERT mantém ~97% da performance do BERT original
2. **Mesmo problema:** Modelo genérico multilingual, sem especialização PT-BR
3. **Vantagem:** 40% mais rápido e 60% menor que BERT full

### Quando usar DistilBERT?

- Quando tempo de inferência é crítico
- Como modelo rápido para ensembles
- Trade-off aceitável: -1.5% score vs 2x mais rápido

---

## ⚠️ BUG: mDeBERTa + Class Weights (0.01008)

**Score praticamente zero indica bug crítico!** Modelo completamente quebrado.

### Diagnóstico

Score de 0.01 em F1-macro significa:
- Modelo predizendo **sempre a mesma classe**
- Ou labels **invertidas/mapeadas incorretamente**
- Ou problema no **training loop**

### Possíveis causas

1. **Mixed precision + class_weights:** Adicionamos `.float()` nos logits, mas pode ter outro problema
2. **Label mismatch:** Verificar se labels do test set batem com train
3. **Peso de classes extremo:** Class weights muito altos podem causar gradientes instáveis
4. **Learning rate alta demais:** Com class weights, LR precisa ser menor

### Próximos passos

1. Testar `submit_deberta.ipynb` (sem class weights) para isolar o problema
2. Verificar distribuição de predições no output
3. Se necessário, remover class weights e usar Focal Loss em vez disso

---

## Análise: BERT Multilingual (0.56095)

**Resultado decepcionante.** BERT Multilingual ficou **28% abaixo** do TF-IDF baseline (0.77885).

### Por que falhou?

1. **Modelo genérico:** BERT Multilingual é treinado em 104 idiomas, diluindo conhecimento de português
2. **Sem domínio médico:** Vocabulário médico/radiológico não está bem representado
3. **Tokenização subótima:** Warning de regex do Mistral indica problemas no tokenizer
4. **Epoch insuficientes:** F1 ainda estava subindo (0.43 → 0.50 → 0.56), precisava de mais epochs
5. **Incompatibilidade LayerNorm:** Warnings de `gamma/beta` vs `weight/bias` indicam checkpoint com formato antigo

### Lições aprendidas

- Transformers genéricos **não** superam TF-IDF automaticamente
- Precisa de modelo especializado em português (BERTimbau) ou domínio médico (BioBERTpt)
- Hiperparâmetros precisam de tuning (mais epochs, learning rate schedule)

---

## Inputs Kaggle Necessários

Todos os modelos transformer precisam ser adicionados como **Input** no Kaggle:

| Modelo | Kaggle Input |
|--------|--------------|
| BERTimbau base | `neuralmind/bert-base-portuguese-cased` |
| BERTimbau large | `neuralmind/bert-large-portuguese-cased` |
| BioBERTpt | `pucpr/biobertpt-all` |
| mDeBERTa-v3 | `microsoft/mdeberta-v3-base` |
| DistilBERT | `distilbert-base-multilingual-cased` |
| XLM-RoBERTa | `xlm-roberta-large` |
| ModernBERT | `answerdotai/ModernBERT-base` |

---

## Hipóteses

### 1. Potencial Vantagens sobre TF-IDF

- **Contexto semântico:** Transformers entendem "ausência de malignidade" vs "presença de malignidade"
- **Transfer learning:** Pré-treinamento em português pode ajudar
- **Embeddings contextuais:** Mesma palavra em contextos diferentes tem representações diferentes

### 2. Riscos

- **Overfitting:** Dataset pequeno + modelo grande = alto risco
- **Tempo de inferência:** Kaggle tem limite de tempo
- **GPU requirements:** Pode falhar offline

### 3. Modelos Prioritários

1. **BERTimbau base:** Melhor modelo PT-BR, baseline obrigatório
2. **BioBERTpt:** Domínio médico, pode ter vantagem
3. **mDeBERTa:** Estado da arte em NLU

---

## ❌ Análise: BERTimbau Large + AWP + Optuna (2026-04-17) — REGRESSAO

**Score: 0.77969 (-6.1 pp vs winner 0.84027).** Ate abaixo do baseline TF-IDF (0.77885).

### Configuracao submetida

- BERTimbau Large, MAX_LEN=512, 5-fold, Focal Loss (γ=2.0, α=0.25)
- **AWP** (AWP_LR=1e-1, AWP_EPS=1e-2, start epoch 2)
- **Label Smoothing 0.05** (combinado com Focal)
- **Optuna 300 trials** (8 dimensoes: 1 temperatura + 7 thresholds)
- **Texto raw** (sem `build_input_text`)

### Diagnostico das causas (ordem de impacto)

| # | Causa | Delta estimado |
|---|-------|----------------|
| 1 | Faltou `build_input_text` (texto raw em vez de secoes extraidas) | -3 a -5 pp |
| 2 | Optuna overfitou OOF (8 dim x 300 trials, ~3000 amostras) | -1 a -2 pp |
| 3 | Tripla regularizacao: AWP + Label Smoothing + Focal | -0.5 a -1 pp |
| 4 | AWP_LR=1e-1 excessivo (literatura: 1e-4 a 1e-3) | -0.5 a -1 pp |

### Por que o OOF nao previu o score publico?

**OOF com Optuna inflado:** otimizar 8 parametros sobre ~3000 amostras cria overfit sintetico.
O F1 OOF "melhor" e artificial: thresholds escolhidos para memorizar quirks do OOF
nao generalizam para o test set. **Lição: calibracao otimizada no OOF nao e confiavel
com muitos graus de liberdade.**

### Por que AWP nao ajudou?

AWP e efetivo quando o modelo ja treina bem e voce precisa de generalizacao extra.
Aqui, com texto raw + label smoothing + Focal, o modelo ja estava subotimo no treino.
AWP em modelo subotimo = ruido adicional, nao regularizacao util.

### Licoes Aprendidas (2026-04-17)

- ❌ **Nao mudar pipeline de pre-processamento** sem validar no OOF
- ❌ **Nao empilhar regularizadores** (AWP + LS + Focal = demais)
- ❌ **Optuna com muitos trials/dims em OOF pequeno = overfit garantido**
- ✅ **Reproducao primeiro, inovacao depois** (nao pulamos essa etapa)
- ✅ **Thresholds fixos do winner** sao mais confiaveis que Optuna OOF
- ✅ **O diferencial do winner e o pre-processamento** (`build_input_text`), nao truques de treino

### Proxima rodada (estrategia corretiva)

1. **Reproduzir o winner exato** (controle) - usa `build_input_text` + thresholds fixos
2. **Pseudo-labeling** (test preds com conf >0.95 → retrain)
3. **Multi-Sample Dropout** (5x dropout no head, media)
4. **Ensemble das 3 acima**

---

## ❌ Análise: BERTimbau v5 Variações (2026-03-03)

**Nenhuma variação superou o v4 (0.82073).** Resultados decepcionantes em todas as tentativas.

| Variação | Score | Delta vs v4 |
|----------|-------|-------------|
| Gamma Search (γ=2.5) | 0.75574 | **-6.5%** |
| LR Search (1e-5/3e-5) | 0.75508 | **-6.6%** |
| Class Weights | 0.69238 | **-12.8%** |

### Por que falharam?

1. **Gamma γ=2.5:** Aumentar gamma focou demais nos exemplos difíceis, ignorando os fáceis
2. **LR diferente:** Learning rate 2e-5 do v4 já era ótima para o dataset
3. **Class Weights:** Pior resultado - Focal Loss já trata desbalanceamento, class weights é redundância prejudicial

### Insight Chave

> **Threshold Tuning é o diferencial do v4, não os hiperparâmetros de treino.**

O v4 usou thresholds otimizados por classe:
```python
THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.30, 6: 0.25}
```

Classes 5 e 6 (minoria) têm thresholds mais baixos, aumentando recall nessas classes.

### Lições Aprendidas

- ❌ Não iterar sobre hiperparâmetros de treino quando já estão otimizados
- ✅ Focar em **pós-processamento** (thresholds, calibração)
- ❌ Class weights + Focal Loss = redundância que prejudica
- ✅ γ=2.0 é o sweet spot para Focal Loss neste dataset

---

## Configurações Recomendadas

### Fine-tuning básico
```python
{
    "learning_rate": 2e-5,
    "epochs": 3-5,
    "batch_size": 16,
    "max_length": 128,  # textos são curtos
    "warmup_ratio": 0.1
}
```

### Para dataset desbalanceado
- Focal Loss (γ=2)
- Class weights inversamente proporcionais
- Oversampling da classe minoritária

### Para poucos dados
- LoRA (r=8, alpha=16)
- Gradient checkpointing
- Early stopping agressivo

---

## Resumo

| Modelo | Score | vs Baseline |
|--------|-------|-------------|
| **BERTimbau v4 + Threshold** | **0.82073** | **+5.4%** 🏆 |
| BERTimbau + Focal | 0.79696 | +2.3% |
| BERTimbau v5 (Gamma) | 0.75574 | -3.0% |
| BERTimbau v5 (LR) | 0.75508 | -3.0% |
| BioBERTpt | 0.72480 | -7.0% |
| BERTimbau v5 (Class) | 0.69238 | -11.1% |
| XLM-RoBERTa | 0.68767 | -11.7% |
| ModernBERT | 0.68578 | -12.0% |
| BERTimbau | 0.64319 | -17.4% |
| BERT Multilingual | 0.56095 | -28.0% |
| DistilBERT | 0.55229 | -29.1% |
| BERTimbau + LoRA | 0.13261 | ❌ Falhou |
| mDeBERTa + CW | 0.01008 | ❌ BUG |

**Conclusão:** 🏆 **BERTimbau v4 + Threshold Tuning é o melhor modelo (0.82073)**

**Insights v5:**
- ❌ Variações de LR/Gamma não melhoram
- ❌ Class weights + Focal Loss é redundância
- ✅ Threshold tuning é o diferencial

**Próximos passos:**
1. Seed ensemble (3-5 seeds) com v4
2. CV-based threshold estimation
3. Testar alpha α weights no Focal Loss

---

## Rodada 2026-04-08 (colegas)

Cinco notebooks rodados pelos colegas, sem scores publicos ainda (em avaliacao):

| Notebook | Modelo | Tecnicas Novas | Epochs | Batch efetivo |
|----------|--------|----------------|--------|---------------|
| bertimbau_awp_v1 | BERTimbau Large | AWP + Focal + Label Smooth + Optuna 300 trials | 5 | 8 |
| bertimbau_large_optuna_v1 | BERTimbau Large | Focal + Label Smooth + Optuna 300 trials | 5 | 8 |
| mdeberta_v3_focal_v1 | mDeBERTa-v3-base | Focal + Label Smooth + Optuna 300 trials + Early Stop | 7 | 16 |
| xlmroberta_large_focal_v1 | XLM-RoBERTa Large | Focal + Label Smooth + Grad Accum (4x4) + Optuna 300 | 5 | 16 |
| ensemble_stacking_v1 | BERTimbau + mDeBERTa + XLM-R base | Stacking LightGBM meta-learner + Optuna 200 | 3 | variavel |

### Tecnicas novas em relacao ao estado anterior

**AWP (Adversarial Weight Perturbation):**
```python
AWP_LR = 1e-1
AWP_EPS = 1e-2
AWP_START_EPOCH = 1  # Warmup de 1 epoch antes de ativar
# Perturba pesos na direcao do gradiente -> minimos mais planos -> melhor generalizacao
# Estimativa dos colegas: +0.3-0.5 pp
```

**Optuna Joint Calibration (300 trials):**
```python
# Otimiza simultaneamente: 1 temperatura + 7 thresholds por classe
# Sampler: TPESampler(seed=42)
# Muito superior ao grid search sequencial
N_TRIALS = 300
```

**Label Smoothing:**
```python
LABEL_SMOOTHING = 0.05  # Regularizacao leve, nao prejudica threshold tuning
```

**Gradient Accumulation (XLM-R Large):**
```python
BATCH_SIZE = 4
GRAD_ACCUM = 4  # Effective batch = 16
LR = 1e-5       # LR menor para modelo de 550M params
```

**Stacking com LightGBM meta-learner:**
```python
# OOF predictions de 3 modelos -> stack de (N, 21) features
# Meta-learner LGB com early stopping (50 rounds)
# Optuna 200 trials sobre meta-learner OOF
lgb_params = {'learning_rate': 0.05, 'num_leaves': 31, 'n_estimators': 500}
```

### Correcao importante: mDeBERTa

O bug fp16 anterior (score 0.01) foi porque o modelo anterior usava `token_type_ids` incorretamente.
mDeBERTa NAO usa token_type_ids. Correto:
```python
# CORRETO para mDeBERTa e XLM-RoBERTa:
inputs = {'input_ids': ..., 'attention_mask': ...}  # sem token_type_ids
# ERRADO (causava o bug):
inputs = {'input_ids': ..., 'attention_mask': ..., 'token_type_ids': ...}
```

---

*Atualizado em: 17/04/2026*
