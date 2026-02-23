# Transformers - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 🏆 | **BERTimbau + Focal Loss** | **0.79696** | ✅ Submetido |
| 2 | ModernBERT base | 0.68578 | ✅ Submetido |
| 3 | BERTimbau base | 0.64319 | ✅ Submetido |
| 4 | BERT Multilingual | 0.56095 | ✅ Submetido |
| 5 | DistilBERT Multilingual | 0.55229 | ✅ Submetido |
| ❌ | BERTimbau + LoRA (Offline) | 0.13261 | ⚠️ Falhou |
| ❌ | mDeBERTa + class weights | 0.01008 | ⚠️ BUG |
| - | BioBERTpt | - | ⏳ Pendente |
| - | mDeBERTa-v3 (sem class weights) | - | ⏳ Pendente |
| - | XLM-RoBERTa + Mean Pool | - | ⏳ Pendente |
| - | Custom Transformer | - | ⏳ Pendente |

---

## 🏆 Análise: BERTimbau + Focal Loss (0.79696) - MELHOR SCORE!

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
| **BERTimbau + Focal** | **0.79696** | **+2.3%** 🏆 |
| ModernBERT | 0.68578 | -12% |
| BERTimbau | 0.64319 | -17% |
| BERT Multilingual | 0.56095 | -28% |
| DistilBERT | 0.55229 | -29% |
| BERTimbau + LoRA | 0.13261 | ❌ Falhou |
| mDeBERTa + CW | 0.01008 | ❌ BUG |

**Conclusão:** 🎉 **BERTimbau + Focal Loss superou o baseline TF-IDF (0.77885)!** Primeiro transformer a vencer.

**Próximos passos:**
1. Investigar por que LoRA falhou (0.13261)
2. Testar BioBERTpt com Focal Loss
3. Testar mDeBERTa SEM class weights

---

*Atualizado em: 23/02/2026*
