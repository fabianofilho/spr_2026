# Transformers - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 1 | **ModernBERT base** | 0.68578 | ✅ Submetido |
| 2 | **BERTimbau base** | 0.64319 | ✅ Submetido |
| 3 | BERT Multilingual | 0.56095 | ✅ Submetido |
| - | BERTimbau large + Focal | - | ⏳ Pendente |
| - | BERTimbau + LoRA | - | ⏳ Pendente |
| - | BioBERTpt | - | ⏳ Pendente |
| - | mDeBERTa-v3 | - | ⏳ Pendente |
| - | mDeBERTa + class weights | - | ⏳ Pendente |
| - | DistilBERT | - | ⏳ Pendente |
| - | XLM-RoBERTa + Mean Pool | - | ⏳ Pendente |
| - | Custom Transformer | - | ⏳ Pendente |

---

## Análise: ModernBERT (0.68578) 🏆

**Melhor transformer até agora!** ModernBERT ficou apenas **12% abaixo** do TF-IDF baseline (0.77885).

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

## Análise (a ser preenchida após submissões)

*Aguardando resultados...*

---

*Atualizado em: 20/02/2026*
