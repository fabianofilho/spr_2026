# Transformers - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| - | BERTimbau base | - | ⏳ Pendente |
| - | BERTimbau large + Focal | - | ⏳ Pendente |
| - | BERTimbau + LoRA | - | ⏳ Pendente |
| - | BioBERTpt | - | ⏳ Pendente |
| - | mDeBERTa-v3 | - | ⏳ Pendente |
| - | mDeBERTa + class weights | - | ⏳ Pendente |
| - | DistilBERT | - | ⏳ Pendente |
| - | XLM-RoBERTa + Mean Pool | - | ⏳ Pendente |
| - | Custom Transformer | - | ⏳ Pendente |

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
