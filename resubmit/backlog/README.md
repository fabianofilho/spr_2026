# Backlog - Notebooks para Submissões Futuras

Notebooks preparados mas ainda não submetidos.

## Prioridade Alta

| Notebook | Descrição | Origem |
|----------|-----------|--------|
| resubmit_bertimbau_threshold_v6.ipynb | **Threshold tuning por classe** | Colab |
| resubmit_super_ensemble_v2.ipynb | Stacking + Tratamento + SMOTE | - |
| resubmit_ensemble_v4.ipynb | Pesos otimizados | - |
| resubmit_bertimbau_v5.ipynb | Melhorias incrementais | - |

## Prioridade Média

| Notebook | Descrição |
|----------|-----------|
| resubmit_modernbert_v2.ipynb | Focal Loss |
| resubmit_xlmroberta_v2.ipynb | Focal Loss |
| resubmit_biobertpt_v3.ipynb | Ajustes |
| resubmit_ensemble_v5.ipynb | Variação |

## LLMs (Baixa Prioridade)

| Notebook | Descrição |
|----------|-----------|
| resubmit_gemma3_oneshot.ipynb | Few-shot |
| resubmit_llama3_oneshot.ipynb | Few-shot |
| resubmit_qwen3_4b_oneshot.ipynb | Few-shot |

## Iterações

| Notebook | Descrição |
|----------|-----------|
| resubmit_sgd_v6.ipynb | Próxima iteração SGD |

---

## Estratégias do Colab

Experimentos testados no Google Colab (`colab/`):

| Técnica | Resultado (Val) | Notebook |
|---------|-----------------|----------|
| MLM + gamma=2.0 + threshold | **0.84896** | otimizacao_threshold_e_gamma.ipynb |
| MLM + gamma=1.0 + threshold | 0.80193 | otimizacao_threshold_e_gamma.ipynb |
| Domain Adaptive Pretrain (MLM) | - | domain_adaptive_pretrain_CORRIGIDO.ipynb |
| Fine-tune BERTimbau + Focal | - | finetune_bertimbau_medical_focal.ipynb |

**Próximo passo:** Testar threshold tuning no Kaggle com `resubmit_bertimbau_threshold_v6.ipynb`
