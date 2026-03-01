# Validação: LLMs Zero/Few-shot

**Objetivo:** Avaliar performance de LLMs sem fine-tuning para classificação BI-RADS.

---

## 📊 Modelos Planejados
| Modelo | Params | Método |
|--------|--------|--------|
| Qwen 3 | 4B | Zero-shot e Few-shot |
| MedGemma | 4B | Zero-shot com instrução BI-RADS |
| Phi-3.5 | 3.8B | Comparação |

## 🎯 Setup
- **Device:** cpu (!) - GPU não detectada
- **Model encontrado:** `/kaggle/input/models/keras/medgemma/keras/medgemma_4b/1`
- **Dataset:** 146 train, 63 val (amostra estratificada)

---

## ❌ RESULTADO: FALHA - ERRO NO TOKENIZER

### MedGemma 4B
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

KeyError: 'added_tokens'
```

**Erro:** O tokenizer do MedGemma (formato Keras) não é compatível com `AutoTokenizer` do HuggingFace.

### Causa
- MedGemma foi salvo em formato **Keras** (não PyTorch)
- Path: `keras/medgemma/keras/medgemma_4b`
- `AutoTokenizer` espera formato HuggingFace com `tokenizer.json` válido

---

## 📊 RESUMO

| Modelo | F1-Macro | Status |
|--------|----------|--------|
| Qwen 3 (4B) | - | ❌ Não encontrado |
| MedGemma (4B) | - | ❌ Erro: KeyError 'added_tokens' |
| Phi-3.5 (3.8B) | - | ❌ Não encontrado |

### Referências
- TF-IDF: 0.77885
- BERTimbau v4: 0.82073

---

## 📝 INSIGHTS

1. **Problema Principal:**
   - MedGemma em formato Keras, não PyTorch
   - Tokenizer incompatível com transformers
   - Precisa usar `keras_nlp` ou converter para PyTorch

2. **Problemas Secundários:**
   - Device = CPU (GPU não detectada)
   - Qwen3 e Phi-3.5 não estavam no Kaggle
   - `find_model_path()` encontrou MedGemma por acaso

3. **Ações Necessárias:**
   - Download MedGemma em formato PyTorch
   - Ou usar `keras_nlp.models.GemmaCausalLM`
   - Upload Qwen3 e Phi-3.5 como datasets

4. **Alternativa Testada:**
   - `resubmit/2026-02-25/resubmit_qwen3_zeroshot.ipynb`
   - `resubmit/2026-02-27/resubmit_mistral_oneshot.ipynb`
   - Verificar resultados nesses notebooks

---

## ⚠️ STATUS: REQUER AÇÃO

- [ ] Baixar MedGemma formato PyTorch (não Keras)
- [ ] Criar notebook `download_qwen3.ipynb`
- [ ] Criar notebook `download_phi3.ipynb`
- [ ] Re-executar validação com modelos corretos
- [ ] Verificar GPU no ambiente Kaggle

### Código para MedGemma (Keras)
Se quiser usar o formato Keras:
```python
import keras_nlp
gemma = keras_nlp.models.GemmaCausalLM.from_preset(
    "gemma_4b_en",
    weights="/kaggle/input/models/keras/medgemma/..."
)
```

*Validação inconclusiva - repetir após correção dos modelos.*
