# Validação: mDeBERTa-v3 (FP32 Fix)

**Modelo:** mDeBERTa-v3-base multilingual (`microsoft/mdeberta-v3-base`)

---

## ⚠️ Histórico
- Score **0.01008** em submissões anteriores (BUG)
- Problema: fp16/mixed precision incompatível
- Fix: `model.float()` força TODOS os parâmetros para fp32

## 🎯 Objetivo
Testar múltiplas configurações para encontrar a melhor performance.

## 📊 Configurações Planejadas
- **Loss:** CrossEntropy, Focal Loss (γ=1,2,3)
- **LR:** 1e-5, 2e-5, 3e-5
- **Batch Size:** 8, 16
- **Max Length:** 256, 512

---

## ❌ RESULTADO: FALHA - MODELO ERRADO CARREGADO

### Ambiente
- **Device:** cpu (!)
- **Model encontrado:** `/kaggle/input/models/qwen-lm/qwen-3/transformers/4b/1`

### Problema
```python
# Esperado:
Model: microsoft/mdeberta-v3-base

# Encontrado:
Model: /kaggle/input/models/qwen-lm/qwen-3/transformers/4b/1
Qwen3ForSequenceClassification LOAD REPORT
```

**A função `find_model_path()` encontrou Qwen3 ao invés de mDeBERTa!**

O mDeBERTa NÃO estava disponível como dataset no Kaggle. A busca recursiva encontrou o primeiro modelo com `config.json` (Qwen3).

---

## 📊 RESUMO

| Config | F1-Macro | Status |
|--------|----------|--------|
| CE_baseline | - | ❌ Modelo errado (Qwen3) |
| Focal_g2 | - | ❌ Não executado |
| Focal_g3 | - | ❌ Não executado |
| Focal_lr1e5 | - | ❌ Não executado |
| Focal_maxlen512 | - | ❌ Não executado |

### Referências
- BERTimbau v4: 0.82073
- Score anterior mDeBERTa: 0.01008 (BUG)

---

## 📝 INSIGHTS

1. **Problema Principal:**
   - mDeBERTa NÃO estava disponível offline no Kaggle
   - `find_model_path()` é falha - encontra qualquer modelo
   - Precisa de busca específica por nome do modelo

2. **O que sabemos sobre mDeBERTa:**
   - Requer `fp16=False` obrigatório
   - Requer `model.float()` após carregar
   - Score anterior de 0.01008 era BUG de precisão

3. **Ações Necessárias:**
   - Criar notebook `models/deberta/download_mdeberta.ipynb`
   - Upload para Kaggle como dataset privado
   - Re-executar validação com modelo correto
   - Modificar `find_model_path()` para busca específica

4. **Fix para find_model_path:**
   ```python
   def find_model_path(model_name):
       """Busca modelo específico no Kaggle."""
       base = '/kaggle/input'
       for item in os.listdir(base):
           if model_name.lower() in item.lower():
               # continuar busca...
   ```

---

## ⚠️ STATUS: REQUER AÇÃO

- [ ] Criar notebook download_mdeberta.ipynb
- [ ] Upload mDeBERTa para Kaggle
- [ ] Modificar função de busca para ser específica
- [ ] Re-executar validação com fp16=False
- [ ] Documentar resultados reais

### Expectativa se Funcionar
Com base em modelos similares:
- CrossEntropy: ~0.65-0.70
- Focal Loss γ=2: ~0.70-0.75
- Com threshold tuning: ~0.75-0.78

mDeBERTa é conhecido por bom desempenho multilingual, pode competir com BERTimbau se configurado corretamente.

*Validação inconclusiva - repetir após download do modelo correto.*
