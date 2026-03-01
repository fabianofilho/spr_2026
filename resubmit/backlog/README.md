# Backlog - Notebooks para Submissões Futuras

✅ **Backlog vazio!** Todos os experimentos foram movidos para dias específicos.

---

## 📅 Cronograma de Submissões

| Data | Foco | Notebooks |
|------|------|-----------|
| 2026-03-01 | Threshold Tuning | BERTimbau v5 threshold_grid, cv_thresholds |
| 2026-03-02 | **Transformers Críticos** | mDeBERTa FP32 fix, α weights, γ search, LR search, maxlen 512 |
| 2026-03-03 | Regularização + Ensemble | Label smoothing, seed ensemble, super ensemble, ClinicalBERT |
| 2026-03-04 | LLMs Menores | BioGPT, MedGemma 4B, MedMO, BERTimbau ultimate |
| 2026-03-05 | LLMs Grandes | MedGemma 1.5 4B, MedGemma 27B, Qwen few-shot, MedGemma CoT |

---

## 🎯 Prioridade por Dia

### 2026-03-02 (Alta Prioridade)
- `resubmit_mdeberta_fp32_fix.ipynb` - **NUNCA TESTADO CORRETAMENTE** (potencial 0.80+)
- Hyperparameters do campeão (α, γ, LR, max_len)

### 2026-03-03 (Média Prioridade)
- Regularização (label smoothing)
- Ensembles e variações

### 2026-03-04-05 (Exploratório)
- LLMs para comparação

---

## 📊 Arquitetura Base (BERTimbau v4)

```python
# Config campeã
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.30, 6: 0.25}
```

## 🏆 Score Atual
- **BERTimbau v4 (Threshold Tuning): 0.82073** (+3% sobre baseline)
