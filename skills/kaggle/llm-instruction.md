# Skill: LLM com Instruction (BI-RADS)

## 🎯 Quando Ativar

- Usar LLMs para classificação BI-RADS
- Criar notebooks com prompt engineering
- Abordagem instruction-based (não zero-shot ou one-shot)

---

## 📋 Abordagem Instruction vs Zero/One-Shot

| Abordagem | Descrição | Quando Usar |
|-----------|-----------|-------------|
| Zero-shot | Só pergunta, sem exemplos | Modelos muito grandes |
| One-shot | 1 exemplo por classe | Modelos médios |
| **Instruction** | Explicação detalhada do domínio | Modelos médicos/especializados |

**Por que Instruction funciona melhor para BI-RADS:**
- Classificação requer conhecimento médico específico
- Categorias têm critérios técnicos bem definidos
- Modelo precisa entender semântica de malignidade

---

## 🔧 System Prompt BI-RADS (USAR SEMPRE)

```python
BIRADS_SYSTEM_PROMPT = '''Você é um radiologista especialista em mamografia.

Sua tarefa é classificar laudos de mamografia na categoria BI-RADS correta (0 a 6).

## CATEGORIAS BI-RADS:

**0 - INCOMPLETO**
- Avaliação incompleta, necessita exames adicionais
- Palavras-chave: "complementar", "inconclusivo", "necessita", "ultrassonografia adicional"

**1 - NEGATIVO**  
- Exame normal, sem achados
- Palavras-chave: "normal", "negativo", "sem alterações", "exame normal"

**2 - ACHADOS BENIGNOS**
- Achados definitivamente benignos
- Palavras-chave: "benigno", "cisto simples", "linfonodo", "calcificação benigna", "fibroadenoma"

**3 - PROVAVELMENTE BENIGNO**
- Achado provavelmente benigno (<2% malignidade)
- Palavras-chave: "provavelmente benigno", "controle", "seguimento", "6 meses"

**4 - SUSPEITO**
- Achado suspeito (2-95% malignidade)
- Subtipos: 4A (baixa), 4B (média), 4C (alta suspeita)  
- Palavras-chave: "suspeito", "biópsia", "4A", "4B", "4C", "PAAF"

**5 - ALTAMENTE SUSPEITO**
- Alta probabilidade de malignidade (>95%)
- Palavras-chave: "altamente suspeito", "maligno", "carcinoma", "espiculado"

**6 - MALIGNIDADE CONFIRMADA**
- Malignidade já comprovada por biópsia
- Palavras-chave: "confirmado", "biópsia prévia positiva", "carcinoma confirmado"

## INSTRUÇÕES:
1. Leia o laudo de mamografia
2. Identifique palavras-chave e contexto clínico
3. Classifique na categoria BI-RADS mais apropriada
4. Responda APENAS com o número da categoria (0-6)
'''
```

---

## 📝 Template: Notebook LLM Instruction

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================
# CONFIGURAÇÃO
# ============================================
MODEL_PATH = find_model_path()  # Usar função padrão
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BIRADS_SYSTEM_PROMPT = '''...'''  # Prompt completo acima

# ============================================
# CARREGAR MODELO
# ============================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map='auto',
    local_files_only=True
)

# ============================================
# FUNÇÃO DE CLASSIFICAÇÃO
# ============================================
def classify_report(report_text):
    messages = [
        {"role": "system", "content": BIRADS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Classifique este laudo:\n\n{report_text}"}
    ]
    
    # Para modelos chat
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{BIRADS_SYSTEM_PROMPT}\n\nLaudo:\n{report_text}\n\nCategoria:"
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extrair número
    for char in response.strip():
        if char.isdigit() and int(char) <= 6:
            return int(char)
    return 2  # fallback

# ============================================
# CLASSIFICAR DATASET
# ============================================
predictions = []
for report in tqdm(test_df['report'].values):
    pred = classify_report(report)
    predictions.append(pred)
```

---

## 📊 Modelos LLM Recomendados

### Modelos Médicos (preferidos):
| Modelo | Tamanho | Observação |
|--------|---------|------------|
| MedGemma 4B IT | 4B | Instruções médicas |
| MedGemma 27B Text IT | 27B | Maior, mais preciso |
| BioGPT Large | 1.5B | Treinado em PubMed |
| MedMO | 4B/8B | Multimodal médico |

### Modelos Gerais:
| Modelo | Tamanho | Observação |
|--------|---------|------------|
| Qwen 2.5 | 0.5B-72B | Bom para português |
| Phi-3.5 | 3.8B | Rápido e eficiente |
| Mistral | 7B | Boa qualidade |

---

## ⚠️ Considerações Importantes

1. **Temperatura baixa (0.1)** - Classificação precisa ser determinística
2. **max_new_tokens=5** - Só precisa de um dígito
3. **Fallback para classe 2** - Achados benignos é a mais comum
4. **Batch size 1** - LLMs não suportam bem batching
5. **Sempre `local_files_only=True`**

---

## 🔧 Extração de Resposta

```python
def extract_category(response):
    """Extrai categoria BI-RADS da resposta do LLM."""
    response = response.strip()
    
    # Tentar encontrar dígito 0-6
    for char in response:
        if char.isdigit():
            digit = int(char)
            if 0 <= digit <= 6:
                return digit
    
    # Fallback: buscar por palavras
    response_lower = response.lower()
    if 'zero' in response_lower or 'incompleto' in response_lower:
        return 0
    elif 'negativo' in response_lower:
        return 1
    elif 'benigno' in response_lower and 'provavelmente' not in response_lower:
        return 2
    elif 'provavelmente' in response_lower:
        return 3
    elif 'suspeito' in response_lower and 'altamente' not in response_lower:
        return 4
    elif 'altamente' in response_lower:
        return 5
    elif 'confirmad' in response_lower or 'maligno' in response_lower:
        return 6
    
    return 2  # Fallback seguro
```

---

## 📋 Checklist LLM Instruction

- [ ] System prompt com BI-RADS completo
- [ ] `local_files_only=True`
- [ ] `torch_dtype=torch.float16` para economia de memória
- [ ] Temperatura 0.1 (determinístico)
- [ ] max_new_tokens pequeno (5-10)
- [ ] Função de extração robusta
- [ ] Fallback para classe mais comum
