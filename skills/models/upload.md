# Skill: Upload Manual de Modelos

## 🎯 Quando Ativar

- Modelo não disponível no Kaggle Models
- Precisa usar BERTimbau, BioBERTpt, etc.
- Upload de modelos HuggingFace para Kaggle Datasets

---

## 📋 Workflow Completo

### 1. Baixar Modelo Localmente

```bash
huggingface-cli download neuralmind/bert-base-portuguese-cased \
    --local-dir models/bert-base-portuguese-cased
```

### 2. Limpar Arquivos Desnecessários

```bash
cd models/bert-base-portuguese-cased
rm -rf .cache flax_model.msgpack tf_model.h5
```

Arquivos a remover (se existirem):
- `.cache/`
- `flax_model.msgpack` (Jax/Flax)
- `tf_model.h5` (TensorFlow)
- `.git/` (se baixou com git clone)

### 3. Upload no Kaggle

1. Ir para **Kaggle → Datasets → New Dataset**
2. **Upload pasta** do modelo
3. Dar nome descritivo (ex: `bert-base-portuguese-cased`)
4. Definir como **Private** se necessário
5. Clicar **Create**

### 4. Usar no Notebook

1. No notebook Kaggle → **Add Input**
2. Aba **Your Work** ou **Datasets**
3. Selecionar o dataset criado
4. Usar path: `/kaggle/input/{dataset-slug}/...`

---

## 📝 Modelos Comuns para Upload

| Modelo | Comando de Download |
|--------|---------------------|
| BERTimbau | `huggingface-cli download neuralmind/bert-base-portuguese-cased --local-dir models/bert-base-portuguese-cased` |
| BERTimbau Large | `huggingface-cli download neuralmind/bert-large-portuguese-cased --local-dir models/bert-large-portuguese-cased` |
| BioBERTpt | `huggingface-cli download pucpr/biobertpt-all --local-dir models/biobertpt-all` |
| XLM-RoBERTa | `huggingface-cli download xlm-roberta-base --local-dir models/xlm-roberta-base` |

---

## 🔧 Script: Download e Limpeza

```bash
#!/bin/bash
# download_model.sh

MODEL_NAME=$1  # Ex: neuralmind/bert-base-portuguese-cased
LOCAL_DIR=$2   # Ex: models/bertimbau

# Download
huggingface-cli download $MODEL_NAME --local-dir $LOCAL_DIR

# Limpeza
cd $LOCAL_DIR
rm -rf .cache flax_model.msgpack tf_model.h5 .git
echo "Modelo pronto para upload!"
du -sh .
```

Uso:
```bash
./download_model.sh neuralmind/bert-base-portuguese-cased models/bertimbau
```

---

## 📁 Estrutura Esperada Após Upload

```
/kaggle/input/{dataset-slug}/
├── config.json
├── pytorch_model.bin  (ou model.safetensors)
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── special_tokens_map.json
```

---

## ⚠️ Dicas Importantes

1. **Tamanho máximo**: 100GB por dataset
2. **Tempo de upload**: Modelos grandes podem demorar
3. **Versionamento**: Kaggle mantém versões do dataset
4. **Privacidade**: Datasets privados só você pode usar

---

## ✅ Checklist

- [ ] Modelo baixado localmente
- [ ] Arquivos desnecessários removidos
- [ ] Upload para Kaggle Datasets concluído
- [ ] Dataset adicionado como input no notebook
- [ ] Testado `find_model_path()` para encontrar
