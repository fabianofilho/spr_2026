# TODO 

Este documento descreve uma lista priorizada de 20 experimentos para melhorar o desempenho na competição, com base na análise do repositório, da competição, da disciplina CS224N de Stanford e da literatura de NLP para textos médicos.

## Prioridade Alta: Modelos Robustos e Técnicas Comprovadas

O foco inicial deve ser em otimizar os modelos mais promissores, individualmente, usando técnicas robustas para lidar com as características do problema (texto em português, desbalanceamento de classes).

- [ ] **1. BERTimbau-Large + Focal Loss:**
  - **Modelo:** `neuralmind/bert-large-portuguese-cased`
  - **Técnica:** Substituir a loss padrão por `Focal Loss` para dar mais importância às classes minoritárias.
  - **Justificativa:** BERTimbau é o estado da arte para português. Focal Loss é uma técnica comprovada para desbalanceamento.

- [ ] **2. mDeBERTa-v3-base + Class Weights:**
  - **Modelo:** `microsoft/mdeberta-v3-base`
  - **Técnica:** Usar `CrossEntropyLoss` com `class_weights` calculados de forma inversamente proporcional à frequência de cada classe no treino.
  - **Justificativa:** mDeBERTa tem uma arquitetura robusta. Class weights é uma alternativa mais simples e eficaz que a loss padrão.

- [ ] **3. XLM-RoBERTa-Large + Otimização de Pooling:**
  - **Modelo:** `xlm-roberta-large`
  - **Técnica:** Comparar o desempenho da estratégia de pooling padrão (`CLS token`) com `mean pooling` dos outputs da última camada.
  - **Justificativa:** Modelos RoBERTa podem se beneficiar de estratégias de pooling diferentes do CLS. XLM-R é um baseline multilingue forte.

- [ ] **4. BioBERT em Português (BioBERTpt) + Fine-tuning Completo:**
  - **Modelo:** `pucpr/biobertpt-all`
  - **Técnica:** Realizar fine-tuning completo do modelo, que foi pré-treinado em textos biomédicos em português.
  - **Justificativa:** Modelos pré-treinados em domínios específicos (biomédico) tendem a ter um desempenho superior em tarefas do mesmo domínio.

- [ ] **5. PeLLE (RoBERTa-large-br) + Layer-wise Learning Rate Decay:**
  - **Modelo:** `pelle/roberta-large-br`
  - **Técnica:** Aplicar um learning rate maior para as camadas de classificação e um learning rate menor para as camadas de base do transformer (decaimento gradual).
  - **Justificativa:** PeLLE é um modelo RoBERTa treinado em um grande corpus de português brasileiro. Layer-wise LR decay pode evitar o esquecimento catastrófico.

## Prioridade Média: Ensembles e Adaptação Eficiente

Após ter baselines sólidos com modelos individuais, o próximo passo é combinar seus pontos fortes através de ensembles e explorar técnicas de fine-tuning mais eficientes.

- [ ] **6. Ensemble Ponderado (Top 3 Modelos):**
  - **Modelos:** Os 3 melhores modelos da fase anterior.
  - **Técnica:** Fazer uma média ponderada das probabilidades de saída. Os pesos podem ser definidos com base no F1-Score macro de validação (OOF) de cada modelo.
  - **Justificativa:** Ensembles simples são uma forma eficaz de aumentar a robustez e generalização.

- [ ] **7. Stacking com Meta-Learner:**
  - **Modelos:** Predições OOF dos 3-5 melhores modelos como features.
  - **Técnica:** Treinar um classificador simples (e.g., `LogisticRegression` ou `LightGBM`) sobre as predições dos modelos de base.
  - **Justificativa:** Stacking pode aprender a combinar as predições de forma mais inteligente que uma média simples.

- [ ] **8. BERTimbau-Large + LoRA:**
  - **Modelo:** `neuralmind/bert-large-portuguese-cased`
  - **Técnica:** Aplicar Low-Rank Adaptation (LoRA) para um fine-tuning mais rápido e com menos memória, treinando apenas uma pequena fração dos pesos.
  - **Justificativa:** PEFT/LoRA acelera a experimentação e pode levar a resultados comparáveis ou melhores que o fine-tuning completo, com menor custo computacional.

- [ ] **9. mDeBERTa-v3 + LoRA com Otimização de Hiperparâmetros:**
  - **Modelo:** `microsoft/mdeberta-v3-base`
  - **Técnica:** Usar LoRA e fazer uma busca de hiperparâmetros para o rank (`r`) e alpha da adaptação.
  - **Justificativa:** Encontrar os hiperparâmetros ótimos para LoRA é crucial para seu desempenho.

- [ ] **10. Ensemble de Modelos com e sem LoRA:**
  - **Modelos:** Combinar um modelo com fine-tuning completo e sua versão com LoRA.
  - **Técnica:** Voting ou Stacking.
  - **Justificativa:** Modelos treinados com diferentes técnicas de fine-tuning podem ter erros não correlacionados, tornando o ensemble mais eficaz.

- [ ] **11. Aumento de Dados com Back-Translation:**
  - **Técnica:** Traduzir exemplos das classes minoritárias para outra língua (e.g., inglês) e depois de volta para o português para gerar novas amostras sintéticas.
  - **Justificativa:** Aumento de dados pode ajudar a mitigar o desbalanceamento de classes e melhorar a robustez do modelo.

## Prioridade Baixa: Abordagens Experimentais e Otimizações Avançadas

Esta fase explora técnicas mais avançadas e arquiteturas que podem oferecer ganhos marginais ou exigir maior esforço de implementação.

- [ ] **12. Custom Transformer com Camadas Adicionais:**
  - **Modelo:** Arquitetura customizada (como no notebook `10_custom_transformer.ipynb`).
  - **Técnica:** Adicionar mais camadas de self-attention no topo de um encoder pré-treinado (e.g., BERTimbau) e treiná-las do zero.
  - **Justificativa:** Pode permitir que o modelo aprenda representações mais específicas para a tarefa.

- [ ] **13. Multi-Task Learning: Classificação + Priorização:**
  - **Modelo:** Qualquer transformer de bom desempenho.
  - **Técnica:** Criar uma cabeça de classificação com duas saídas: uma para a categoria BI-RADS (7 classes) e outra para a prioridade (alta/baixa).
  - **Justificativa:** Forçar o modelo a aprender uma tarefa relacionada pode atuar como uma forma de regularização e melhorar o desempenho na tarefa principal.

- [ ] **14. Otimização de `max_length`:**
  - **Técnica:** Avaliar o impacto de aumentar o `max_length` para 1024 (ou mais, se a memória permitir), especialmente para modelos como Longformer ou BigBird, se disponíveis para português.
  - **Justificativa:** Laudos médicos podem ser longos, e um `max_length` maior pode capturar mais contexto.

- [ ] **15. T5 (Text-to-Text) para Classificação:**
  - **Modelo:** `google/mt5-base` ou uma versão em português.
  - **Técnica:** Formular a tarefa como um problema de geração de texto, onde o modelo deve gerar o número da classe BI-RADS como uma string (e.g., "2").
  - **Justificativa:** Abordagem diferente que pode capturar nuances da linguagem de forma distinta dos modelos baseados em encoders.

- [ ] **16. Ensemble de TF-IDF e Transformers:**
  - **Modelos:** Combinar o melhor modelo TF-IDF (e.g., com Regressão Logística) e o melhor modelo Transformer.
  - **Técnica:** Stacking.
  - **Justificativa:** Modelos de `bag-of-words` e transformers aprendem representações muito diferentes; sua combinação pode ser poderosa.

- [ ] **17. Fine-tuning com RAG (Retrieval-Augmented Generation):**
  - **Técnica:** Implementar um pipeline onde, para cada laudo, o sistema recupera exemplos semelhantes do conjunto de treino e os fornece como contexto adicional para o modelo de classificação.
  - **Justificativa:** RAG pode fornecer exemplos `in-context` que ajudam o modelo a tomar decisões mais informadas, especialmente para casos raros.

- [ ] **18. Explorar Modelos de Embedding de Sentenças (Sentence Transformers):**
  - **Modelo:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` ou `neuralmind/bert-base-portuguese-cased` (com pooling de SBERT).
  - **Técnica:** Gerar embeddings para os laudos e treinar classificadores `LightGBM` ou `XGBoost` sobre eles.
  - **Justificativa:** Alternativa mais rápida ao fine-tuning de transformers, que pode ter um bom desempenho.

- [ ] **19. Pseudo-Labeling nos Dados de Teste:**
  - **Técnica:** Usar o melhor modelo atual para gerar predições no conjunto de teste. Adicionar as predições mais confiantes (e.g., probabilidade > 0.95) ao conjunto de treino e re-treinar o modelo.
  - **Justificativa:** Técnica de aprendizado semi-supervisionado que pode ajudar o modelo a generalizar melhor, mas deve ser usada com cautela.

- [ ] **20. Otimização de Hiperparâmetros com Optuna/Hyperopt:**
  - **Técnica:** Realizar uma busca bayesiana de hiperparâmetros para o melhor modelo, otimizando learning rate, batch size, weight decay e parâmetros da loss function.
  - **Justificativa:** Uma busca sistemática pode encontrar uma combinação de hiperparâmetros superior à busca manual.
