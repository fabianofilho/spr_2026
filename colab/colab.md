Peguei o BERTimbau (BERT em português) e fiz um pré-treino extra com os laudos de mamografia pra ele aprender termos médicos (MLM). Depois fiz fine-tuning com Focal Loss e otimizei o gamma e os thresholds por classe.
Resultados:
Baseline Kaggle (sem MLM):        0.79696
MLM + gamma=2.0 + argmax:         0.78665
MLM + gamma=2.0 + threshold:      0.84896
MLM + gamma=1.0 + argmax:         0.79060
MLM + gamma=1.0 + threshold:      0.80193
Meta (leaderboard):               ~0.84
O 0.849 do gamma=2.0 + threshold é o melhor na validação mas pode ser overfitting no threshold (pouca amostra nas classes raras). O 0.802 do gamma=1.0 + threshold é mais seguro.
Sugiro submeter os dois e ver qual vai melhor no Kaggle. Os submissions estão no Drive em /spr_2026/submissions/
Te mandei 3 notebooks + um README explicando tudo.