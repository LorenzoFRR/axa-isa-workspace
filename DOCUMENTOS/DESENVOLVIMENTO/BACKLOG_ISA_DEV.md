# ###################################################################
Ícones prioridade (Alta > Média > Baixa): 🔴 🟠 🟡
# ###################################################################

🔵 (Implementação tem correspondência em BACKLOG_GERAL)
🟣 (Implementação sprint atual)

# ###################################################################
# ###################################################################

## PENDÊNCIAS SEM ETAPA
- [ ] Criação diagramas/visualizações de processo
  - Ramificações de execuções (via download MLflow)
  - Diagrama específico para TREINO_MODE_C/D (lógicas/fluxos/etc)
  - Dashboard consumo dados modelos/performance/lineage/etc
  - Análise lineage pipeline
    - Para uma dada tabela, verificar quais runs estão associadas
    - Para uma dada run, verificar quais tabelas estão associadas
    - Para cada VX, qual MODE está associada
    - Mostrar visão ramificada

- [ ] Avaliar necessidade análise capacidade operacional

- [ ] Organizar desenvolvimento 🔴
  - Atualizar LINEAGE_TABELAS_MANUAL.md
  - Revisar logs ao longo das etapas (evitar redundância, etc)
  - Renomear experiments/tables/notebooks
    - EXP: ISA_EXP -> ISA_DEV
      - Rename MLflow - OK
      - 0
      - 1
      - 2
      - 3
      - 4
      - 5
      - 6
  - Limpar workspace
    - Notebooks
    - Tables

- [ ] Criação comparação entre modelos (6_1_REPORT?)

## 0_INGESTAO
- [ ] Re-avaliar nomenclatura tabelas de corretor
- [ ] Logar profiling tabelas de corretor

## 1_PRE_PROC
  - [ ] Fazer análise função que calcula regra temporal colunas DIAS_*
  - [ ] Entender como trabalhar com DIAS_ANALISE_SUBSCRICAO para cotações DIGITAL
  - [ ] Remover DT_INICIO_VIGENCIA (tá aparecendo em silver.cotacao_seg_timestamp e em diante - removida em TREINO) 🟡

## 2_JOIN

## 0_INGESTAO
- [ ] Ver modos de ingestão para desenvolvimento
- [ ] Logar count por DATA_COTACAO

## 1_PRE_PROC
- [ ] Verificar leakage em regras geradas a partir de colunas de FLAG

## 2_JOIN

## 3_TREINO
  - [ ] Ajustar erro de import matplotlib
  - [ ] Testar novos modelos de clusterização

  ### SUBETAPA PRE_PROC_MODEL

  ### SUBETAPA CLUSTERING_EXPLORE (MODE_D specific)
  - [ ] Testar demais modelos de clustering
  - [ ] Logar sillouette e elbow curve no mesmo gráfico
  - [ ] Revisar lógica estratificação dos atributos dos corretores
  - [ ] Avaliar novas colunas para inserir na etapa de clustering
  - [ ] Entender leitura sillouette/elbow
  - [ ] Entender estratégias de normalização implementadas

  ### SUBETAPA CLUSTERING_FIT (MODE_D specific)

  ### SUBETAPA FEATURE SELECTION
  - [ ] Métodos paralelos de Análise de Importância
    - SHAP // Fornece sentido de relação (positivo, negativo)
    - Permutation Importance
    - RFE - Recursive Feature Elimination
  - [ ] Testar FS em grid
  - [ ] Ajustar nome da run container

  ### SUBETAPA TREINO
  - [ ] Explorar novos modelos e arquiteturas
    - Verificar códigos Jackson LGBM
    - Criar modelos abrindo além de SEG_NOVO/RENOVACAO/MANUAL/DIGITAL
    - Ver treinamento assumindo peso para features a partir de FS
    - Avaliar implementação Modelos de ranking direto (LambdaMART, LambdaRank)
  - [ ] Eleger Top-K features com base em curva de desempenho (AUC-PR x número de features)
  - [ ] Entender como trabalhar com calibração do score e influência decisão de threshold
    - Platt Scaling, Isotonic Regression
  - [ ] Entender como fica a métrica para a distribuição (talvez ranking puro não sirva)
  - [ ] Ver diferença de métricas no output da célula de execução
  - [ ] Inserir variação de parâmetros de clustering em grid de treinamento (T_TREINO)
    - Verificar consistência. Problema: Devo rodar a célula de inspeção (colunas disponíveis TREINO), que fica antes da etapa de TREINO > preencher a célula de config (primeira célula do notebook)
  - [ ] Ajustar número de colunas que entram no modelo (tenho as pinnadas + as resultantes de FS, mas tenho que inserir um limite)

## 4_INFERENCIA

## 5_COMP
- [ ] Executar comparação V11 para demais segmentações 🟡
- [ ] Ajuste LOGS 🟠
  - Artifacts/clustering
    - Remover clustering/score_distribution_by_cluster.png
    - Ver como é gerado ap/auc_pr _by_cluster.png
  - Artifacts/model_configs
    - Mostrar decisão: COLS ARBITRARIAS ou COLS FS
  - Artifacts/temporal
  - Verificar funcionamento lógica logs condicionais por MODE_CODE = D
  - Ajustar f-string nos gráficos


- [ ] Metrificar overfitting
- [ ] Plotar MI + Methods Score
- [ ] Estratificar análises por linha de produto

## 6_REPORT
- [ ] Aprimorar report de resultados 🟣 🔵
- [ ] Definir e espelhar logs de 5_COMP 🟡