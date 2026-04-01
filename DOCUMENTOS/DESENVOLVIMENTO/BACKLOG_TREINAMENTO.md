# ###################################################################

Backlog — Ideias, Pontos Pendentes, Anotações, Dúvidas
Organizado por categoria.
Atualizar conforme novos pontos surgirem nas sessões de trabalho.

Ícones prioridade (Alta > Média > Baixa): 🔴 🟠 🟡

OBS: Atualizar repo: https://github.com/LorenzoFRR/axa-isa-workspace/tree/main :
git add .
git commit -m "descrição"
git push

OBS: Materiais de entregas estão no Confluence: https://pswdigital.atlassian.net/wiki/spaces/AI/overview

# ###################################################################
# ###################################################################

# Definição desenvolvimento 30/03
- Pipeline distribuição 🔵
- Desenvolvimento V11 🔵

# Mapeamento desenvolvimento [STDBY]
- Avaliar necessidade análise capacidade operacional
- Criar modelos abrindo além de SEG_NOVO/RENOVACAO/MANUAL/DIGITAL
- Explorar novos modelos/arquiteturas
- Ver treinamento assumindo peso para features a partir de FS

# ###################################################################
# ###################################################################

# Integrações
- [ ] Integração Claude/Databricks 🔴
- [ ] Integração versionamento ISA_DEV git 🟠
- [ ] Unity Catalog 🔴

# Otimização desenvolvimento
- [ ] Revisar CLAUDE.md 🔴
- [ ] Verificar consumo/usage 🔴
- [ ] Estudar seção AI/ML Databricks
  - Models
- [ ] Pipeline distribuição 🔵
- [ ] Criar visualização pra verificar relações notebooks/fluxos/modes/etc 🟠
  - Análise lineage pipeline
    - Para uma dada tabela, verificar quais runs estão associadas
    - Para uma dada run, verificar quais tabelas estão associadas
    - Para cada VX, qual MODE está associada
    - Mostrar visão ramificada

## PENDÊNCIAS SEM ETAPA
- [ ] Padronizar nomes tabelas geradas no pipeline
- [ ] Executar/comparar treino com mesmas colunas que MODE_C, substituindo por CLF_CORRETOR 🔴
  - Criar toggle de definição de colunas manualmente, independente do resultado de FS (OBS adicionada em TREINO)


## 0_INGESTAO
- [ ] Re-avaliar nomenclatura tabelas de corretor
- [ ] Logar profiling tabelas de corretor

## 1_PRE_PROC
  - [ ] Fazer análise função que calcula regra temporal colunas DIAS_*
  - [ ] Entender como trabalhar com DIAS_ANALISE_SUBSCRICAO para cotações DIGITAL
  - [ ] Remover DT_INICIO_VIGENCIA (tá aparecendo em silver.cotacao_seg_timestamp e em diante - removida em TREINO) 🔴


## 2_JOIN

## 0_INGESTAO

## 1_PRE_PROC

## 2_JOIN

## 3_TREINO

  ## SUBETAPA PRE_PROC_MODEL

  ## SUBETAPA CLUSTERING_EXPLORE (MODE_D specific)
  - [ ] Logar sillouette e elbow curve no mesmo gráfico
  - [ ] Revisar lógica estratificação dos atributos dos corretores
  - [STDBY] Avaliar novas colunas para inserir na etapa de clustering
  - [x] Plotar distribuições para as colunas envolvidas
  - [ ] Entender leitura sillouette/elbow 🟡
  - [ ] Entender estratégias de normalização implementadas 🟡

  ## SUBETAPA CLUSTERING_FIT (MODE_D specific)
  - [ ] Mostrar e selecionar lista de colunas a serem alimentadas no treino só após CLUSTERING_FIT 🔴

  ## SUBETAPA FEATURE SELECTION
  - [ ] Métodos paralelos de Análise de Importância
    - SHAP // Fornece sentido de relação (positivo, negativo)
    - Permutation Importance
    - RFE - Recursive Feature Elimination
  - [ ] Testar FS em grid

  ## SUBETAPA TREINO
  - [ ] Avaliar implementação Modelos de ranking direto (LambdaMART, LambdaRank)
  - [ ] Eleger Top-K features com base em curva de desempenho (AUC-PR x número de features)
  - [ ] Entender como trabalhar com calibração do score e influência decisão de threshold
    - Platt Scaling, Isotonic Regression
  - [ ] Entender como fica a métrica para a distribuição (talvez ranking puro não sirva)
  - [ ] Ver diferença de métricas no output da célula de execução
  - [ ] Ajustar número de colunas que entram no modelo (tenho as pinnadas + as resultantes de FS, mas tenho que inserir um limite) 🔴
  - [STDBY] Inserir variação de parâmetros de clustering em grid de treinamento (T_TREINO)
  - [ ] Verificar ajuste número de colunas inseridas no treinamento [MODE_C, MODE_D] 🔴
  - [ ] Criar toggle de definição de colunas manualmente, independente do resultado de FS 🔴

## 4_INFERENCIA

## 5_COMP
- [ ] Metrificar overfitting
- [ ] Plotar MI + Methods Score
- [ ] p_emitida por cluster 🔴

## 6_REPORT
- [ ] Ajustar visualização temporal/precision_monthly.png (mostrar para a segmentação correspondente) 🔴
- [ ] Logar colunas alimentadas nos modelos 🔴
- [ ] Logar informações de clustering, caso seja MODE_D 🔴
- [ ] Análise de resultados
  - Estratificar análises por linha de produto

