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

# Definição desenvolvimento 23/03
- Pipeline demais segmentações 🔵
- Pipeline distribuição 🔵
- Desenvolvimento V11 🔵

# Mapeamento desenvolvimento [STDBY]
- Avaliar necessidade análise capacidade operacional
- Criar modelos abrindo além de SEG_NOVO/RENOVACAO/MANUAL/DIGITAL
- Explorar novos modelos/arquiteturas
  - Utilizar classificação, removendo HR, QTDs (TODAS SEGMENTACOES)
  - Utilizar clusterização
- Ver treinamento assumindo peso para features a partir de FS

# ###################################################################
# ###################################################################

# Integrações / Otimização desenvolvimento
- [ ] Integração Claude/Databricks 🔴
- [ ] Integração versionamento ISA_DEV git 🟠
- [ ] Refazer integração Github (consulta Gabriel) 🔴
- [ ] Definir acompanhamento/gerenciamento projeto 🔴

# Desenvolvimento
- [ ] Revisar CLAUDE.md
- [ ] Unity Catalog 🔴
- [ ] Pipeline demais segmentações 🔵
  - [x] SEGURO_NOVO_MANUAL
  - [ ] RENOVACAO_MANUAL - Em andamento 28/03
  - [x] SEGURO_NOVO_DIGITAL
  - [ ] RENOVACAO_DIGITAL
- [ ] Pipeline distribuição 🔵

## 0_INGESTAO
- [ ] Re-avaliar nomenclatura tabelas de corretor
- [ ] Logar profiling tabelas de corretor

## 1_PRE_PROC
  - [ ] Fazer análise função que calcula regra temporal colunas DIAS_*
  - [ ] Ver quando remover DT_INICIO_VIGENCIA justamente porque já crio DIAS_INICIO_VIGENCIA
  - [ ] Entender como trabalhar com DIAS_ANALISE_SUBSCRICAO para cotações DIGITAL

## 2_JOIN

## 0_INGESTAO

## 1_PRE_PROC

## 2_JOIN

## 3_TREINO

  ## ETAPA PRE_PROC_MODEL
  - [ ] Implementação Clustering (V11) 🔵
    - Definir/entender dinâmica de seleção de parâmetros (EXPLORE, FIT?)
    - Logar distribuições e remover outliers pra não distorcer clustering
    - Ver como definir parâmetros de clustering (manual, automatizado?)
    - [STDBY] Inserir variação de parâmetros de clustering em grid de treinamento (T_TREINO)?
    - Logar, em rules, clustering como rule (toggle True, False pra aplicar clustering)
    - Logar análises/visualizações clustering + Entender interpretação análises
      - QTD corretor por cluster
      - Elbow / Sillouette
      - Heatmaps clusters
      - Distribuições HR vs QTD cotações por corretor
      - Distribuições HR vs QTD produtos por corretor
      - Distribuições QTD cotações por corretor vs QTD produtos por corretor
      - Tabela corretores típicos

    - Utilizar mais colunas no clustering, além de DS_PRODUTO_NOME, QTD_EMITIDO (não utilizado por ora), QTD_COTACAO, HR, por CD_CORRETOR
    - Implementar lógica de tal modo que os resultados da etapa possam ser referenciados em 5_COMP
    - Implementar regra de limpeza de base antes de executar clustering (como isso funcionaria? clustering seria necessariamente após subetapa de limpeza, usando cotacao_seg)
    - Definir análises executivas que podem ser realizadas a partir do resultado do clustering
    - [x] Verificar criação CLF_CORRETOR em df_model e df_validacao. R: Criado, OK.
    - [x] Dependendo do toggle de clusterizar por SEG, a tabela de cotacao_seg vem com SEG = todas. Verificar se isso acontece para MODE_C. R: Sim, cotacao_model/validacao vêm com SEG = todas.

  ## ETAPA FEATURE SELECTION
  - [ ] Métodos paralelos de Análise de Importância
    - SHAP // Fornece sentido de relação (positivo, negativo)
    - Permutation Importance
    - RFE - Recursive Feature Elimination
  - [ ] Testar FS em grid

  ## ETAPA TREINO
  - [ ] Avaliar implementação Modelos de ranking direto (LambdaMART, LambdaRank)
  - [ ] Eleger Top-K features com base em curva de desempenho (AUC-PR x número de features)
  - [ ] Entender como trabalhar com calibração do score e influência decisão de threshold
    - Platt Scaling, Isotonic Regression
  - [ ] Avaliar anomalias em EVAL
  - [ ] Entender como fica a métrica para a distribuição (talvez ranking puro não sirva)
  - [ ] Ver diferença de métricas no output da célula de execução

## 4_INFERENCIA

## 5_COMP
- [ ] Metrificar overfitting
- [ ] Plotar MI + Methods Score
- [ ] p_emitida por cluster

## 6_REPORT
- [ ] Análise de resultados
  - Estratificar análises por linha de produto
  - Lógica de report automático
  - Análise orientada decisões
    - Relacionar com esforço do time comercial

- [ ] Análise lineage pipeline
  - Para uma dada tabela, verificar quais runs estão associadas
  - Para uma dada run, verificar quais tabelas estão associadas
  - Para cada VX, qual MODE está associada
  - Mostrar visão ramificada