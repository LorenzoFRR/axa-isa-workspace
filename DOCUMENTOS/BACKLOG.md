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
- Pipeline distribuição 🔵 !
- Desenvolvimento V11 🔵 !

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
  - [ ] RENOVACAO_MANUAL
  - [ ] SEGURO_NOVO_DIGITAL
  - [ ] RENOVACAO_DIGITAL
- [ ] Pipeline distribuição 🔵
- [ ] Desenvolvimento V11 🔵
  - [x] Criar documento ANALISE_V11.md com plano de análise exploratória CLF_CORRETOR
  - [ ] Executar análise exploratória
  - [ ] Decidir viabilidade e integração ao pipeline

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
- [x] Ajustar HR_2025 corretor_detalhe (bronze -> silver)

## 2_JOIN
- [ ] Deu problema nas mesmas colunas de _detalhe. Ajustar

## 3_TREINO

  ## ETAPA PRE_PROC_MODEL

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