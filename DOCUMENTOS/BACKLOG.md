####################################################################

Backlog — Ideias, Pontos Pendentes, Anotações, Dúvidas
Organizado por categoria.
Atualizar conforme novos pontos surgirem nas sessões de trabalho.

Ícones prioridade (Alta > Média > Baixa): 🔴 🟠 🟡
Ícone em andamento/Dependente de tarefa em andamento: 🔵
Ícone documento específico: 📄

OBS: Atualizar repo https://github.com/LorenzoFRR/axa-isa-workspace/tree/main :
git add .
git commit -m "descrição"
git push

####################################################################

# Tópicos 
## Databricks
- Spark Declarative Pipelines

## AI, desenvolvimento
- Boas práticas desenvolvimento de projetos Claude Clode
- XP Pair Programming
- Diagrama Entidade Relacionamento
- Arquitetura SDD
- Ralph Loops
- AI agents autoresearch (https://github.com/karpathy/autoresearch)
- Plugin Superpowers

####################################################################

# Integrações / Otimização desenvolvimento
- [ ] Integração Claude/Databricks
- [ ] Integração versionamento ISA_DEV git 🔴

# Documentação
- [ ] Gerar documentação automatizada a partir dos notebooks (ex: via nbconvert)
  - Documentar cada notebook do ISA_DEV com descrição de inputs, outputs e lógica
  - Mapear dependências entre os notebooks (qual alimenta qual)
  - Criação análise lineage
    - e.g. Para uma dada tabela, verificar quais runs estão associadas
    - e.g. Para uma dada run, verificar quais tabelas estão associadas
    - e.g. Para cada VX, qual MODE está associada
    - e.g. Mostrar visão ramificada

####################################################################

# Direcionamentos Gerais
- [ ] Avaliar necessidade análise capacidade operacional
- [STDBY] Criar modelos abrindo além de SEG_NOVO/RENOVACAO/MANUAL/DIGITAL
- [ ] Treinar modelos com novas abordagens 
  - [STDBY] Utilizar classificação, removendo HR, QTDs (TODAS SEGMENTACOES)
  - [STDBY] Utilizar clusterização
- [ ] Ver materiais Fabiano 🔴
- [STDBY] Ver treinamento assumindo peso para features a partir de FS

## Definição desenvolvimento 23/03
- [ ] Desenvolver lógica de report automático
- [ ] Desenvolver diagramas pipeline
- [ ] Análise de resultados
  - [ ] Desenvolver análise de resultados mais próxima do negócio -> relacionar com esforço do time comercial
  - [ ] Estratificar análises por linha de produto
- [ ] Executar pipeline para demais segmentações
- [ ] Desenvolver pipeline de distribuição
  - Executar e distribuir inferência com dados jan/fev 2026?
  - Criar pipeline geral
  - Entender como fica a métrica para a distribuição (talvez ranking puro não sirva)




## Possibilidades de direcionamento
- [ ] Explorar novos modelos
- [ ] Executar runs exploratórias -> Report sistemático

####################################################################
- [ ] Limpar catalogo 🔴
- [ ] Verificar lógicas de lineage 🟠
- [ ] Verificar erros TREINO/INF/COMP em v10.0.0 (20/03)

# 0_INGESTAO
- [ ] Re-avaliar nomenclatura tabelas de corretor 🟠
- [ ] Logar profiling tabelas de corretor 🟠


# 1_PRE_PROC
  - [ ] Fazer análise função que calcula regra temporal colunas DIAS_* 🟠
  - [ ] Ver quando remover DT_INICIO_VIGENCIA justamente porque já crio DIAS_INICIO_VIGENCIA 🟠
  - [ ] Entender como trabalhar com DIAS_ANALISE_SUBSCRICAO para cotações DIGITAL 🟠

# 2_JOIN

# 0_INGESTAO

# 1_PRE_PROC

# 2_JOIN

# 3_TREINO

  # ETAPA PRE_PROC_MODEL
  - [ ] Ajustar json pra acomodar 'rules_feature_prep' 🔴

  # ETAPA FEATURE SELECTION
  - [ ] Métodos paralelos de Análise de Importância
    - SHAP // Fornece sentido de relação (positivo, negativo)
    - Permutation Importance
    - RFE - Recursive Feature Elimination
  - [ ] Testar FS em grid
  - [ ] Ver se as regras de pre-processamento foram aplicadas e são passadas à etapa de FS (DS_SISTEMA, por exemplo, tá entrando em FS) 🔴

  # ETAPA TREINO
  - [ ] Avaliar implementação Modelos de ranking direto (LambdaMART, LambdaRank)
  - [ ] Eleger Top-K features com base em curva de desempenho (AUC-PR x número de features)
  - [ ] Entender como trabalhar com calibração do score e influência decisão de threshold
    - Platt Scaling, Isotonic Regression

  - [ ] Avaliar anomalias em EVAL 🟠

# 4_INFERENCIA

# 5_COMP

# 6_REPORT
- [ ] Referenciar run_id de modelos e gerar comparações 🔴


