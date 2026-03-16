####################################################################
Backlog — Ideias, Pontos Pendentes, Anotações, Dúvidas

Organizado por categoria.
Atualizar conforme novos pontos surgirem nas sessões de trabalho.

Ícones prioridade (Alta > Média > Baixa): 🔴 🟠 🟡
Ícone em andamento: 🔵
Ícone documento específico: 📄

OBS: Atualizar repo https://github.com/LorenzoFRR/axa-isa-workspace/tree/main :
git add .
git commit -m "descrição curta"
git push

####################################################################

# Integrações / Otimização desenvolvimento
- [STDBY] Integrar Claude Code diretamente ao Databricks (aguardando permissões)
- [STDBY] Automatizar exportação/importação de notebooks via Databricks CLI ou API
- [STDBY] Criar repositório Github e fornecer acesso para consulta do Workspace Claude
- [STDBY] Criar 'agente' contextualizado para tarefas de desenvolvimento de ML Pipelines
- [ ] Estudar frameworks
  - Boas práticas desenvolvimento de projetos Claude Clode
  - XP Pair Programming
  - Diagrama Entidade Relacionamento
  - Arquitetura SDD
- [ ] Ver integração Claude Code/Genie Code no ambiente 🔴
- [ ] Estudar Ralph Loops

# Tópicos Databricks
- [ ] Spark Declarative Pipelines

# Documentação
- [STDBY] Gerar documentação automatizada a partir dos notebooks (ex: via nbconvert)
  - [STDBY] Documentar cada notebook do ISA_DEV com descrição de inputs, outputs e lógica
  - [STDBY] Mapear dependências entre os notebooks (qual alimenta qual)
  - [STDBY] Criação análise lineage
    - e.g. Para uma dada tabela, verificar quais runs estão associadas
    - e.g. Para uma dada run, verificar quais tabelas estão associadas
    - e.g. Para cada VX, qual MODE está associada

####################################################################

# Direcionamentos Gerais
- [STDBY] Fazer ingestão completa para desenvolvimento 
- [STDBY] Criar modelos abrindo além de SEG_NOVO/RENOVACAO/MANUAL/DIGITAL
- [ ] Treinar modelos com novas abordagens
  - [ ] Utilizar dados de corretor atualizados 🔵
    - HR, QTDs
    - HR somente
  - [ ] Utilizar classificação, removendo HR, QTDs 🔵
  - [STDBY] Utilizar clusterização

####################################################################

# 1_PRE_PROC
  - [STDBY] Implementar lógica: se não foi atualizada regra pra uma tabela e ela já existe, então não recriá-la
  - [DEPENDENTE] Implementar pré-processamento + regras definidas em análise base

# 2_JOIN
  - OK

# MODE_C
  # 0_INGESTAO
  - OK

  # 1_PRE_PROC
  - OK

  # 2_JOIN
  - OK

  # 3_TREINO
  - [x] Adicionar lineage nos logs

    # ETAPA PRE_PROC_MODEL
    - [x] Definir lista de atributos presentes em silver.cotacao_seg que vão passar pro FS
    - [x] Executar regras/pre-proc
    - [ ] Ajustar json pra acomodar 'rules_feature_prep' 🟠

    # ETAPA FEATURE SELECTION
    - [STDBY] Implementar partição exclusiva ou CV na etapa de FS - explorar aspecto de variância com etapa de treinamento
    - [STDBY] Verificar possibilidade de explorar splits, SEEDs, params, etc
      - [STDBY] Analisar custo e tempo de execução - Mais coerente explorar CV ou SEEDs diferentes?
      - [STDBY] Variar depth/regularização para verificar estabilidade do ranking
    - [STDBY] Métodos paralelos de Análise de Importância
      - [STDBY] SHAP // Fornece sentido de relação (positivo, negativo)
      - [STDBY] Permutation Importance
      - [STDBY] RFE - Recursive Feature Elimination
    - [ ] Ver FEATURE_CANDIDATES que não estão contempladas na tabela 🟠

    # ETAPA TREINO
    - [STDBY] Avaliar implementação Modelos de ranking direto (LambdaMART, LambdaRank)
    - [STDBY] Eleger Top-K features com base em curva de desempenho (AUC-PR x número de features)

    - [STDBY] Entender como trabalhar com calibração do score e influência decisão de threshold
      - Platt Scaling, Isotonic Regression

    - [ ] 🔴 AJUSTE FEITO EM 16/03 (VERIFICAR): Verificar quais treinos do grid tão sendo registrados
    - [ ] 🔴 AJUSTE FEITO EM 16/03 (VERIFICAR): Para etapa de INFERENCIA, adiante
      - Quero ter apenas uma tabela por inferencia. Como terei vários modelos utilizados paralelamente para inferir, terei colunas repetidas, variando o modelo (model_id), como por exemplo
        - p_emitida_model_id
        - rank_global_model_id
        - treino_exec_model_id_run_id (caso esta coluna for específica por model_id)
      Ou seja, minha ideia é ter as colunas repetidas, mantendo cada cotação com apenas uma linha. Assim poderei fazer as análises posteriores diretamente da mesma tabela.
    - [ ] Ajustar 3_TREINO_MODE_C pra não fechar a run antes da etapa de EVAL (deixar tudo na mesma run)


  # 4_INFERENCIA
  - [ ] Ajustar nome tabela de inferência criada (pode ser _mode_code_segmentacao_timestamp) 🔴
  - [ ] Verificar se tabela de inferência contempla os demais modelos do grid 🔴
  - [ ] Ajustar nome runs - T_INF_timestamp (MODE, SEG e VERSAO são logadas como tags/params) 🔴
  
  # 5_COMP
  - [ ] Desenvolver e testar código 🔴

  # 6_REPORT
  - [ ] Referenciar run_id de modelos e gerar comparações