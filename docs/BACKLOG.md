####################################################################
Backlog — Ideias, Pontos Pendentes, Anotações, Dúvidas

Organizado por categoria.
Atualizar conforme novos pontos surgirem nas sessões de trabalho.

Ícones prioridade (Alta > Média > Baixa): 🔴 🟠 🟡
Ícone em andamento: 🔵
Ícone documento específico: 📄

OBS: Atualizar repo:
git add .
git commit -m "descrição curta"
git push

####################################################################

# Integrações / Otimização desenvolvimento
- [STDBY] Integrar Claude Code diretamente ao Databricks (aguardando permissões)
- [STDBY] Automatizar exportação/importação de notebooks via Databricks CLI ou API
- [STDBY] Criar repositório Github e fornecer acesso para consulta do Workspace Claude
- [STDBY] Criar 'agente' contextualizado para tarefas de desenvolvimento de ML Pipelines
- [STDBY] Verificar integração Git
- [ ] Compartilhamento do workspace Claude 🟡
- [ ] Estudar frameworks
  - XP Pair Programming
  - Diagrama Entidade Relacionamento
  - Arquitetura SDD
- [ ] Ver integração Claude Code/Genie Code no ambiente

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

# Pastas/Workspace
- [x] Fazer upload dos notebooks novamente

####################################################################
- [ ] Fazer ingestão completa para desenvolvimento 🟡
- [ ] Fazer análise base (pré processamento, regras de negócio) - (PRE_PROC, PRE_PROC_MODEL) - [ANALISE_BASE.md] 🔵 📄

# 1_PRE_PROC
  - [STDBY] Implementar lógica: se não foi atualizada regra pra uma tabela e ela já existe, então não recriá-la
  - [DEPENDENTE] Implementar pré-processamento + regras definidas em análise base

# 2_JOIN
  - [ ] 

# MODE_C - Revisar arquitetura modelo - [ARQ_MODELO.md] 🔴 📄
  # 0_INGESTAO
  - OK

  # 1_PRE_PROC
  - [ ] Definir regras

  # 2_JOIN
  - OK

  # 3_TREINO
    - [ ] Ver como vai ser trabalhada a questão das seeds ao longo das etapas e das runs (splits model e validacao, FS, TREINO)
      - [ ] Entender o que é SPLIT_SALT

    # ETAPA PRE_PROC_MODEL
    - [ ] Definir lista de atributos presentes em silver.cotacao_seg que vão passar pro FS
      - Em célula separada, quero dar dum df_seg.columns -> verificar quais colunas existem -> manualmente remover as que não serão enviadas
    - [ ] Definir regras 🔴
      - [ ] Definir threshold de truncagem por cardinalidade de features (ajustado para 15)
      - [ ] Remoção de features com >90% nulos
      - [ ] Truncagem de alta cardinalidade (>15 categorias → top 10 + OUTROS)
      - [ ] Remoção de colunas constantes
      - [ ] Encoding → Imputer (média) → VectorAssembler
    - [ ] Logar lineage 🔴
    
    # ETAPA FEATURE SELECTION
    - [STDBY] Implementar partição exclusiva ou CV na etapa de FS - explorar aspecto de variância com etapa de treinamento
    - [STDBY] Verificar possibilidade de explorar splits, SEEDs, params, etc
      - [STDBY] Analisar custo e tempo de execução - Mais coerente explorar CV ou SEEDs diferentes?
      - [STDBY] Variar depth/regularização para verificar estabilidade do ranking
    - [STDBY] Métodos paralelos de Análise de Importância
      - [STDBY] SHAP // Fornece sentido de relação (positivo, negativo)
      - [STDBY] Permutation Importance
      - [STDBY] RFE - Recursive Feature Elimination

    - [ ] Definir implementação/mecanismo dos algortimos de FS 🔴
    - [ ] Definir regra ensemble para seleção de variáveis pós análise importância 🔴

    # ETAPA TREINO
    - [STDBY] Avaliar implementação Modelos de ranking direto (LambdaMART, LambdaRank)
    - [STDBY] Eleger Top-K features com base em curva de desempenho (AUC-PR x número de features)

    - [ ] Definir implementação/mecanismo do treino do algoritmo em TREINO
      - [ ] Esclarecer definição de threshold operacional/capacidade/conversão
      - [ ] Entender como trabalhar com calibração do score e influência decisão de threshold
        - Platt Scaling, Isotonic Regression
    - [ ] Basear decisão de modelo ótimo no CV/Hiperparams em métricas adequadas
      - [ ] Quais as formas de analisar performance, quais pontos de vista? [Relaciona-com-5_COMP]
      - [ ] Basear em precision,recall,FN,TP@K?
      - [ ] Como analisar relação precision/recall para definir melhor modelo?
      - [ ] Entender como entra threshold/capacidade/etc na escolha do melhor modelo pós CV/hiperparms tuning

  # 4_INFERENCIA
  - [ ] Refatorar código para não precisar re-treinar modelo 🔴
  - [ ] Ajustar nome tabela de inferência criada 🔴
  - [ ] Ajustar nome runs 🔴
  - [ ] Ajustar regra de salvar PP - deu erro 🔴

  # 5_COMP
  - [ ] Definir análises
    - [ ] Precision/Recall@K para o grid
    - [ ] TP/FN@K para o grid

  # 6_REPORT
  - [ ] Referenciar run_id de modelos e gerar comparações - [Análises-definidas-em-3_TREINO/5_COMP]