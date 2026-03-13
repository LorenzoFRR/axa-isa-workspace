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
- [ ] Estudar frameworks 🔴
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

####################################################################
- [ ] Fazer ingestão completa para desenvolvimento 🟡
- [ ] Fazer análise base (pré processamento, regras de negócio) - (PRE_PROC, PRE_PROC_MODEL) - [ANALISE_BASE.md] 🔵 📄

# 1_PRE_PROC
  - [STDBY] Implementar lógica: se não foi atualizada regra pra uma tabela e ela já existe, então não recriá-la
  - [DEPENDENTE] Implementar pré-processamento + regras definidas em análise base

# 2_JOIN
  - [ ] 

# MODE_C
  # 0_INGESTAO
  - OK

  # 1_PRE_PROC
  - OK

  # 2_JOIN
  - OK

  # 3_TREINO

    # ETAPA PRE_PROC_MODEL
    - [ ] VER COMO FICOU: Definir lista de atributos presentes em silver.cotacao_seg que vão passar pro FS 🔴
      - Em célula separada, quero dar dum df_seg.columns -> verificar quais colunas existem -> manualmente remover as que não serão enviadas
    - [ ] VER COMO FICOU: Definir regras 🔴
      - [ ] Definir threshold de truncagem por cardinalidade de features (ajustado para 15)
      - [ ] Remoção de features com >90% nulos
      - [ ] Truncagem de alta cardinalidade (>15 categorias → top 10 + OUTROS)
      - [ ] Remoção de colunas constantes
      - [ ] Encoding → Imputer (média) → VectorAssembler
    
    # ETAPA FEATURE SELECTION
    - [STDBY] Implementar partição exclusiva ou CV na etapa de FS - explorar aspecto de variância com etapa de treinamento
    - [STDBY] Verificar possibilidade de explorar splits, SEEDs, params, etc
      - [STDBY] Analisar custo e tempo de execução - Mais coerente explorar CV ou SEEDs diferentes?
      - [STDBY] Variar depth/regularização para verificar estabilidade do ranking
    - [STDBY] Métodos paralelos de Análise de Importância
      - [STDBY] SHAP // Fornece sentido de relação (positivo, negativo)
      - [STDBY] Permutation Importance
      - [STDBY] RFE - Recursive Feature Elimination

    # ETAPA TREINO
    - [STDBY] Avaliar implementação Modelos de ranking direto (LambdaMART, LambdaRank)
    - [STDBY] Eleger Top-K features com base em curva de desempenho (AUC-PR x número de features)

    - [STDBY] Entender como trabalhar com calibração do score e influência decisão de threshold
      - Platt Scaling, Isotonic Regression

  # 4_INFERENCIA
  - [ ] Ajustar nome tabela de inferência criada 🔴
  - [ ] Ajustar nome runs 🔴
  - [ ] Entender Listas de Tipo (MODE_C) 🔴
    - Qual o impacto desta lista e como faço para defini-la de forma mais robusta?
    - Ao invés de redefinir, reutilizar de 3_TREINO_MODE_C
  - [ ] MODEL_IDS ficou só com um model. Acho que o problema é na etapa de 3_TREINO_MODE_C. Verificar 🔴
  
  # 5_COMP
  - [ ] Desenvolver e testar código 🔴
  - [ ] Definir análises
    - [ ] Precision/Recall@K para o grid
    - [ ] TP/FN@K para o grid

  # 6_REPORT
  - [ ] Referenciar run_id de modelos e gerar comparações