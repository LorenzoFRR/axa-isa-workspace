# ###################################################################
Ícones prioridade (Alta > Média > Baixa): 🔴 🟠 🟡
# ###################################################################

🔵 (Implementação tem correspondência em BACKLOG_GERAL)
🟣 (Implementação sprint atual)

# ###################################################################
# ###################################################################

# PENDÊNCIAS SEM ETAPA
- Verificar FLAG modelos aprovados
- Validar horário de execução
- Definir dinâmica de seleção de modelos usados em produção
- Ver como fica regras temporais aplicadas (e.g. DIAS_INICIO_VIGENCIA)
- Atualizar LINEAGE_TABELAS.md
- Verificar se levamos corretor_resumo ao longo do pipeline (Implementar toggle de processamento?)
- Verificar se a lógica de FL_NOVO funciona ao longo do pipeline 🔴

# I_PR_INGESTAO
- Lógicas colunas (constantes entre as tabelas)
    - TS_ARQ -> Data do arquivo na fonte (sFTP)
    - TS_ATUALIZACAO -> Data de processamento de cada linha. Uma vez processado, essa coluna não varia?
    - SOURCE_FILE -> Nome do arquivo na fonte (sFTP)
    - FL_NOVO -> Quando processa uma linha nova (true). No mesmo processamento, o que era true vira false
- Lógicas de processamento
    - Como fica o reflexo do MODE ao longo da primeira execução e próximas?
    - FL_NOVO funcionando?
    - Só modificar MODE de DADOS_ANTIGO -> COTACAO na segunda execução já funciona?
    - Como funciona cálculo de colunas que mudam com o tempo?
    - Como funciona lógica de cotações repetidas?
        - Na etapa de PRE_PROC (bronze -> silver) filtrar apenas a versão mais recente de cada cotação (TS_ARQ) [MAPEADO EM PRE_PROC]
- Processamento cotacao_generico
- Processamento tabelas corretor
    - Criar lógica análoga para FL_NOVO
    - Remover corretor_resumo do fluxo 🔴
    - Verificar dinâmica de disponibilização sFTP e processamento Databricks [KIMURA] 🔴
        - OBS: Tabelas vindo com colunas nulas. Testei processando tabelas de corretor usadas em ISA_DEV (provisório)
            - SRC_CORRETOR_RESUMO = "bronze.corretor_resumo_carga_1703"
            - SRC_CORRETOR_DETALHE = "bronze.corretor_detalhe_carga_1703"
    - corretor_resumo
        - Verificar se a tabela vai ser utilizada downstream 🔴
    - corretor_detalhe
- Ajustes código
    - Mudar FL_NOVO (true/false) para valor booleano (True/False) 🟡

# I_PR_PRE_PROC
- Lógicas de processamento
    - Etapa de PRE_PROC (bronze -> silver) filtrar apenas a versão mais recente de cada cotação (TS_ARQ)
    - Verificar se FL_NOVO tá atualizando corretamente
    - CORRETOR_DETALHE
        - Verificar se mantenho drop de colunas ['SOURCE_FILE', 'TS_ARQ', 'TS_ATUALIZACAO']
    - Criar tabela de DUMP em silver pra armazenar as cotações que passaram por PRE_PROC, mas em execuções anteriores? (Manter clean_inferencia sempre com cotações mais recentes)
- Operações de pré-processamento
    - Verificar como puxar/referenciar regras aplicadas em ISA_DEV/PRE_PROC



- [ ] Implementar lógica   
    - No pipeline de treinamento (ISA_EXP, ISA_DEV) a etapa PRE_PROC pode ser diferente entre os testes. O pipeline de inferência deverá representar essas etapas diversas de PRE_PROC (pode até inserir lógica de puxar as mesmas rules_execution.json de ISA_EXP/PRE_PROC).
    - Basicamente vai existir fluxos paralelos (aqui e em etapas posteriores)
    - Consumir as rules_execution/rules_catalog direto da run PRE_PROC de ISA_EXP (pegar direto da run, ao invés de definir com redundância no notebook)

# I_PR_JOIN

# I_PR_INFERENCIA

# I_PR_MONITORAMENTO
- [ ] Dnir quais análises serão logadas
- [ ] Logar run provisória dados novos (hoje está em ISA_EXP)

# I_PR_DEVOLUCAO
- [ ] Verificar formato de devolução