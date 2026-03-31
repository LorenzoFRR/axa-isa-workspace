# Pendiencias sem etapa específica
- Definir mapeamento cotações incrementais ao longo do pipeline
- Verificar diagrama Danni que representa o fluxo do pipeline de TREINAMENTO (0 -> 3/PRE_PROC_MODEL)

# sFTP
- Verificar lógica de ingestão + mover arquivos DADOS_ANTIGOS

# 0_INGESTAO_INF
- cotacao_generico
    - Verificar lógica de ingestão incremental

# 1_PRE_PROC_INF
- No pipeline de treinamento (ISA_EXP, ISA_DEV) a etapa PRE_PROC pode ser diferente entre os testes. O pipeline de inferência deverá representar essas etapas diversas de PRE_PROC (pode até inserir lógica de puxar as mesmas rules_execution.json de ISA_EXP/PRE_PROC).
    - Basicamente vai existir fluxos paralelos (aqui e em etapas posteriores)
    - Consumir as rules_execution/rules_catalog direto da run PRE_PROC de ISA_EXP (pegar direto da run, ao invés de definir com redundância no notebook)

# 2_JOIN

# 3_TREINO
