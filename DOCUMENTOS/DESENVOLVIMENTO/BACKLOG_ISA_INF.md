# ###################################################################
Ícones prioridade (Alta > Média > Baixa): 🔴 🟠 🟡

OBS: Atualizar repo: https://github.com/LorenzoFRR/axa-isa-workspace/tree/main :
git add .
git commit -m "descrição"
git push

OBS: Materiais de entregas estão no Confluence: https://pswdigital.atlassian.net/wiki/spaces/AI/overview
# ###################################################################
# ###################################################################

# Pendencias sem etapa específica
- Definir lógica cotações incrementais ao longo do pipeline
- Adicionar FLAG modelos aprovados

# I_PR_INGESTAO
- [ ] Verificar lógica de ingestão + mover arquivos DADOS_ANTIGOS

# I_PR_PRE_PROC
- [ ] Implementar lógica   
    - No pipeline de treinamento (ISA_EXP, ISA_DEV) a etapa PRE_PROC pode ser diferente entre os testes. O pipeline de inferência deverá representar essas etapas diversas de PRE_PROC (pode até inserir lógica de puxar as mesmas rules_execution.json de ISA_EXP/PRE_PROC).
    - Basicamente vai existir fluxos paralelos (aqui e em etapas posteriores)
    - Consumir as rules_execution/rules_catalog direto da run PRE_PROC de ISA_EXP (pegar direto da run, ao invés de definir com redundância no notebook)


# I_PR_JOIN

# I_PR_INFERENCIA

# I_PR_MONITORAMENTO
- [ ] Implementar
- [ ] Executar run provisória dados novos (hoje está em ISA_EXP)

# I_PR_DEVOLUCAO
