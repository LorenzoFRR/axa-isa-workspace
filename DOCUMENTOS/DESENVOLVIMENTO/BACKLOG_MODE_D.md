Backlog para desenvolvimento da versão 11, que adiciona clustering ao fluxo no notebook 3_TREINO_MODE_D. Esta versão, pela natureza do desenvolvimento (adição da etapa de clustering), assume novo MODE_D.

- Re-centralizar BACKLOGs
- Criar visualização pra verificar relações notebooks/fluxos/modes/etc

# T_PR_TREINO
- Ajustar local de log das informações (ver de logar com redundância)
- Ajustar local de configs
    - Mostrar e selecionar lista de colunas a serem alimentadas no treino só após CLUSTERING_FIT
- Remover DT_INICIO_VIGENCIA de silver.cotacao_seg_timestamp
- Executar/comparar treino com mesmas colunas que MODE_C, substituindo por CLF_CORRETOR
- Verificar usabilidade 4_INFERENCIA e 5_COMP para MODE_C e MODE_D
- Padronizar nomes tabelas geradas no pipeline

## T_PRE_PROC_MODEL
- [STDBY] Adicionar visualização de distribições/outliers ->  Criação de rule de limpeza
    - A princípio não fazer limpeza, apenas visualizar normalizado em T_CLUSTERING_EXPLORE

## T_CLUSTERING_EXPLORE
- Logar sillouette e elbow curve no mesmo gráfico
- Revisar lógica estratificação dos atributos dos corretores
- [STDBY] Avaliar novas colunas para inserir na etapa de clustering
- [x] Plotar distribuições para as colunas envolvidas
- Entender leitura sillouette/elbow 🟡
- Entender estratégias de normalização implementadas 🟡

## T_CLUSTERING_FIT
- [x] Célula de decisão: Implementar guia para decisão de seleção de K
- [STDBY] Logar, tipo como em rules DE PRE_PROC_MODEL, ativação controlada de clustering (toggle True, False pra aplicar clustering)
- [x] Plotar visualizações de resultados normalizadas/escala real
- [x] Organizar visualizações nos artefatos

- Logar análises/visualizações clustering + Entender interpretação análises
    - [x] QTD corretor por cluster
    - [x] Heatmaps clusters
    - [x] Distribuições HR vs QTD cotações por corretor
    - [x] Distribuições HR vs QTD produtos por corretor
    - [x] Distribuições QTD cotações por corretor vs QTD produtos por corretor
    - [x] json corretores típicos

- [x] Verificar criação CLF_CORRETOR em df_model e df_validacao.
- [x] Dependendo do toggle de clusterizar por SEG, a tabela de cotacao_seg vem com SEG = todas. Verificar se isso acontece para MODE_C. R: Sim, cotacao_model/validacao vêm com SEG = todas.

## T_FEATURE_SELECTION
- Executar etapa respeitando consistência 🔴

## T_TREINO
- Executar etapa respeitando consistência 🔴
- [STDBY] Inserir variação de parâmetros de clustering em grid de treinamento (T_TREINO)?
- Ajustar número de colunas inseridas no treinamento 🔴

# 4_INFERENCIA
- Executar etapa respeitando consistência 🔴

# 5_COMP
- Executar etapa respeitando consistência 🔴
- [STDBY] Revisar etapa de clustering para implementar lógica de tal modo que os resultados da etapa possam ser referenciados em 5_COMP
- Definir análises executivas que podem ser realizadas a partir do resultado do clustering