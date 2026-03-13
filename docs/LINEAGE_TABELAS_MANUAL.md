# LINEAGE
## 0_INGESTAO
input (isa_bronze ou sFTP):
    - ingestão customizada (cotacao_generico e corretor_resumo/detalhe)
ouput (bronze):
    - cotacao_generico_timestamp
    - corretor_resumo
    - corretor_detalhe


## 1_PRE_PROC
input (bronze):
    - cotacao_generico_timestamp
    - corretor_resumo
    - corretor detalhe
output (silver):
    - cotacao_generico_clean_timestamp
    - corretor_resumo_clean_timestamp
    - corretor detalhe_clean_timestamp

## 2_JOIN
input (silver):
    - cotacao_generico_clean_timestamp
    - corretor_resumo_clean_timestamp
    - corretor detalhe_clean_timestamp
output (silver):
    - cotacao_seg_timestamp

## 3_ML_TREINO
### PRE_PROC_MODEL
    input (silver):
        - cotacao_seg_timestamp
    output (gold):
        - cotacao_model_timestamp
        - cotacao_validacao_timestamp
## 4_INF
input (gold):
    - cotacao_validacao_timestamp
output (gold):
    - cotacao_inferencia_timestamp

----
# TABELAS POR SCHEMA
bronze:
    - cotacao_generico_timestamp
    - corretor_resumo
    - corretor_detalhe

silver:
    - cotacao_generico_clean_timestamp
    - corretor_resumo_clean_timestamp
    - corretor detalhe_clean_timestamp

    - cotacao_seg_timestamp

gold:
    - cotacao_model_timestamp
    - cotacao_validacao_timestamp

    - cotacao_inferencia_timestamp

