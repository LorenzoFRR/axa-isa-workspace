# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AXA ISA — Guia de Desenvolvimento

## Instruções para o Claude
- Este documento é a **fonte única de verdade estrutural** do projeto.
- Atualizar este documento quando decisões arquiteturais ou convenções forem estabelecidas ou alteradas.
- **Não registrar aqui**: regras de negócio específicas, colunas, parâmetros de modelo — esses vivem nos notebooks e mudam entre iterações.
- Antes de modificar qualquer notebook, lê-lo para entender o estado atual das regras.

---

## Contexto
Modelo de **ranking/scoring** para priorização de capacidade comercial.
- Classifica cotações por probabilidade de conversão
- **Dual objetivo**: ranker eficaz nos top-K (K = capacidade do time) + classificador geral (distinguir Emitida/Perdida sem restringir ao top-K). **Treinamento e avaliação de performance devem contemplar os dois aspectos** — métricas de ranking (Precision@K, Recall@K, Lift@K) e de classificação geral (AUC-PR, AP).
- Foco no tradeoff precision/recall - encontrar um sistema que balanceie as duas métricas

Dois pipelines previstos:
- **Treino** (atual, ISA_DEV): processa cotações com desfecho conhecido
- **Produção** (futuro): inferência sobre cotações sem desfecho

---

## Pipeline (ISA_DEV/)
Notebooks `.py` em formato Databricks, executados em ordem:

| Notebook | Etapa | MLflow Parent Run |
|---|---|---|
| `0_INGESTAO.py` | Ingestão bronze | `T_PR_INGESTAO` |
| `1_PRE_PROC.py` | Pré-proc + feature eng + label | `T_PR_PRE_PROC` |
| `2_JOIN.py` | Join fato + dimensões → silver | `T_PR_JOIN` |
| `3_TREINO_MODE_*.py` | Treinamento (um notebook por mode) | `T_PR_TREINO` |
| `4_INFERENCIA_MODE_*.py` | Inferência/scoring | `T_PR_INFERENCIA` |
| `5_COMP_MODE_*.py` | Comparação de modelos | `T_PR_COMP` |

**Variantes MODE**: existe um notebook por mode (`MODE_A`, `MODE_B`, `MODE_C`) para `3_TREINO`, `4_INFERENCIA` e `5_COMP`. Cada mode tem sua própria abordagem de pré-processamento e modelagem — não compartilham código entre si.

**MLflow experiment** (único para todo o pipeline):
```
/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP
```

Versões antigas: `ISA_DEV (versoes antigas)/` — consultar só se necessário.

---

## Versionamento de Notebooks
- **Edição no mesmo notebook**: editar o arquivo diretamente, sem renomear.
- **Novo notebook derivado de outro** (quando solicitado pelo usuário): adicionar sufixo numérico no novo (`CODIGO_2.py`). O original não é alterado.
- Não existe CHANGELOG.
- **Padrão de execução**: params são alterados diretamente nas células de Config do notebook a cada execução — não se cria um novo notebook para testar combinações diferentes de parâmetros.

---

## Convenções de Tabelas

**Hierarquia de schemas (medallion):** `bronze` → `silver` → `gold`

**Padrão de nomes:** `<schema>.<tabela>_<TS_EXEC>` onde `TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")`. Tabelas do `3_TREINO` recebem também um `RUN_UUID` (8 hex chars) como sufixo adicional para garantir unicidade por execução.

**Cada execução cria novas tabelas com timestamp** — o input do próximo notebook é preenchido manualmente na célula de Config após a execução anterior.

---

## Convenções MLflow

**Estrutura de runs:**
- **Parent run** (nome fixo, ex: `T_PR_PRE_PROC`): container da etapa — não contém execução direta.
- **Child run** (`nested=True`, nome = timestamp `TS_EXEC`): execução real.
- `PR_RUN_ID_OVERRIDE`: preencher para reutilizar parent run existente sem criar novo container.
- **Referência cruzada entre notebooks**: notebooks downstream referenciam runs de etapas anteriores via variáveis como `TREINO_EXEC_RUN_ID` e `JOIN_EXEC_RUN_ID`, usando `client.download_artifacts()` ou `client.get_run()`. O run_id correto é impresso ao final de cada execução e preenchido manualmente na Config do notebook seguinte.
- Tags obrigatórias nas child runs: `pipeline_tipo`, `etapa`, `run_role`.
- **Não logar** tags de versão de notebook (ex: `ingestao_versao`, `join_versao`).
- Runs fechadas ao final de cada bloco via context manager `with mlflow.start_run(...)`.
- **Seeds e salts devem sempre ser logados** nas runs que os utilizam — são parte do contrato experimental. Regra: qualquer parâmetro que afete particionamento ou aleatoriedade deve aparecer no MLflow params da execução correspondente.

**Profiling:**
Artefato padrão: `profiling/profiling.json`
```json
{
  "input":  { "<tabela>": { "total_linhas": N, "total_colunas": N, "colunas": [...] } },
  "output": { "<tabela>": { "total_linhas": N, "total_colunas": N, "colunas": [...] } }
}
```
Complementar com `profiling/ts_arq_contagem.png` quando aplicável.

---

## Convenções de Regras

**Engine:** `rule_def` + `apply_rules`. Cada regra declarada como `rule_def` com nome, descrição, função, `enabled` e `requires_columns`.
- Em `1_PRE_PROC` e `2_JOIN`: organizado como `RULES_BY_TABLE` (dict keyed por nome de tabela) com toggles em `RULE_TOGGLES`.
- Em `3_TREINO`: organizado como `RULES_BY_BLOCK` (dict keyed por bloco de processamento: `rules_on_df_seg`, `rules_build_base`, `rules_build_df_model`, `rules_build_df_validacao`, `rules_feature_prep`) com toggles em dicts separados por bloco (`TOGGLES_RULES_ON_DF_SEG`, etc.).

**Prefixos por tabela/notebook:**
| Prefixo | Tabela | Notebook |
|---|---|---|
| `GEN_Rnn` | cotacao_generico | 1_PRE_PROC |
| `RES_Rnn` | corretor_resumo | 1_PRE_PROC |
| `DET_Rnn` | corretor_detalhe | 1_PRE_PROC |
| `JOIN_Rnn` | resultado do join | 2_JOIN |
| `SEG_Rnn` | segmentação | 2_JOIN |
| `PP_Rnn` / `BUILD_Rnn` | pré-proc modelo / features | 3_TREINO |

**Princípio:** Centralizar regras no notebook mais upstream possível. Não duplicar entre notebooks.

**Regras de preparação para ML** (remoção de nulos, truncagem de cardinalidade, remoção de constantes, encoding/imputer/assembler) pertencem ao notebook `3_TREINO`, não ao `1_PRE_PROC`. São específicas de modelagem e não devem ser movidas para upstream.

**Pipeline pode operar sem as tabelas de CORRETOR**: novas regras podem ser adicionadas apenas sobre `COTACAO_GENERICO`, sem depender de `corretor_resumo` ou `corretor_detalhe`.

---

## Documentação
| Arquivo | Uso |
|---|---|
| `docs/BACKLOG.md` | Tarefas e pendências — atualizar conforme avanço |
| `docs/ANALISE_BASE.md` | Análise dos dados (paralela ao desenvolvimento) |
| `docs/ARQ_MODELO.md` | Schemas de colunas e anotações para guiar novas abordagens de modelagem |
| `docs/LINEAGE_TABELAS_MANUAL.md` | Lineage de tabelas — mantido manualmente, não editar |
| `docs/REFS_MODELAGEM.md` | Referências técnicas de modelagem — somente leitura |
| `docs/PLANO_MODELAGEM_MODE_C.md` | Plano de modelagem MODE_C — decisões arquiteturais do mode ativo |
| `entregas/PONTUAIS.md` | Entregáveis com prazo |
