# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AXA ISA — Guia de Desenvolvimento

## Instruções para o Claude
- Este documento é a **fonte única de verdade estrutural** do projeto.
- Atualizar este documento quando decisões arquiteturais ou convenções forem estabelecidas ou alteradas — inclusive durante o desenvolvimento do ISA_INF, à medida que os padrões forem sendo definidos.
- **Não antecipar convenções**: registrar apenas o que foi explicitamente decidido. Seções referentes a pipelines em construção (ex: ISA_INF) devem permanecer neutras até que as convenções sejam estabelecidas com o usuário.
- **Não registrar aqui**: regras de negócio específicas, colunas, parâmetros de modelo — esses vivem nos notebooks e mudam entre iterações.
- Antes de modificar qualquer notebook, lê-lo para entender o estado atual das regras.

---

## Contexto
Modelo de **ranking/scoring** para priorização de capacidade comercial.
- Classifica cotações por probabilidade de conversão
- **Dual objetivo**: ranker eficaz nos top-K (K = capacidade do time) + classificador geral (distinguir Emitida/Perdida sem restringir ao top-K). **Treinamento e avaliação de performance devem contemplar os dois aspectos** — métricas de ranking (Precision@K, Recall@K, Lift@K) e de classificação geral (AUC-PR, AP).
- Foco no tradeoff precision/recall - encontrar um sistema que balanceie as duas métricas

Dois pipelines ativos:
- **Treino** (ISA_DEV): processa cotações com desfecho conhecido — treina e avalia modelos
- **Produção** (ISA_INF): inferência sobre cotações em aberto (sem desfecho) — aplica modelos treinados no ISA_DEV

**Stack:** Python, PySpark, MLflow, Delta Lake, Databricks Notebooks

---

## Estrutura de Pastas

```
AXA/
├── CLAUDE.md                         # Este documento — guia estrutural do projeto
├── ISA_DEV/                          # Pipeline de treinamento (notebooks .py)
│   ├── 0_INGESTAO.py
│   ├── 1_PRE_PROC.py
│   ├── 2_JOIN.py
│   ├── 3_TREINO_MODE_C.py
│   ├── 3_TREINO_MODE_D.py
│   ├── 4_INFERENCIA.py
│   ├── 5_COMP.py
│   └── 6_REPORT.py
├── ISA_INF/                          # Pipeline de inferência/produção (notebooks .py)
│   ├── * Notebooks a serem desenvolvidos
├── DOCUMENTOS/
│   ├── PADRAO_DESENVOLVIMENTO.md     # Convenções de código e variáveis
│   ├── LINEAGE_TABELAS_MANUAL.md     # Lineage de tabelas — somente leitura
│   ├── REFS_MODELAGEM.md             # Referências técnicas de modelagem
│   └── DESENVOLVIMENTO/              # Backlogs e referências
│       ├── BACKLOG_GERAL.md
│       ├── BACKLOG_ISA_DEV.md
│       └── BACKLOG_ISA_INF.md
├── DIAGRAMAS/
│   ├── PIPELINE_OVERVIEW_MODE_C.drawio
│   ├── PIPELINE_OVERVIEW_MODE_D.drawio
│   ├── PIPELINE_OVERVIEW_ISA_INF.drawio
│   └── FLUXO_DESENVOLVIMENTO.drawio
└── ENTREGAS/                         # Entregáveis (PDFs, apresentações)
```

---

## Pipeline (ISA_DEV/) — Treinamento
Notebooks `.py` em formato Databricks, executados em ordem:

| Notebook | Etapa | MLflow Parent Run |
|---|---|---|
| `0_INGESTAO.py` | Ingestão bronze | `T_PR_INGESTAO` |
| `1_PRE_PROC.py` | Pré-proc + feature eng + label | `T_PR_PRE_PROC` |
| `2_JOIN.py` | Join fato + dimensões → silver | `T_PR_JOIN` |
| `3_TREINO_MODE_C.py` | Treinamento (Mode C) | `T_PR_TREINO` |
| `3_TREINO_MODE_D.py` | Treinamento (Mode D — com clustering) | `T_PR_TREINO` |
| `4_INFERENCIA.py` | Inferência/scoring (mode-agnostic) | `T_PR_INFERENCIA` |
| `5_COMP.py` | Comparação de modelos (mode-agnostic) | `T_PR_COMP` |
| `6_REPORT.py` | Report consolidado de resultados | `T_PR_REPORT` |

**Modes:** Notebooks `3_TREINO` são específicos por mode (um arquivo por mode). Notebooks `4_INFERENCIA`, `5_COMP` e `6_REPORT` são mode-agnostic — o mode é selecionado via `MODE_CODE` na célula de Config.

**Fluxo de dados:**
```
0_INGESTAO → bronze.cotacao_generico_{TS}
1_PRE_PROC → silver.cotacao_generico_clean_{TS}
2_JOIN     → silver.cotacao_seg_{TS}
3_TREINO   → gold.cotacao_model_{mode}_{TS}_{UUID}, gold.cotacao_validacao_{mode}_{TS}_{UUID}
4_INFERENCIA → gold.cotacao_inferencia_mode_{X}_{SEG_SLUG}_{TS}
5_COMP     → Análise e comparação (sem output de tabela)
6_REPORT   → Artefatos consolidados de report (logados no MLflow)
```

**MLflow experiment** (único para todo o pipeline):
```
/Users/psw.service@pswdigital.com.br/ISA_DEV/ISA_DEV
```

### Modes de Treinamento

**MODE_C** — Treinamento direto (sem clustering). Hierarquia MLflow flat: parent → child exec run.

**MODE_D** — Treinamento com clustering de corretores. Hierarquia MLflow com step-level runs:
```
T_PR_TREINO (parent)
└── T_MODE_D (mode container)
    ├── T_PRE_PROC_MODEL_{TS}     — pré-proc específico de modelo
    ├── T_CLUSTERING_EXPLORE_{TS} — exploração de algoritmos de clustering
    ├── T_CLUSTERING_FIT_{TS}     — ajuste final de clustering
    ├── T_FS_{TS}                 — feature selection
    └── T_TREINO_{TS}            — treinamento do modelo final
```
Cada step tem seu próprio `STEP_*_RUN_ID_OVERRIDE` na Config.

---

## Pipeline (ISA_INF/)

Pipeline de inferência em produção — será construído a partir de ISA_DEV, adaptado para cotações em aberto (sem label). Construído etapa a etapa a partir dos notebooks do ISA_DEV, com as devidas alterações.

Todas as convenções/padrões de códigos serão atualizadas neste documento (CLAUDE.md) conforme o projeto avança.

---

## Versionamento de Notebooks
- **Edição no mesmo notebook**: editar o arquivo diretamente, sem renomear.
- **Novo notebook derivado de outro** (quando solicitado pelo usuário): adicionar sufixo numérico no novo (`CODIGO_2.py`). O original não é alterado.
- Não existe documento de versionamento.
- **Padrão de execução**: params são alterados diretamente nas células de Config do notebook a cada execução — não se cria um novo notebook para testar combinações diferentes de parâmetros.
- **Versionamento semântico** nos notebooks de treino/inferência/comparação (ex: V11.0.0).

---

## Convenções de Tabelas

**Hierarquia de schemas (medallion):** `bronze` → `silver` → `gold`

### ISA_DEV (tabelas com timestamp)
**Padrão de nomes:** `<schema>.<tabela>_<TS_EXEC>` onde `TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")`. Tabelas do `3_TREINO` recebem também um `RUN_UUID` (8 hex chars) como sufixo adicional para garantir unicidade por execução.

**Cada execução cria novas tabelas com timestamp** — o input do próximo notebook é preenchido manualmente na célula de Config após a execução anterior.

---

## Convenções MLflow

**Estrutura de runs:**
- **Parent run** (nome fixo, ex: `T_PR_PRE_PROC`): container da etapa — não contém execução direta.
- **Child run** (`nested=True`, nome = timestamp `TS_EXEC`): execução real.
- `PR_RUN_ID_OVERRIDE`: preencher para reutilizar parent run existente sem criar novo container.
- **Referência cruzada entre notebooks**: notebooks downstream referenciam runs de etapas anteriores via variáveis como `TREINO_EXEC_RUN_ID` e `JOIN_EXEC_RUN_ID`, usando `client.download_artifacts()` ou `client.get_run()`. O run_id correto é impresso ao final de cada execução e preenchido manualmente na Config do notebook seguinte.
- Tags obrigatórias nas child runs do ISA_DEV: `pipeline_tipo` (`"T"`), `etapa`, `run_role`. Convenções equivalentes para ISA_INF serão definidas durante seu desenvolvimento.
- **Não logar** tags de versão de notebook (ex: `ingestao_versao`, `join_versao`).
- Runs fechadas ao final de cada bloco via context manager `with mlflow.start_run(...)`.
- **Seeds e salts devem sempre ser logados** nas runs que os utilizam — são parte do contrato experimental. Regra: qualquer parâmetro que afete particionamento ou aleatoriedade deve aparecer no MLflow params da execução correspondente.

---

**Princípio:** Centralizar regras no notebook mais upstream possível. Não duplicar entre notebooks.

**Regras de preparação para ML** (remoção de nulos, truncagem de cardinalidade, remoção de constantes, encoding/imputer/assembler) pertencem ao notebook `3_TREINO`, não ao `1_PRE_PROC`. São específicas de modelagem e não devem ser movidas para upstream.

**Pipeline pode operar sem as tabelas de CORRETOR**: novas regras podem ser adicionadas apenas sobre `COTACAO_GENERICO`, sem depender de `corretor_resumo` ou `corretor_detalhe`.

---

## Convenções de Código

**Estrutura dos notebooks:** `Imports → Configs → Helpers → Lógica de negócio/regras → Execução MLflow`

**Nomenclatura de variáveis** — detalhes completos em `DOCUMENTOS/PADRAO_DESENVOLVIMENTO.md`:
- Constantes/Config: `SCREAMING_SNAKE_CASE` (`EXPERIMENT_NAME`, `TS_EXEC`, `MODE_CODE`)
- Referência a tabelas: sufixo `*_FQN` (`BRONZE_FACT_FQN`, `INPUT_TABLE_FQN`)
- Colunas de referência: sufixo `*_COL` (`STATUS_COL`, `LABEL_COL`)
- DataFrames: prefixo `df_` (`df_fact`, `df_seg`, `df_model_scores`)
- Funções de regra: `PREFIXO_Rnn_descricao()` (`GEN_R01_data_cotacao()`)
- Imports PySpark: `functions as F`, `types as T`

**Funções utilitárias compartilhadas** (duplicadas entre notebooks — candidatas para extração futura):
`ensure_schema()`, `table_exists()`, `assert_table_exists()`, `mlflow_get_or_create_experiment()`, `safe_drop_cols()`, `rule_def()`, `apply_rules_*()`, `profile_basic()`

---

## Documentação
| Arquivo | Uso |
|---|---|
| `DOCUMENTOS/PADRAO_DESENVOLVIMENTO.md` | Convenções de código, variáveis e padrões recorrentes |
| `DOCUMENTOS/LINEAGE_TABELAS_MANUAL.md` | Lineage de tabelas — mantido manualmente, não editar |
| `DOCUMENTOS/REFS_MODELAGEM.md` | Referências técnicas de modelagem — somente leitura |
| `DOCUMENTOS/DESENVOLVIMENTO/BACKLOG_GERAL.md` | Pendências gerais |
| `DOCUMENTOS/DESENVOLVIMENTO/BACKLOG_ISA_DEV.md` | Pendências por etapa do pipeline de treino |
| `DOCUMENTOS/DESENVOLVIMENTO/BACKLOG_ISA_INF.md` | Pendências por etapa do pipeline de inferência |
| `DIAGRAMAS/` | Pasta com diagramas/fluxogramas que representam implementação de forma visual |
