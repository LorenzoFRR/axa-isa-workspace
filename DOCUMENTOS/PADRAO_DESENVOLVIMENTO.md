# Panorama de Padroes de Desenvolvimento — AXA ISA Workspace

**Data:** 2026-03-25 | **Projeto:** Pipeline ML de scoring/ranking de cotacoes (Databricks/PySpark)
**Stack:** Python, PySpark, MLflow, Delta Lake, Databricks Notebooks

---

## O QUE TEMOS (padroes existentes)

| Categoria | Detalhes |
|---|---|
| **Arquitetura de dados** | Medallion (bronze → silver → gold) com naming convention `<schema>.<tabela>_<TS_EXEC>` |
| **Experiment tracking** | MLflow robusto — hierarquia parent/child runs, params, metricas, artifacts, tags, profiling, lineage JSON |
| **Pipeline sequencial** | 6 notebooks bem definidos (0→5), cada um com I/O documentado e etapa clara |
| **Rule engine** | Sistema declarativo `rule_def` + `apply_rules` com toggles habilitaveis/desabilitaveis por regra |
| **Feature management** | Dict `FEATURE_CANDIDATES` com toggles True/False por feature, inferencia automatica de tipo |
| **Versionamento semantico** | VX.Y.Z nos notebooks de treino/inferencia/comparacao (atualmente V10.0.0) |
| **Secrets** | `dbutils.secrets.get()` para credenciais SFTP — nao expostos em codigo |
| **Documentacao** | CLAUDE.md (guia estrutural), BACKLOG.md, LINEAGE_TABELAS_MANUAL.md, REFS_MODELAGEM.md, ANALISE_V11.md |
| **Diagramas** | draw.io com visao de pipeline e fluxo de desenvolvimento |
| **Reprodutibilidade** | Seeds, salts e parametros logados no MLflow; referencia cruzada entre runs via run_id |
| **Git** | Repositorio provisório GitHub com remote configurado, `.gitignore` presente |

---

## PADROES DE CODIGO E VARIAVEIS

### Estrutura dos Notebooks

Todos os notebooks seguem a mesma organizacao sequencial de celulas:

```
Imports → Cria schema → Configs → Helpers → Logica de negocio/regras → Execucao MLflow
```

- Formato Databricks: `# COMMAND ----------` como separadores de celula
- Markdown via `# MAGIC %md` para titulos de secao
- Cada celula de Config agrupa parametros com blocos `# =========================`
- Parametros que exigem ajuste manual marcados com `<<< AJUSTE`
- Checkpoints visuais com `print("✅ ...")` ao final de cada secao carregada

### Convencoes de Nomenclatura — Variaveis

| Tipo | Padrao | Exemplos |
|---|---|---|
| **Constantes/Config** | `SCREAMING_SNAKE_CASE` | `EXPERIMENT_NAME`, `TARGET_SCHEMA`, `TS_EXEC`, `MODE_CODE`, `WRITE_MODE` |
| **Referencia a tabelas (FQN)** | `*_FQN` sufixo | `BRONZE_FACT_FQN`, `SILVER_FACT_FQN`, `INPUT_TABLE_FQN`, `OUTPUT_TABLE_FQN` |
| **Colunas de referencia** | `*_COL` sufixo | `STATUS_COL`, `LABEL_COL`, `ID_COL`, `SEG_COL`, `DATE_COL` |
| **DataFrames** | `df_` prefixo, snake_case | `df_fact`, `df_seg`, `df_inf_prep`, `df_wide`, `df_model_scores` |
| **Funcoes helper** | `snake_case` | `ensure_schema()`, `table_exists()`, `safe_drop_cols()`, `profile_basic()` |
| **Funcoes de regra** | `PREFIXO_Rnn_descricao()` | `GEN_R01_data_cotacao()`, `PP_R01_normaliza_status()` |
| **IDs de regra** | `PREFIXO_Rnn` | `GEN_R01`, `JOIN_R02`, `PP_R03`, `BUILD_R01`, `SEG_R01` |
| **Toggles de regra** | `TOGGLES_RULES_*` (dict) | `TOGGLES_RULES_ON_DF_SEG`, `TOGGLES_RULES_BUILD_BASE` |
| **Toggles de features** | `FEATURE_CANDIDATES` (dict) | `{"VL_PREMIO_ALVO": True, "DS_SISTEMA": False}` |
| **Runs MLflow (parent)** | `T_PR_<ETAPA>` | `T_PR_INGESTAO`, `T_PR_PRE_PROC`, `T_PR_TREINO` |
| **Runs MLflow (child)** | `T_<ETAPA>_{TS_EXEC}` | `T_INGESTAO_20260318_110703`, `T_INF_20260319_195931` |
| **Override de run pai** | `PR_*_RUN_ID_OVERRIDE` | `PR_RUN_ID_OVERRIDE`, `PR_INF_RUN_ID_OVERRIDE` |
| **Timestamp de execucao** | `TS_EXEC` | `datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")` |
| **UUID de execucao** | `RUN_UUID` | `uuid.uuid4().hex[:8]` (apenas no 3_TREINO) |

### Convencoes de Nomenclatura — Tabelas

| Schema | Padrao | Exemplo |
|---|---|---|
| `bronze` | `<entidade>_{TS_EXEC}` | `cotacao_generico_20260318_110703` |
| `silver` | `<entidade>_clean_{TS_EXEC}` | `cotacao_generico_clean_20260318_110703` |
| `silver` | `<entidade>_seg_{TS_EXEC}` (pos-join) | `cotacao_seg_20260323_102048` |
| `gold` | `<entidade>_{TS_EXEC}_{RUN_UUID}` | `cotacao_model_20260319_195931_cdc3278a` |
| `gold` | `<entidade>_mode_{X}_{SEG_SLUG}_{TS_EXEC}` (inferencia) | `cotacao_inferencia_mode_c_seguro_novo_manual_20260317_144656` |

Tabelas auxiliares (corretor_resumo, corretor_detalhe) nao recebem timestamp — nome fixo por carga.

### Prefixos de Regras por Notebook

| Prefixo | Contexto | Notebook |
|---|---|---|
| `GEN_Rnn` | Regras sobre cotacao_generico | `1_PRE_PROC.py` |
| `RES_Rnn` | Regras sobre corretor_resumo | `1_PRE_PROC.py` |
| `DET_Rnn` | Regras sobre corretor_detalhe | `1_PRE_PROC.py` |
| `JOIN_Rnn` | Regras de join fato + dimensoes | `2_JOIN.py` |
| `SEG_Rnn` | Regras de segmentacao | `2_JOIN.py` |
| `PP_Rnn` | Pre-processamento para ML | `3_TREINO_MODE_C.py` |
| `BUILD_Rnn` | Construcao de splits e features | `3_TREINO_MODE_C.py` |
| `MODEL_Rnn` | Filtragem df_model | `3_TREINO_MODE_C.py` |
| `VALID_Rnn` | Filtragem df_validacao | `3_TREINO_MODE_C.py` |

### Padroes de Codigo Recorrentes

**1. Rule Engine (declarativo + executor)**
```python
# Declaracao
rule_def("GEN_R01", "descricao", fn_regra, enabled=True, requires_columns=["COL"])

# Execucao
df_out, exec_log = apply_rules_for_table("table_key", df, rules, ENABLE_RULES, RULE_TOGGLES)
```

**2. MLflow Parent/Child com Override**
```python
pr_ctx = (
    mlflow.start_run(run_id=PR_RUN_ID_OVERRIDE)
    if PR_RUN_ID_OVERRIDE
    else mlflow.start_run(run_name=PARENT_RUN_NAME)
)
with pr_ctx as pr:
    with mlflow.start_run(run_name=f"T_ETAPA_{TS_EXEC}", nested=True) as cr:
        mlflow.set_tag("run_role", "child")
        # ... execucao ...
```

**3. Profiling como artifact**
```python
profile = profile_basic(df, "nome_tabela", key_cols=["ID_COL"])
mlflow.log_dict(profile, "profiling/profiling.json")
```

**4. Lineage como artifact**
```python
tables_lineage = {"stage": "ETAPA", "inputs": {...}, "outputs": {...}}
mlflow.log_dict(tables_lineage, "tables_lineage.json")
```

**5. Validacao de pre-condicoes**
```python
assert_table_exists(INPUT_TABLE_FQN)
if table_exists(OUTPUT_TABLE_FQN):
    raise ValueError(f"Output ja existe: {OUTPUT_TABLE_FQN}")
```

**6. Encerramento seguro de runs**
```python
while mlflow.active_run() is not None:
    mlflow.end_run()
```

### Imports Padronizados

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F      # sempre alias F
from pyspark.sql import types as T          # sempre alias T
from pyspark.sql.window import Window

import mlflow
from mlflow.tracking import MlflowClient

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Callable, Dict, List, Optional, Tuple
```

### Funcoes Utilitarias Compartilhadas (duplicadas entre notebooks)

As seguintes funcoes aparecem em 3+ notebooks com implementacao identica ou quase identica:

| Funcao | Notebooks onde aparece |
|---|---|
| `ensure_schema()` | 0, 1, 2, 3, 4 |
| `table_exists()` | 0, 1, 2, 3, 4 |
| `assert_table_exists()` | 1, 2, 3, 4 |
| `mlflow_get_or_create_experiment()` | 0, 1, 2, 3, 4, 5 |
| `safe_drop_cols()` | 1, 2, 3 |
| `rule_def()` | 1, 2, 3 |
| `apply_rules_*()` | 1, 2, 3 |
| `profile_basic()` | 2, 3, 4 |

Candidatas naturais para extracao em modulo compartilhado (`utils.py`).

---

## O QUE FALTA (lacunas identificadas)

### Prioridade Alta

| Lacuna | Impacto | Recomendacao |
|---|---|---|
| **Sem testes automatizados** | Nenhum unit/integration test. Regressoes passam despercebidas | Criar testes para funcoes utilitarias e regras (pytest) |
| **Sem CI/CD** | Nenhum GitHub Actions ou pipeline automatizado. Deploys manuais | Minimo: lint + testes no push. Ideal: validacao de notebooks |
| **Sem orquestracao** | Notebooks executados manualmente, params preenchidos a mao entre etapas | Databricks Workflows ou Jobs encadeados |
| **Codigo duplicado entre notebooks** | `ensure_schema`, `table_exists`, `mlflow_get_or_create_experiment`, `profile_basic` replicados em 3+ notebooks | Extrair para modulo compartilhado (`utils.py` ou wheel) |
| **Sem gerenciamento de dependencias** | Nenhum `requirements.txt`, `pyproject.toml` ou lock file | Documentar dependencias com versoes pinadas |
| **Commits genericos** | 100% dos commits sao "atualizacao" / "atualizacoes" | Adotar Conventional Commits (`feat:`, `fix:`, `docs:`) |
| **Token PAT exposto no git remote** | Credencial em plaintext na config do git | Revogar imediatamente, migrar para SSH ou credential helper |

### Prioridade Media

| Lacuna | Impacto | Recomendacao |
|---|---|---|
| **Sem branch strategy** | Apenas branch `main`, sem feature branches ou PRs | Git Flow simplificado: `main` + feature branches + PRs com review |
| **Sem linting/formatting** | Sem Black, Ruff, Flake8 — inconsistencia de estilo | Configurar Ruff ou Black + isort com pre-commit |
| **Sem validacao de dados automatizada** | Sem gates entre etapas (null checks, schema validation, row count assertions) | Great Expectations ou validacoes minimas no notebook |
| **Error handling inconsistente** | Bare `except:` sem tipo em multiplos notebooks | Capturar excecoes especificas, logging estruturado |
| **Sem logging estruturado** | Apenas `print()` para rastreamento — impossivel filtrar/buscar | Usar `logging` module ou integrar com Databricks log analytics |
| **Sem Model Registry** | Modelos nao registrados no MLflow Model Registry — sem staging/prod | Registrar modelos com estagio (Staging → Production) |
| **Configuracao hardcoded** | Params em celulas de notebook, editados manualmente a cada execucao | Externalizar configs (YAML/JSON) ou Databricks Widgets |

### Prioridade Baixa (melhorias futuras)

| Lacuna | Impacto | Recomendacao |
|---|---|---|
| **Sem README no repositorio** | Onboarding dificil para novos devs | README.md com setup, pre-requisitos e fluxo |
| **Sem pre-commit hooks** | Nada impede commits com erros de lint ou secrets | Configurar pre-commit (ruff, detect-secrets) |
| **Lineage apenas manual** | `LINEAGE_TABELAS_MANUAL.md` desatualiza facilmente | Automatizar via MLflow artifacts + script de consolidacao |
| **Sem monitoramento de pipeline** | Sem alertas de falha ou drift | Integrar alertas (Slack/email) em Databricks Jobs |
| **Sem Feature Store** | Features recalculadas a cada execucao | Avaliar Databricks Feature Store para consistencia treino↔inferencia |

---

## METRICAS DE MATURIDADE

| Dimensao | Nivel (1-5) | Observacao |
|---|---|---|
| **Experiment tracking** | 4/5 | Maduro — MLflow bem estruturado, falta Model Registry |
| **Arquitetura de dados** | 4/5 | Medallion clara, lineage documentada, falta automacao |
| **Qualidade de codigo** | 2/5 | Sem testes, lint, ou modularizacao |
| **DevOps/CI-CD** | 1/5 | Inexistente |
| **Versionamento** | 2/5 | Git presente, mas sem branching, PRs, ou commits descritivos |
| **Documentacao** | 3/5 | Boa para o tamanho do projeto, falta README e docs automatizadas |
| **Seguranca** | 2/5 | Secrets ok no codigo, mas PAT exposto no git remote |
| **Operacao/Producao** | 1/5 | Pipeline 100% manual, sem orquestracao nem monitoramento |

---

## RECOMENDACAO DE ROADMAP (ordem sugerida)

1. **Revogar token PAT** do git remote (imediato)
2. **Extrair funcoes compartilhadas** para `utils.py`
3. **Adicionar `requirements.txt`** com dependencias pinadas
4. **Adotar Conventional Commits** + branch strategy minima
5. **Configurar GitHub Actions** com lint basico (Ruff)
6. **Implementar testes** para regras e funcoes utilitarias
7. **Configurar Databricks Workflows** para orquestracao
8. **Registrar modelos** no MLflow Model Registry
