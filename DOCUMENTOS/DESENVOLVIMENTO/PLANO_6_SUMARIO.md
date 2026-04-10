# Plano de Execução — ISA_DEV/6_SUMARIO.py

## Objetivo

Criar um novo notebook [ISA_DEV/6_SUMARIO.py](../../ISA_DEV/6_SUMARIO.py) que consolida, em uma sub-run `T_SUMARIO_{TS_EXEC}` dentro do parent `T_PR_REPORT`, o **resumo completo de uma linha de execução do pipeline** — onde uma linha de execução é definida pela run de [5_COMP.py](../../ISA_DEV/5_COMP.py) (seg × versão × mode × resultado final).

Todos os itens vêm de `run_ids` informados manualmente na Config. O notebook **não lê tabelas Delta** — apenas puxa params/tags/metrics/artifacts via `MlflowClient`.

- **Fonte de verdade da estrutura de Artifacts:** [../../estrutura_report_0904.md](../../estrutura_report_0904.md)
- **Referência de padrões a reusar:** [../../ISA_DEV/6_REPORT.py](../../ISA_DEV/6_REPORT.py)

---

## Decisões-chave

| Tema | Decisão |
|---|---|
| Config de run_ids | Todas manuais e explícitas (uma linha por etapa upstream) |
| Macro fields visíveis no MLflow | `mode_code`, `seg_target`, `treino_versao` como **params**; `versao_ref` como **tag** |
| Log no MLflow | Apenas Artifacts (exceto os 4 macros acima) |
| Lógica por `MODE_CODE` | Só na pasta `3_TREINO` (MODE_C ≠ MODE_D) |
| Célula de Config | **Uma só**, no topo do notebook |
| Helpers novos | `download_and_relog()`, `log_json()` |
| Pasta `LINEAGE_TABELAS` | Adiada para a última iteração (formato a definir) |

---

## Status das iterações

- [ ] **Iter 0.1** — Sync `ISA_DEV/` local com Databricks (ação do usuário, fora do Claude)
- [x] **Iter 0.2** — Criar este plano (`DOCUMENTOS/DESENVOLVIMENTO/PLANO_6_SUMARIO.md`)
- [ ] **Iter 1** — Skeleton do `6_SUMARIO.py` (Config + helpers + abertura de run `T_SUMARIO` + tags/params macro + `metadata.json`)
- [ ] **Iter 2** — Pasta `0_INGESTAO/`
- [ ] **Iter 3** — Pasta `1_PRE_PROC/`
- [ ] **Iter 4** — Pasta `2_JOIN/`
- [ ] **Iter 5** — Pasta `3_TREINO_MODE_C/` (guarda `if MODE_CODE == "C"`)
- [ ] **Iter 6** — Pasta `3_TREINO_MODE_D/` (guarda `if MODE_CODE == "D"`)
- [ ] **Iter 7** — Pasta `4_INFERENCIA/`
- [ ] **Iter 8** — Pasta `5_COMP/`
- [ ] **Iter 9** — Pasta `LINEAGE_TABELAS/` (formato a definir)

**Regra:** após validar cada iteração no Databricks, marcar o checkbox aqui e commitar junto com a mudança de código.

---

## Itens detalhados por iteração

### Iter 0.1 — Sync `ISA_DEV/` local com Databricks

Ação do usuário (fora do Claude). Baixar/sincronizar os notebooks de produção do Databricks para `ISA_DEV/` local:

- `0_INGESTAO.py`, `1_PRE_PROC.py`, `2_JOIN.py`, `3_TREINO_MODE_C.py`, `3_TREINO_MODE_D.py`, `4_INFERENCIA.py`, `5_COMP.py`, `6_REPORT.py`

**Por quê:** o `6_SUMARIO.py` vai chamar `client.download_artifacts(run_id, "<src_path>", ...)` com paths literais extraídos de [../../estrutura_report_0904.md](../../estrutura_report_0904.md). Se o notebook upstream loga em path ligeiramente diferente, o download falha silenciosamente (`⚠️`) e a pasta fica incompleta.

Se houver divergência entre o notebook sincronizado e o `estrutura_report_0904.md`, atualizar **primeiro** o `estrutura_report_0904.md` (ou o notebook upstream) antes de começar a Iter 1.

Commit sugerido: `sync ISA_DEV local com Databricks`.

---

### Iter 1 — Skeleton

Criar [../../ISA_DEV/6_SUMARIO.py](../../ISA_DEV/6_SUMARIO.py) em formato Databricks notebook. Ainda **sem nenhuma pasta de artifacts upstream** — apenas infraestrutura.

**Células (nesta ordem):**

**1. Célula única de Config** (`%md ## Config` + `COMMAND`) — único ponto onde o usuário edita run_ids.

```python
from datetime import datetime
from zoneinfo import ZoneInfo
import mlflow

# MLflow / Estrutura
EXPERIMENT_NAME    = "/Users/psw.service@pswdigital.com.br/ISA_DEV/ISA_DEV"
PR_REPORT_NAME     = "T_PR_REPORT"
SUMARIO_NAME       = "T_SUMARIO"
MODE_CODE          = "D"                     # <<< AJUSTE "C" ou "D"
SUMARIO_VERSAO     = "V11.0.0"
VERSAO_REF         = SUMARIO_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# Override: preencha para reutilizar o container T_PR_REPORT existente
PR_RUN_ID_OVERRIDE = ""                      # <<< AJUSTE

# INPUTS — run IDs de referência (todas manuais)
INGESTAO_EXEC_RUN_ID         = ""            # <<< AJUSTE
PRE_PROC_EXEC_RUN_ID         = ""            # <<< AJUSTE
JOIN_EXEC_RUN_ID             = ""            # <<< AJUSTE
PRE_PROC_MODEL_EXEC_RUN_ID   = ""            # <<< AJUSTE
CLUSTERING_EXPLORE_RUN_ID    = ""            # <<< AJUSTE (só se MODE_CODE == "D")
CLUSTERING_FIT_RUN_ID        = ""            # <<< AJUSTE (só se MODE_CODE == "D")
FS_EXEC_RUN_ID               = ""            # <<< AJUSTE
TREINO_EXEC_RUN_ID           = ""            # <<< AJUSTE
INFERENCIA_EXEC_RUN_ID       = ""            # <<< AJUSTE
COMP_EXEC_RUN_ID             = ""            # <<< AJUSTE

print(f"✅ CONFIG SUMARIO MODE_{MODE_CODE} carregada")
```

**2. Imports e helpers** (`%md ## Imports e helpers` + `COMMAND`)

Reusar o bloco de [6_REPORT.py:58-69](../../ISA_DEV/6_REPORT.py#L58-L69) (`os`, `json`, `shutil`, `tempfile`, `MlflowClient`, `mlflow_get_or_create_experiment`). Adicionar 2 helpers novos:

```python
def download_and_relog(client, src_run_id, src_artifact_path, dest_artifact_path, tmpdir):
    """Baixa artifact de uma run upstream e re-loga no exec run atual."""
    try:
        local = client.download_artifacts(src_run_id, src_artifact_path, tmpdir)
        mlflow.log_artifact(local, artifact_path=dest_artifact_path)
        print(f"✅ {dest_artifact_path}/{os.path.basename(src_artifact_path)}")
    except Exception as e:
        print(f"⚠️  {src_artifact_path} — {e}")

def log_json(obj, filename, artifact_path, tmpdir):
    """Serializa obj em <tmpdir>/<filename> e loga no artifact_path."""
    fp = os.path.join(tmpdir, filename)
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    mlflow.log_artifact(fp, artifact_path=artifact_path)
    print(f"✅ {artifact_path}/{filename}")
```

**3. Abertura de runs MLflow** (`%md ## Abre runs MLflow` + `COMMAND`)

Espelhar [6_REPORT.py:78-107](../../ISA_DEV/6_REPORT.py#L78-L107), com:

- Exec run name: `run_name_vts("T_SUMARIO")`
- Tags obrigatórias:
  ```python
  mlflow.set_tags({
      "pipeline_tipo": "T",
      "etapa":         "SUMARIO",
      "run_role":      "exec",
      "mode":          MODE_CODE,
      "versao_ref":    VERSAO_REF,
  })
  ```

**4. Carrega `SEG_TARGET` e `TREINO_VERSAO`** (`%md ## Carrega macros` + `COMMAND`)

```python
client = MlflowClient()

SEG_TARGET = None
TREINO_VERSAO = None
try:
    _comp_params = client.get_run(COMP_EXEC_RUN_ID).data.params
    SEG_TARGET = _comp_params.get("seg_target", "N/A")
except Exception as e:
    print(f"⚠️  seg_target não carregado: {e}")

try:
    _treino_params = client.get_run(TREINO_EXEC_RUN_ID).data.params
    TREINO_VERSAO = _treino_params.get("treino_versao", "N/A")
except Exception as e:
    print(f"⚠️  treino_versao não carregado: {e}")
```

**5. Loga params macro + metadata.json** (`%md ## Params macro e metadata` + `COMMAND`)

```python
mlflow.log_params({
    "mode_code":     MODE_CODE,
    "seg_target":    SEG_TARGET,
    "treino_versao": TREINO_VERSAO,
    # Rastreabilidade — run_ids
    "pr_run_id":                  PR_RUN_ID,
    "ingestao_exec_run_id":       INGESTAO_EXEC_RUN_ID,
    "pre_proc_exec_run_id":       PRE_PROC_EXEC_RUN_ID,
    "join_exec_run_id":           JOIN_EXEC_RUN_ID,
    "pre_proc_model_exec_run_id": PRE_PROC_MODEL_EXEC_RUN_ID,
    "clustering_explore_run_id":  CLUSTERING_EXPLORE_RUN_ID,
    "clustering_fit_run_id":      CLUSTERING_FIT_RUN_ID,
    "fs_exec_run_id":             FS_EXEC_RUN_ID,
    "treino_exec_run_id":         TREINO_EXEC_RUN_ID,
    "inferencia_exec_run_id":     INFERENCIA_EXEC_RUN_ID,
    "comp_exec_run_id":           COMP_EXEC_RUN_ID,
})

# metadata.json na raiz dos artifacts
_metadata = {
    "mode_code":      MODE_CODE,
    "seg_target":     SEG_TARGET,
    "treino_versao":  TREINO_VERSAO,
    "versao_ref":     VERSAO_REF,
    "sumario_versao": SUMARIO_VERSAO,
    "ts_exec":        TS_EXEC,
    # + todos os *_RUN_ID
}
with tempfile.TemporaryDirectory() as _td:
    _p = os.path.join(_td, "metadata.json")
    with open(_p, "w") as _f:
        json.dump(_metadata, _f, indent=2, ensure_ascii=False)
    mlflow.log_artifact(_p)
```

**6. Encerramento** (`%md ## Encerramento` + `COMMAND`)

```python
while mlflow.active_run() is not None:
    mlflow.end_run()
print("✅ Runs encerradas")
```

**Validação da Iter 1:** rodar no Databricks → conferir na UI do MLflow que existe `T_SUMARIO_{TS}` dentro de `T_PR_REPORT`, com as 5 tags, os params macro nas colunas, e `metadata.json` na raiz de Artifacts.

---

### Iter 2 — Pasta `0_INGESTAO/`

Fonte: `INGESTAO_EXEC_RUN_ID`. Itens em [../../estrutura_report_0904.md:4-18](../../estrutura_report_0904.md).

Adicionar **uma célula nova** antes do bloco de encerramento, envelopada em `with tempfile.TemporaryDirectory() as tmpdir:`:

- `run_name.json` — `log_json({"run_name": client.get_run(INGESTAO_EXEC_RUN_ID).info.run_name}, "run_name.json", "0_INGESTAO", tmpdir)`
- `infos_processamento/nomes_tabelas.json` — nomes das 3 tabelas bronze (`cotacao_generico`, `corretor_detalhe`, `corretor_resumo`). **Antes de implementar**, abrir [../../ISA_DEV/0_INGESTAO.py](../../ISA_DEV/0_INGESTAO.py) sincronizado e identificar onde esses nomes estão (params `*_FQN`, tags, ou dentro de um JSON artifact).
- `infos_processamento/profiling.json` — `download_and_relog(client, INGESTAO_EXEC_RUN_ID, "profiling/profiling.json", "0_INGESTAO/infos_processamento", tmpdir)`
- `infos_processamento/linhas_por_arquivo.json` — download direto
- `infos_processamento/processados.json` — download direto
- `infos_processamento/metrics.json` — consolida em um único JSON as 6 métricas (`n_move_ok`, `n_move_fail`, `n_linhas_totais`, `n_processados`, `n_candidatos`, `n_arquivos_listados`), via `client.get_run(INGESTAO_EXEC_RUN_ID).data.metrics` + `log_json(...)`

**Validação:** rodar, conferir a pasta `0_INGESTAO/` na UI do MLflow com todos os 6 arquivos/subpastas esperados.

---

### Iter 3 — Pasta `1_PRE_PROC/`

Fonte: `PRE_PROC_EXEC_RUN_ID`. Itens em [../../estrutura_report_0904.md:19-25](../../estrutura_report_0904.md).

- `run_name.json`
- `regras_aplicadas/rules_execution.json` — download direto
- `resultados_processamento/tables_lineage.json` — download direto
- `resultados_processamento/profiling.json` — download de `profiling/profiling.json`

---

### Iter 4 — Pasta `2_JOIN/`

Fonte: `JOIN_EXEC_RUN_ID`. Itens em [../../estrutura_report_0904.md:26-43](../../estrutura_report_0904.md).

- `run_name.json`
- `infos_processamento/metrics.json` — consolida 7 métricas: `n_after_join_1`, `n_after_join_2`, `n_detalhe_in`, `n_gen_in`, `n_resumo_in`, `n_seg_final`, `n_seg_not_null`
- `infos_processamento/transforms_execution.json` — download
- `infos_processamento/tables_lineage.json` — download
- `infos_processamento/status_dist_by_seg.json` — download
- `infos_processamento/seg_counts.json` — download
- `infos_processamento/profiling_seg.json` — download
- `infos_processamento/join_spec.json` — download
- `infos_processamento/profiling.json` — download de `profiling/profiling.json`
- `infos_processamento/analysis/` — **download recursivo** de `analysis/` (pode conter vários `status_by_month_stacked_*`). Usar `client.download_artifacts(JOIN_EXEC_RUN_ID, "analysis", tmpdir)` e percorrer com `os.walk`, re-logando cada arquivo com `mlflow.log_artifact(local_file, artifact_path="2_JOIN/infos_processamento/analysis")`.

---

### Iter 5 — Pasta `3_TREINO_MODE_C/`

Célula inteira envolta em `if MODE_CODE == "C":`, senão `print("⏭️  Pulando 3_TREINO_MODE_C (MODE_CODE != 'C')")`.

Fontes: `PRE_PROC_MODEL_EXEC_RUN_ID`, `FS_EXEC_RUN_ID`, `TREINO_EXEC_RUN_ID`. Estrutura em [../../estrutura_report_0904.md:46-94](../../estrutura_report_0904.md).

Três sub-pastas dentro de `3_TREINO_MODE_C/`:

**`T_PRE_PROC_MODEL/`** (fonte: `PRE_PROC_MODEL_EXEC_RUN_ID`)
- `run_name.json`
- `params_raiz.json` — consolida `mode_code`, `seg_target`
- `regras_aplicadas/rules_execution.json` — download
- `regras_aplicadas/params_regras.json` — consolida `high_card_threshold`, `null_drop_pct`
- `regras_aplicadas/rules_feature_prep.json` — download de `pre_proc_feature/rules_feature_prep.json`
- `regras_aplicadas/rules_feature_prep_catalog.json` — download de `pre_proc_feature/rules_feature_prep_catalog.json`
- `resultados_processamento/metrics.json` — consolida `n_df_model`, `n_df_validacao`
- `resultados_processamento/tables_lineage.json` — download
- `resultados_processamento/profiling_df_validacao.json` — download
- `resultados_processamento/profiling_df_model.json` — download
- `resultados_processamento/cat_cardinality.json` — download de `pre_proc_feature/cat_cardinality.json`
- `resultados_processamento/null_profile.json` — download de `pre_proc_feature/null_profile.json`

**`T_FEATURE_SELECTION/`** (fonte: `FS_EXEC_RUN_ID`)
- `run_name.json`
- `params_raiz.json` — consolida `mode_code`, `seg_target`
- `entradas_feature_selection/features_candidates.json` — consolida params `features_candidates_enabled` + `features_candidates_disabled`
- `configs_feature_selection/fs_configs.json` — consolida params `fs_methods_config` + `fs_seeds`
- `resultados_processamento/table_lineage.json` — download (atenção: em MODE_C o estrutura usa `table_lineage/json` na linha 76, verificar path real)
- `resultados_processamento/features_ranked.json` — download de `summary/features_ranked.json`
- `resultados_processamento/pearson_heatmap.png` — download de `pearson/pearson_heatmap.png`

**`T_TREINO/`** (fonte: `TREINO_EXEC_RUN_ID`)
- `run_name.json`
- `params_raiz.json` — consolida `seg_target`, `mode_code`
- `entradas_modelos/entradas.json` — consolida `feature_cols`, `n_features`
- `configs_modelos/configs.json` — consolida `cv_folds`, `eval_criterion`, `eval_precision_target`, `gbt_param_grid`, `label_rate`, `model_ids`
- `resultados_processamento/tables_lineage.json` — download

---

### Iter 6 — Pasta `3_TREINO_MODE_D/`

Célula inteira envolta em `if MODE_CODE == "D":`. Estrutura em [../../estrutura_report_0904.md:95-156](../../estrutura_report_0904.md).

Fontes adicionais (além de `PRE_PROC_MODEL_EXEC_RUN_ID`, `FS_EXEC_RUN_ID`, `TREINO_EXEC_RUN_ID`): `CLUSTERING_EXPLORE_RUN_ID`, `CLUSTERING_FIT_RUN_ID`.

Cinco sub-pastas:

**`T_PRE_PROC_MODEL/`** — idêntico ao MODE_C.

**`T_CLUSTERING_EXPLORE/`** (fonte: `CLUSTERING_EXPLORE_RUN_ID`)
- `run_name.json`
- `params_raiz.json` — consolida `seg_target`
- `configs_clustering/configs.json` — consolida params `clf_k_range_explore`, `clf_cluster_features`
- `resultados_clustering/` — **download recursivo** de `clustering/` inteiro. Re-logar cada arquivo preservando subpastas: `3_TREINO_MODE_D/T_CLUSTERING_EXPLORE/resultados_clustering/<rel_path>`.

**`T_CLUSTERING_FIT/`** (fonte: `CLUSTERING_FIT_RUN_ID`)
- `run_name.json`
- `definicoes_clustering/definicoes.json` — consolida params `clf_scale_strategy_final`, `clf_k_final`
- `resultados_clustering_fit/` — **download recursivo da raiz de artifacts** dessa run, re-logando tudo dentro de `3_TREINO_MODE_D/T_CLUSTERING_FIT/resultados_clustering_fit/<rel_path>`.

**`T_FS/`** (fonte: `FS_EXEC_RUN_ID`)
- Mesma estrutura que MODE_C (`T_FEATURE_SELECTION/`), mas:
  - Nomes dos params de entrada são `features_candidate_enabled`/`features_candidate_disabled` (singular, conforme estrutura_report_0904.md:133-134)
  - `resultados_processamento/pearson_heatmap.json` (em MODE_C é `.png`) — verificar no notebook real qual é o path correto

**`T_TREINO/`** (fonte: `TREINO_EXEC_RUN_ID`)
- `run_name.json`
- `params_raiz.json` — consolida `mode_code`, `seg_target`
- `entradas_modelos/entradas.json` — consolida `feature_cols`, `n_features`, `n_model`
- `configs_modelos/configs.json` — consolida `cv_folds`, `cv_seed`, `gbt_param_grid`, `label_rate`
- `resultados_processamento/tables_lineage.json` — download

---

### Iter 7 — Pasta `4_INFERENCIA/`

Fonte: `INFERENCIA_EXEC_RUN_ID`. Itens em [../../estrutura_report_0904.md:157-164](../../estrutura_report_0904.md).

- `run_name.json`
- `params.json` — consolida `seg_target`, `models_id_inferred`, `mode_code`, `inf_versao`, `feature_cols`
- `tables_lineage.json` — download direto

---

### Iter 8 — Pasta `5_COMP/`

Fonte: `COMP_EXEC_RUN_ID`. Itens em [../../estrutura_report_0904.md:165-182](../../estrutura_report_0904.md).

**Antes de implementar:** abrir [../../ISA_DEV/5_COMP.py](../../ISA_DEV/5_COMP.py) e confirmar se `versao_ref`, `seg_target`, `mode_code`, `inferencia_table_fqn`, `base_rate`, `n_rows` estão como params, tags ou métricas. Consolidar todos em um único JSON.

- `run_name.json`
- `info_geral.json` (raiz da pasta) — consolida `versao_ref`, `seg_target`, `mode_code`, `inferencia_table_fqn`, `base_rate`, `n_rows`
- `metricas_treinamento/classification/pr_curves.png` — download
- `metricas_treinamento/classification/threshold_metrics.json` — download
- `metricas_treinamento/concordance/overlap_at_k.png` — download
- `metricas_treinamento/concordance/rank_correlation_heatmap.png` — download
- `metricas_treinamento/overfitting/overfitting_comparison.png` — download
- `metricas_ranking/ranking/curves_cm_at_k.png` — download
- `metricas_ranking/ranking/curves_ranking.png` — download
- `metricas_ranking/scores/score_distribution.png` — download

---

### Iter 9 — Pasta `LINEAGE_TABELAS/`

Adiada. Formato a definir no momento da implementação. Hipótese inicial: um único `lineage.json` agregando `*_FQN` de todas as etapas, lido dos params ou dos `tables_lineage.json` de cada run upstream, organizado por camada (bronze/silver/gold).

---

## Convenções aplicadas

- **Estrutura de células:** `Config → Imports/Helpers → Abre runs → Macros → Params/metadata → [Pastas upstream, uma por célula] → Encerramento`
- **Célula única de Config** no topo (diferente de `6_REPORT.py` que espalha config)
- **Tags obrigatórias (CLAUDE.md):** `pipeline_tipo="T"`, `etapa="SUMARIO"`, `run_role="exec"`, `mode`, `versao_ref`
- **Nomenclatura:** `SCREAMING_SNAKE_CASE` para constantes/run_ids
- **Versionamento:** `SUMARIO_VERSAO = "V11.0.0"` (semver)
- **Reuso de [../../ISA_DEV/6_REPORT.py](../../ISA_DEV/6_REPORT.py):** `mlflow_get_or_create_experiment`, `run_name_vts`, padrão de abertura PR+exec, padrão try/except silencioso em downloads

---

## Verificação end-to-end (por iteração)

1. Ajustar os run_ids relevantes na Config.
2. Executar o notebook no Databricks.
3. Conferir na UI do MLflow:
   - A run `T_SUMARIO_{TS}` existe dentro de `T_PR_REPORT`.
   - Tags: `pipeline_tipo=T`, `etapa=SUMARIO`, `run_role=exec`, `mode`, `versao_ref`.
   - Params macro visíveis nas colunas: `mode_code`, `seg_target`, `treino_versao`.
   - A pasta recém-adicionada aparece em Artifacts com todos os itens esperados.
   - Nenhum `⚠️` no log da execução (se houver, investigar se é ausência real ou bug de path).
4. Marcar o checkbox da iteração neste arquivo.
5. Commit da iteração (mudança em `6_SUMARIO.py` + checkbox atualizado aqui = um commit).

---

## Como retomar em uma nova sessão do Claude

1. Pedir: *"Vamos retomar a implementação do 6_SUMARIO — próxima iteração pendente"*.
2. Claude lê este arquivo, localiza o primeiro checkbox `[ ]` desmarcado, lê a seção "Itens detalhados" correspondente, e lê [../../ISA_DEV/6_SUMARIO.py](../../ISA_DEV/6_SUMARIO.py) (a partir da Iter 1) para entender o estado atual do código.
3. Claude adiciona a célula nova da iteração atual **sem alterar** o que já estava funcionando.
4. Claude atualiza o checkbox e sugere o commit.

Isso evita reexplorar todo o contexto a cada sessão.
