# PLANO_MODELAGEM — MODE_C

Plano conceitual de implementação dos notebooks `3_TREINO_MODE_C`, `4_INFERENCIA_MODE_C` e `5_COMP_MODE_C`.
Referência primária: `NOTAS_MODELAGEM.md`. Referência de implementação: `4_INFERENCIA_MODE_B`, `5_COMP`.

---

## Status de implementação

| Etapa | Status | Observação |
|---|---|---|
| `3_TREINO_MODE_C` | ✅ Implementado e executado | Ver seção abaixo — divergências do plano original documentadas |
| `4_INFERENCIA_MODE_C` | ⚠️ Existe, desatualizado | Config usa listas manuais (`FS_CAT_COLS` etc.); precisa alinhar com 3_TREINO |
| `5_COMP_MODE_C` | ⏳ Pendente | Não existe — a implementar |

---

## Visão geral do fluxo MODE_C

```
3_TREINO_MODE_C
  ├── T_PRE_PROC_MODEL  → df_model (gold), df_validacao (gold)
  ├── T_FEATURE_SELECTION → features rankeadas, top-K sets, MI, Pearson
  ├── T_TREINO          → N modelos salvos (um por combo do grid)
  └── T_EVAL            → avaliação prévia no hold-out por modelo (threshold, P/R/F1, CM)

4_INFERENCIA_MODE_C
  └── Para cada model_id: scoring do df_validacao → tabela unificada

5_COMP_MODE_C
  ├── Overfitting: AUC-PR treino vs val, curvas PR, distribuição de scores
  ├── Desempenho: Precision/Recall/Lift @K, curvas PR por modelo, TP/FP/FN/TN @K
  └── Exploratório (apenas notebook): performance por atributo, análise mensal
```

**Diferenças principais em relação ao MODE_B:**

| Aspecto | MODE_B | MODE_C |
|---|---|---|
| Feature config | Listas manuais (`FS_DECIMAL_COLS` etc.) | `FEATURE_CANDIDATES` dict com toggles; tipo inferido do schema |
| Seleção de vencedor | Automática (melhor avg AUC-PR do CV) | Nenhuma — todos os modelos são salvos |
| Modelos treinados | 1 (melhor combo) | N (um por combinação do grid) |
| Avaliação no hold-out no treino | Sim (métricas @K, CAPACIDADE_PCT) | Sim — T_EVAL com PR curve + threshold selection; métricas @K delegadas ao 5_COMP |
| Feature Selection | LR-L1, RF, GBT + MI | Idem + Pearson entre features numéricas |
| Inferência | Um modelo | N modelos → tabela com `model_id` |
| Comparação | Não aplicável | 5_COMP_MODE_C compara todos os modelos |

---

## 3_TREINO_MODE_C — implementado

### Seção 1 — Configs

Config organizada em **um único bloco** no topo, com sub-blocos comentados por etapa.

#### Sub-bloco: Gerais (topo)

```python
# MLflow
EXPERIMENT_NAME          = "..."
PR_TREINO_NAME           = "T_PR_TREINO"
MODE_CODE                = "C"
MODE_NAME                = f"T_MODE_{MODE_CODE}"

# Overrides
PR_RUN_ID_OVERRIDE        = ""
MODE_RUN_ID_OVERRIDE      = ""
PRE_PROC_RUN_ID_OVERRIDE  = ""
FS_RUN_ID_OVERRIDE        = ""
TREINO_RUN_ID_OVERRIDE    = ""

# Step names
STEP_PRE_PROC_NAME          = "T_PRE_PROC_MODEL"
STEP_FEATURE_SELECTION_NAME = "T_FEATURE_SELECTION"
STEP_TREINO_NAME            = "T_TREINO"

# Versionamento
TREINO_VERSAO            = "V9.0.0"          # versão semântica
TREINO_VERSAO_TABLE_SAFE = TREINO_VERSAO.replace(".", "_")
VERSAO_REF               = TREINO_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_UUID   = uuid.uuid4().hex[:8]
RUN_SUFFIX = TS_EXEC

# I/O
COTACAO_SEG_FQN = "silver.cotacao_seg_..."
OUT_SCHEMA      = "gold"
DF_MODEL_FQN    = f"{OUT_SCHEMA}.cotacao_model_{TS_EXEC}_{RUN_UUID}"
DF_VALID_FQN    = f"{OUT_SCHEMA}.cotacao_validacao_{TS_EXEC}_{RUN_UUID}"
WRITE_MODE      = "overwrite"
```

#### Sub-bloco: PRE_PROC_MODEL / FS / TREINO (compartilhado)

```python
STATUS_COL           = "DS_GRUPO_STATUS"
LABEL_COL            = "label"
ID_COL               = "CD_NUMERO_COTACAO_AXA"
SEG_COL              = "SEG"
DATE_COL             = "DATA_COTACAO"
ALLOWED_FINAL_STATUS = ["Emitida", "Perdida"]
VALID_FRAC           = 0.20
SPLIT_SALT           = "split_c1_seg_mes"   # auditável
SEG_TARGET           = "SEGURO_NOVO_MANUAL"

DO_PROFILE = True

# Seeds (logados em cada run que os usa)
FS_SEEDS      = [42, 123, 7]
FS_TRAIN_FRAC = 0.70
CV_SEED       = 42   # logado no bloco TREINO

# Thresholds de limpeza ML (aplicados no T_PRE_PROC_MODEL)
NULL_DROP_PCT       = 0.90
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"

# Colunas nunca features (excluídas estruturalmente, sem toggle)
COLS_NEVER_FEATURE = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL, STATUS_COL, LABEL_COL, "MES"]

# FEATURE_CANDIDATES: toggle por coluna — True = entra no pipeline, False = bloqueada
# Tipo (cat/num) é inferido do schema automaticamente — não declarado explicitamente.
# Célula de inspeção gera o dict a partir da tabela silver.
FEATURE_CANDIDATES = {
    "VL_PREMIO_ALVO":    True,   # decimal → num
    "DS_PRODUTO_NOME":   True,   # string  → cat
    ...
    "DIAS_COTACAO":      False,  # desabilitado explicitamente
}

FS_METHODS_CONFIG = {
    "lr_l1": {"maxIter": 100, "regParam": 0.01, "elasticNetParam": 1.0},
    "rf":    {"numTrees": 200, "maxDepth": 8},
    "gbt":   {"maxIter": 80,  "maxDepth": 5, "stepSize": 0.1},
}

TOPK_LIST = [5]   # lista de K's para feature sets
```

**Decisão técnica: `FEATURE_CANDIDATES`**
Substitui as listas manuais `FS_DECIMAL_COLS / FS_DIAS_COLS / FS_CAT_COLS`. Tipo inferido de `df.dtypes`: `string` → cat, demais → num. Aliases `FS_DECIMAL_COLS = FS_NUM_COLS` e `FS_DIAS_COLS = []` mantidos internamente para compatibilidade downstream, mas não aparecem na config.

#### Sub-bloco: TREINO (bloco separado, após execução do FS)

```python
TREINO_FEATURE_SET_KEY = "top_5"   # <<< chave do feature set do FS

USE_CLASS_WEIGHT       = "auto"
CLASS_WEIGHT_THRESHOLD = 0.30

CV_FOLDS  = 2
CV_SEED   = 42
CV_METRIC = "areaUnderPR"

GBT_PARAM_GRID = {
    "maxDepth": [4, 6],
    "stepSize": [0.1],
    "maxIter":  100,   # fixo
}

# Critério de seleção de threshold na curva PR (usado no T_EVAL — ver seção abaixo)
EVAL_CRITERION        = "max_f1"          # "max_f1" | "max_f2" | "precision_ge_target"
EVAL_PRECISION_TARGET = 0.4              # usado apenas se EVAL_CRITERION = "precision_ge_target"

ID_COLS            = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]
DROP_FROM_FEATURES = ID_COLS + [STATUS_COL]
TREINO_FEATURE_COLS = FS_FEATURE_SETS[TREINO_FEATURE_SET_KEY]
```

---

### Seção 2 — Imports e Helpers Globais

Inclui: `rule_def`, `apply_rules_block`, `rules_catalog_for_logging`, `profile_basic`, `counts_by_seg`, `label_rate_by_seg`, `safe_drop_cols`, `ensure_schema`, `table_exists`, `assert_table_exists`, `mlflow_get_or_create_experiment`, `compute_top_vals`, `apply_truncation`, `build_tables_lineage_preproc`, `build_tables_lineage_fs`.

---

### Seção 3 — T_PRE_PROC_MODEL

#### Engine de regras (silver → gold)

Prefixos e categorias:
- `PP_`: normalização/filtragem/label sobre dados existentes
- `BUILD_`: criação de colunas auxiliares (`MES`, `is_valid`) — pré-requisitos para o split
- `MODEL_` / `VALID_`: filtros de materialização do split

#### Funções de regra

- `PP_R01_normaliza_status` — normaliza EMITIDA/PERDIDA
- `PP_R02_filtra_status_finais` — mantém apenas Emitida e Perdida (sem converter intermediários — comportamento MODE_C)
- `PP_R03_cria_label` — cria coluna `label` (1.0 / 0.0)
- `BUILD_R01_add_mes` — cria `MES` a partir de `DATE_COL`
- `BUILD_R02_add_split_flag` — split determinístico: `xxhash64(ID, SEG, MES, SPLIT_SALT)` → `is_valid`
- `MODEL_R03_filtra_model`, `VALID_R03_filtra_validacao`, `BUILD_R04_drop_aux`

**Regras de feature prep (PP_R04, PP_R05, PP_R06) — aplicadas em T_PRE_PROC_MODEL:**
- `PP_R04`: remoção de features com `> NULL_DROP_PCT` nulos (calculado sobre df_model)
- `PP_R05`: truncagem de alta cardinalidade — `compute_top_vals` → `apply_truncation` (aplicado em df_model **e** df_valid; `top_vals_by_col_prep` fica em memória Python)
- `PP_R06`: remoção de features constantes (cardinalidade ≤ 1)

Estas três regras são aplicadas antes de salvar df_model e df_valid, dentro do context manager da run PRE_PROC_MODEL. O FS e o TREINO recebem os dados já tratados — não repetem a limpeza.

**Log por regra:** `n_linhas_por_regra_{rule_id}` logado como métrica para `RULES_ON_DF_SEG` (PP_R01..PP_R03). As rules_feature_prep (PP_R04–PP_R06) registram status no `exec_log` mas não têm contagem por linha.

#### Logs da run exec T_PRE_PROC_MODEL

- Tags: `pipeline_tipo`, `stage=TREINO`, `run_role=exec`, `mode`, `step=PRE_PROC_MODEL`, `treino_versao`, `versao_ref`
- Params: `ts_exec`, `treino_versao`, `mode_code`, `seg_target`, `input_cotacao_seg_fqn`, `df_model_fqn`, `df_validacao_fqn`, `valid_frac`, `split_salt`, `allowed_final_status`, `label_col`, `id_col`, `seg_col`, `note_pp_r02`, `pr_run_id`, `mode_run_id`, `t_pre_proc_model_container_run_id`, `null_drop_pct`, `high_card_threshold`, `high_card_top_n`, `outros_label`
- Metrics: `n_seg_in`, `n_linhas_por_regra_{rule_id}` (PP_R01–PP_R03), `n_seg_after_rules`, `n_df_model`, `n_df_validacao`, `df_model_saved`, `df_validacao_saved`
- Artifacts: `rules_catalog.json`, `rules_execution.json`, `tables_lineage.json`, `preproc_feature/null_profile.json`, `preproc_feature/cat_cardinality.json` (contém `top_vals_by_col`), `preproc_feature/rules_feature_prep_catalog.json`, `profiling_df_model.json`, `profiling_df_validacao.json`, `eda_df_model_by_seg.json`

---

### Seção 4 — T_FEATURE_SELECTION

#### Imports e Helpers FS

`truncate_high_cardinality`, `compute_ap`, `get_vector_attr_names`, `base_feature_from_attr`, `importance_to_score`, `log_pandas_csv`. Imports: `matplotlib` (Agg), `numpy`, `pandas`, `sklearn.feature_selection.mutual_info_classif`.

#### Derivação das listas de tipo

No início da execução, derivadas a partir do schema e dos toggles:

```python
FS_CAT_COLS = [c for c in _fc_enabled if _schema_fs[c] == "string"]
FS_NUM_COLS = [c for c in _fc_enabled if _schema_fs[c] != "string"]
FS_DECIMAL_COLS = FS_NUM_COLS   # alias
FS_DIAS_COLS    = []            # alias vazio (folded em NUM)
```

O FS não aplica PP_R04/R05/R06 — eles já foram aplicados no PRE_PROC_MODEL. O FS loga perfis de nulos e cardinalidade como informativos, com nota indicando que os tratamentos já foram feitos.

#### Execução

Passos [1]–[8] iguais à referência MODE_B.

**Passo [9] — Correlação de Pearson (MODE_C)**

Executado após o ensemble, sobre `NUM_COLS_FINAL`. Não influencia a seleção.
- `pearson/pearson_correlation.csv` — formato longo (feature_a, feature_b, correlation)
- `pearson/pearson_heatmap.png` — heatmap anotado com matplotlib (sample 50k)
- `pearson/pearson_config.json`

#### Variáveis produzidas para T_TREINO

`FS_FEATURES_RANKED`, `FS_FEATURE_SETS`, `FS_CAT_COLS_FINAL`, `FS_NUM_COLS_FINAL`

#### Logs da run exec T_FEATURE_SELECTION

- Tags: `pipeline_tipo`, `stage=TREINO`, `run_role=exec`, `mode`, `step=FEATURE_SELECTION`, `treino_versao`, `versao_ref`
- Params: `df_model_fqn`, `seg_target`, `fs_seeds`, `fs_train_frac`, `null_drop_pct`, `high_card_threshold`, `high_card_top_n`, `outros_label`, `fs_methods`, `fs_methods_config`, `topk_list`, `ensemble_type`, `mi_parallel`, `mi_in_ensemble`, `pearson_exploratorio`, `run_suffix`, `ts_exec`, `mode_code`, `pr_run_id`, `mode_run_id`, `fs_container_run_id`, `feature_candidates_enabled`, `feature_candidates_disabled`, `feature_type_cat`, `feature_type_num`
- Metrics: `n_rows_seg`, `n_label_invalid_or_null`, `n_train_seed{s}`, `n_val_seed{s}`, `{method}_seed{s}_ap_val`, `{method}_seed{s}_auc_pr_val`, `{method}_avg_ap_val`, `{method}_avg_auc_pr_val`, `n_features_candidate`
- Artifacts: `rules_feature_prep_catalog.json`, `tables_lineage.json`, `fs_stage1/null_profile.json`, `fs_stage1/cat_cardinality.json`, `fs_stage1/fs_feature_contract.json`, `methods/{method}/seed{s}/importance_by_feature.csv`, `methods/{method}/importance_avg.csv`, `summary/ensemble_weights.json`, `summary/feature_ranking_final.csv`, `summary/features_ranked.json`, `summary/topk_sets.json`, `summary/methods_summary.json`, `mi/mutual_information.csv`, `mi/mutual_information.json`, `pearson/pearson_correlation.csv`, `pearson/pearson_heatmap.png`, `pearson/pearson_config.json`

---

### Seção 5 — T_TREINO

#### Imports e Helpers TREINO

`build_preprocess_pipeline`, `add_class_weights`, `kfold_split`, `build_tables_lineage_treino`. Também imports de helpers de avaliação (ver T_EVAL abaixo).

#### Execução

Loop sobre todas as combinações do `GBT_PARAM_GRID`. Para cada combo: CV + treino final + salvamento de artefatos. Não há seleção de vencedor.

```
# [1] Carga, blank→null em strings, separação cat/num por schema
#     top_vals_by_col = top_vals_by_col_prep (calculado no PRE_PROC_MODEL, reutilizado aqui)
#     add_class_weights → (df_model_seg, label_rate, weight_pos, apply_cw)

# [2] CV n-fold determinístico (kfold_split via xxhash64):
#     Para cada combo do GBT_PARAM_GRID:
#       model_id = f"d{maxDepth}_s{str(stepSize).replace('.','')}"
#       Para cada fold: fit pipeline + fit GBT → evaluate (auc_pr)
#       Logar: cv_{model_id}_fold{i}_auc_pr, cv_{model_id}_avg_auc_pr, cv_{model_id}_std_auc_pr

# [3] Treino final — para CADA combo:
#     pp_final_fit = build_preprocess_pipeline(...).fit(df_model_ml)
#     gbt_model    = GBTClassifier(...params...).fit(pp_final_fit.transform(df_model_ml))
#     mlflow.spark.log_model(gbt_model,    f"treino_final/{model_id}/model")
#     mlflow.spark.log_model(pp_final_fit, f"treino_final/{model_id}/preprocess_pipeline")
#     AUC-PR e AP de treino: avaliado em df_model_ml → auc_pr_treino_{model_id}, ap_treino_{model_id}
#     TRAINED_MODELS[model_id] = {params, cv_avg_auc_pr, cv_std_auc_pr, auc_pr_treino, ap_treino}

# [4] Artifacts de resumo:
#     preprocess/top_vals_by_col.json  ← reutilizado pelo 4_INFERENCIA
#     cv/grid_results.json, cv/fold_metrics.json, cv/trained_models_registry.json
```

---

### Seção 6 — T_EVAL (avaliação prévia no hold-out)

> **Nota vs plano original**: o plano original previa que toda avaliação sobre df_validacao seria delegada ao 5_COMP. A implementação inclui um bloco T_EVAL dentro do 3_TREINO que realiza uma avaliação prévia por modelo sobre o hold-out. O 5_COMP_MODE_C ainda é necessário para a análise comparativa completa (@K, overfitting detalhado, curvas sobrepostas).

#### Helpers T_EVAL

- `compute_pr_curve(pdf, label_col, score_col)` — varre thresholds 0.01–0.99; retorna `{threshold, tp, fp, fn, tn, precision, recall, f1, f2}`
- `select_threshold(pdf_curve, criterion, precision_target)` — retorna linha ótima segundo `EVAL_CRITERION`
- `log_figure(fig, artifact_path)` — salva matplotlib e loga no MLflow
- `plot_pr_curve(pdf_curve, tau_row, baseline, model_id)` — curva PR + ponto τ + baseline
- `plot_threshold_metrics(pdf_curve, tau_row, model_id)` — P/R/F1/F2 por threshold + linha τ
- `plot_confusion_matrix(tau_row, model_id)` — heatmap TP/FP/FN/TN

#### Execução

```
# Para cada model_id em model_id_list:
#   1. Carregar pp e model do MLflow (runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/...)
#   2. Transformar df_eval → df_pred
#   3. Converter probability[1] → p1; toPandas
#   4. compute_pr_curve + select_threshold (EVAL_CRITERION)
#   5. Logar métricas eval_{model_id}_threshold/precision/recall/f1/f2/tp/fp/fn/tn
#   6. Logar artifacts eval/{model_id}/pr_curve.csv, eval_summary.json, pr_curve.png,
#                       threshold_metrics.png, confusion_matrix.png
```

**Decisão técnica: T_EVAL vs 5_COMP**
O T_EVAL fornece uma avaliação rápida de sanidade por modelo (threshold, P/R/F1) sem métricas @K. O 5_COMP_MODE_C é responsável por: comparação entre modelos, análise @K, overfitting (treino vs val), curvas sobrepostas, e seleção do vencedor.

#### Logs completos da run exec T_TREINO

- Tags: `pipeline_tipo`, `stage=TREINO`, `run_role=exec`, `mode`, `step=TREINO`, `treino_versao`, `versao_ref`
- Params: `df_model_fqn`, `df_valid_fqn`, `seg_target`, `feature_set`, `feature_cols`, `n_features`, `n_model`, `use_class_weight`, `class_weight_threshold`, `label_rate`, `apply_cw`, `weight_pos`, `cv_folds`, `cv_seed`, `cv_metric`, `gbt_param_grid`, `gbt_maxiter_fixed`, `model_ids`, `mode_code`, `pr_run_id`, `mode_run_id`, `treino_container_run_id`, `treino_cat_cols`, `treino_num_cols`, `note_mode_c`, `eval_criterion`, `eval_precision_target`
- Metrics:
  - CV: `cv_{model_id}_fold{i}_auc_pr`, `cv_{model_id}_avg_auc_pr`, `cv_{model_id}_std_auc_pr`
  - Treino: `auc_pr_treino_{model_id}`, `ap_treino_{model_id}`
  - Eval (hold-out, T_EVAL): `eval_n_valid`, `eval_baseline`, `eval_{model_id}_threshold`, `eval_{model_id}_precision`, `eval_{model_id}_recall`, `eval_{model_id}_f1`, `eval_{model_id}_f2`, `eval_{model_id}_tp`, `eval_{model_id}_fp`, `eval_{model_id}_fn`, `eval_{model_id}_tn`
- Artifacts:
  - `tables_lineage.json`, `preprocess/top_vals_by_col.json`
  - `cv/grid_results.json`, `cv/fold_metrics.json`, `cv/trained_models_registry.json`
  - `treino_final/{model_id}/model` (um por combo)
  - `treino_final/{model_id}/preprocess_pipeline` (um por combo)
  - `eval/{model_id}/pr_curve.csv`, `eval/{model_id}/eval_summary.json`
  - `eval/{model_id}/pr_curve.png`, `eval/{model_id}/threshold_metrics.png`, `eval/{model_id}/confusion_matrix.png`

---

## 4_INFERENCIA_MODE_C — pendente de atualização

> O notebook existe mas está **desatualizado**: usa listas manuais `FS_CAT_COLS`, `FS_DECIMAL_COLS`, `FS_DIAS_COLS` com valores que podem não corresponder ao que foi efetivamente treinado. Precisa ser alinhado com o 3_TREINO implementado.

### O que precisa mudar

1. **Config de features**: remover `FS_CAT_COLS`, `FS_DECIMAL_COLS`, `FS_DIAS_COLS` manuais. Leitura automática de `feature_cols`, `treino_cat_cols`, `treino_num_cols` do param logado no `TREINO_EXEC_RUN_ID`.
2. **`top_vals_by_col`**: ler de `preprocess/top_vals_by_col.json` do `TREINO_EXEC_RUN_ID` (já está lá na implementação do 3_TREINO) — sem recalcular.
3. **`MODEL_IDS`**: leitura automática de `model_ids` param do `TREINO_EXEC_RUN_ID` se vazio.

### Seção 1 — Configs (target após atualização)

```python
# MLflow
EXPERIMENT_NAME        = "..."
PR_INF_NAME            = "T_PR_INFERENCIA"
MODE_CODE              = "C"
INF_VERSAO             = "V1"
PR_INF_RUN_ID_OVERRIDE = ""

# Referência ao treino
TREINO_EXEC_RUN_ID = "..."   # <<< run_id do exec run T_TREINO

# Modelos a inferir
MODEL_IDS = []   # se vazio → ler de MlflowClient.get_run(TREINO_EXEC_RUN_ID).data.params["model_ids"]

# Colunas
SEG_TARGET  = "SEGURO_NOVO_MANUAL"
SEG_COL     = "SEG"
ID_COL      = "CD_NUMERO_COTACAO_AXA"
LABEL_COL   = "label"
STATUS_COL  = "DS_GRUPO_STATUS"
DATE_COL    = "DATA_COTACAO"
ID_COLS     = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]

# Features — ler do MLflow
TREINO_FEATURE_COLS = []   # se vazio → ler de param 'feature_cols'
TREINO_CAT_COLS     = []   # se vazio → ler de param 'treino_cat_cols'
TREINO_NUM_COLS     = []   # se vazio → ler de param 'treino_num_cols'

# Truncagem (mesmos parâmetros do treino)
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"

# Input/Output
INPUT_TABLE_FQN  = "gold.cotacao_validacao_..."
OUT_SCHEMA       = "gold"
OUTPUT_TABLE_FQN = f"{OUT_SCHEMA}.cotacao_inferencia_mode_c_{TS_EXEC}"
WRITE_MODE       = "overwrite"
```

### Seção 4 — Carregamento de Modelos e Pipelines

Para cada `model_id`:
```
gbt_model = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/model")
pp_fit    = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/preprocess_pipeline")
```

`top_vals_by_col` lido de `preprocess/top_vals_by_col.json` no `TREINO_EXEC_RUN_ID`.

### Seção 5 — Scoring por Modelo

```
# [1] Carregar input (df_validacao), blank→null, cast numérico
#     apply_truncation usando top_vals_by_col do MLflow

# [2] Para cada model_id:
#   a. df_vec = pp_fit.transform(df_inf_prep.select(*TREINO_FEATURE_COLS))
#   b. df_pred = gbt_model.transform(df_vec)
#   c. p_emitida = probability[1]
#   d. rank_global = row_number() over (orderBy desc p_emitida)
#   e. df_scored = join com ID_COLS + STATUS_COL + LABEL_COL + metadados
#   f. Acumular: df_scored_all.union(df_scored)

# Colunas do df final (long format):
# ID_COLS + STATUS_COL + LABEL_COL + DATE_COL + model_id + p_emitida + rank_global
# + treino_exec_run_id + inf_versao + mode_code + seg_inferida + inference_ts
```

**Decisão técnica: sem `pred_emitida` na inferência**
Threshold operacional calculado no 5_COMP. A saída da inferência é exclusivamente `p_emitida` + `rank_global`.

### Logs da run exec T_INFERENCIA

- Tags: `pipeline_tipo`, `stage=INFERENCIA`, `run_role=exec`, `mode`, `inf_versao`
- Params: `treino_exec_run_id`, `model_ids_inferred`, `input_table_fqn`, `output_table_fqn`, `seg_target`, `feature_cols`, `high_card_threshold`, `high_card_top_n`
- Metrics: `n_rows_input`, `n_rows_output`, `n_rows_per_model_{model_id}`, `n_null_p_emitida_{model_id}`
- Artifacts: `profiling_input.json`, `profiling_output.json`, `score_profile_per_model.json`, `preprocess/top_vals_by_col.json`

---

## 5_COMP_MODE_C — pendente

> Não existe ainda. Implementar após 4_INFERENCIA atualizado.

**Dados de entrada**: scores do 4_INFERENCIA_MODE_C (long format, uma linha por cotação × modelo). Métricas de treino lidas via `MlflowClient` do `TREINO_EXEC_RUN_ID`.

> **Nota sobre T_EVAL**: o 3_TREINO já produz avaliação por modelo com threshold, P/R/F1 e confusion matrix. O 5_COMP complementa com análise @K (comparando todos os modelos), curvas sobrepostas, análise de overfitting e análise exploratória por atributo. `pred_emitida` e `threshold_capacidade` são calculados apenas no 5_COMP.

### Seção 1 — Configs

```python
# MLflow
EXPERIMENT_NAME         = "..."
PR_COMP_NAME            = "T_PR_COMP"
COMP_VERSAO             = "V1"
PR_COMP_RUN_ID_OVERRIDE = ""

# Referências
TREINO_EXEC_RUN_ID   = "..."
INFERENCIA_TABLE_FQN = "gold.cotacao_inferencia_mode_c_..."

MODEL_IDS = []   # se vazio → ler do MLflow

# Colunas
STATUS_COL   = "DS_GRUPO_STATUS"
LABEL_COL    = "label"
P_COL        = "p_emitida"
MODEL_ID_COL = "model_id"
DATE_COL     = "DATA_COTACAO"
SEG_COL      = "SEG"
ID_COL       = "CD_NUMERO_COTACAO_AXA"

# Segmentação
FILTER_BY_SEG = True
SEG_TARGET    = "SEGURO_NOVO_MANUAL"

# Análise de capacidade
CAPACIDADE_PCT = 0.10
K_LIST         = [0.05, 0.10, 0.15, 0.20]

# Baseline e lift
BASELINE_MODE  = "conversao_time"
CONVERSAO_TIME = 0.30
CONVERSAO_LIST = [0.10, 0.20, 0.30]
LIFT_TARGET    = 2.0

# Análise exploratória por atributo (apenas notebook)
ATTR_ANALYSIS_COLS = ["DS_PRODUTO_NOME"]
```

### Seção 2 — Helpers

```
# compute_metrics_at_k(df_pred, model_id, n_total, k_list, baseline, label_col)
#   → precision, recall, lift, tp, fp, fn, tn @k para cada k em k_list
#
# compute_pr_curve(df_pred, model_id, label_col, p_col)
#   → curva PR completa (threshold, precision, recall, f1) como pandas DataFrame
#
# compute_threshold_f1_max(pr_curve_pdf)
#   → threshold, f1_max, precision_at_f1, recall_at_f1
```

### Seção 5 — Análise de Overfitting

```
# Para cada model_id:
#   1. AUC-PR val (sobre df_inferencia filtrado por model_id)
#   2. Ler auc_pr_treino_{model_id} do MLflow (MlflowClient)
#   3. gap_auc_pr = auc_pr_treino - auc_pr_val
#   4. Logar: auc_pr_treino, auc_pr_val, gap_auc_pr por model_id
#
# Artifacts:
#   overfitting/overfitting_summary.json
#   overfitting/score_distributions.png  (distribuição p_emitida no hold-out por modelo)
#   (curva PR de treino completa não disponível — apenas AUC-PR de treino como referência numérica)
```

### Seção 6 — Análise de Desempenho

**Parte A: Métricas globais por modelo**
- AUC-PR e AP (val)
- `threshold_capacidade` = score do K-ésimo elemento para cada k em `K_LIST`
- TP/FP/FN/TN @K, Precision@K, Recall@K, Lift@K
- `threshold_f1_max`, `f1_max`
- `pred_emitida_f1` e `pred_emitida_k` calculadas aqui

**Parte B: Curvas @K comparativas** — Precision@K, Recall@K, Lift@K com uma linha por modelo

**Parte C: Curvas PR comparativas** — uma curva PR por modelo, pontos @K marcados, threshold F1 max marcado

**Parte D: `comparativo/metrics_summary.json`** — todos os modelos com CV metrics + val metrics + @K metrics

### Seção 7 — Análise Exploratória por Atributo (apenas notebook)

Por `ATTR_ANALYSIS_COLS`: Precision@K, Recall@K, Lift@K por valor do atributo. Não logado no MLflow.

### Logs da run exec T_COMP

- Tags: `stage=COMP`, `run_role=exec`, `seg_target`, `versao_ref`
- Params: `inferencia_table_fqn`, `treino_exec_run_id`, `model_ids`, `capacidade_pct`, `k_list`, `conversao_list`, `lift_target`, `baseline_mode`, `baseline_calculado`, `filter_by_seg`, `seg_target`
- Metrics (por `model_id`): `auc_pr_val`, `ap_val`, `auc_pr_treino`, `gap_auc_pr`, `precision_at_k`, `recall_at_k`, `lift_at_k`, `tp_at_k`, `fp_at_k`, `fn_at_k`, `threshold_f1_max`, `f1_max`
- Artifacts: `comparativo/metrics_summary.json`, `comparativo/topk_precision.png`, `comparativo/topk_recall.png`, `comparativo/topk_lift.png`, `comparativo/pr_curves.png`, `overfitting/overfitting_summary.json`, `overfitting/score_distributions.png`

---

## Sequência de execução

```
1. 3_TREINO_MODE_C  ✅ executado
   → Produz: df_model (gold), df_validacao (gold)
   → Produz: artefatos MLflow com N modelos + model_ids param + eval por modelo
   → Anotar: TREINO_EXEC_RUN_ID (run exec do T_TREINO)

2. 4_INFERENCIA_MODE_C  ⚠️ atualizar antes de executar
   → Input: df_validacao + TREINO_EXEC_RUN_ID + MODEL_IDS (ou ler do MLflow)
   → Produz: tabela gold com scores por modelo (long format)
   → Anotar: INFERENCIA_TABLE_FQN e OUTPUT_TABLE_FQN

3. 5_COMP_MODE_C  ⏳ implementar
   → Input: INFERENCIA_TABLE_FQN + TREINO_EXEC_RUN_ID + MODEL_IDS
   → Produz: análises comparativas, overfitting, curvas PR, métricas @K
   → Resultado: seleção manual do modelo vencedor (se desejado)
```

---

## Adições MODE_C vs MODE_B (estado atual da implementação)

| Ponto | MODE_B | MODE_C (implementado) |
|---|---|---|
| Config de features | `FS_DECIMAL_COLS`, `FS_DIAS_COLS`, `FS_CAT_COLS` manuais | `FEATURE_CANDIDATES` dict com toggles; tipo inferido do schema |
| `COLS_NEVER_FEATURE` | Não existe | Lista explícita na config |
| n_linhas_por_regra | Não logado | Logado como métrica por rule_id (PP_R01–PP_R03) |
| PP_R04/R05/R06 | No TREINO | No PRE_PROC_MODEL — df salvo já tratado |
| Pearson | Não implementado | Após FS, sobre NUM_COLS_FINAL |
| Seleção de vencedor | Automática | Removida — todos os combos salvos |
| Artefatos de modelo | 1 modelo | N modelos (um por combo) |
| Pipeline de pré-proc | Não salvo | Salvo por model_id; carregado no 4_INFERENCIA |
| AUC-PR / AP de treino | Não calculado | Calculado por modelo (para overfitting) |
| T_EVAL no TREINO | PR curve + métricas @K sobre hold-out | PR curve + threshold selection (F1/F2/precision_ge_target); @K delegado ao 5_COMP |
| EVAL_CRITERION | Não existe | `max_f1` \| `max_f2` \| `precision_ge_target` |
| CAPACIDADE_PCT / LIFT_TARGET | Definidos no 3 | Não definidos no 3 — apenas no 5_COMP |
| pred_emitida | Gerada no 4_INFERENCIA | Gerada no 5_COMP (f(threshold_f1_max ou threshold_k)) |
| top_vals_by_col artifact | Gerado na inferência | Gerado no TREINO (preprocess/top_vals_by_col.json) |
