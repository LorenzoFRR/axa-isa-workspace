# PLANO_MODELAGEM — MODE_C

Plano conceitual de implementação dos notebooks `3_TREINO_MODE_C`, `4_INFERENCIA_MODE_C` e `5_COMP_MODE_C`.
Referência primária: `NOTAS_MODELAGEM.md`. Referência de implementação: `3_TREINO_MODE_B`, `4_INFERENCIA_MODE_B`, `5_COMP`.

---

## Visão geral do fluxo MODE_C

```
3_TREINO_MODE_C
  ├── PRE_PROC_MODEL  → df_model (gold), df_validacao (gold)
  ├── FEATURE_SELECTION → features rankeadas, top-K sets, MI, Pearson
  └── TREINO          → N modelos salvos (um por combo do grid), sem seleção de vencedor

4_INFERENCIA_MODE_C
  └── Para cada model_id: scoring do df_validacao → tabela unificada (uma linha por cotação × modelo)

5_COMP_MODE_C
  ├── Overfitting: AUC-PR treino vs val, curvas PR, distribuição de scores
  ├── Desempenho: Precision/Recall/Lift @K, curvas PR por modelo, TP/FP/FN/TN @K
  └── Exploratório (apenas notebook): performance por atributo, análise mensal
```

**Diferenças principais em relação ao MODE_B:**

| Aspecto | MODE_B | MODE_C |
|---|---|---|
| Seleção de vencedor | Automática (melhor avg AUC-PR do CV) | Nenhuma — todos os modelos são salvos |
| Modelos treinados | 1 (melhor combo) | N (um por combinação do grid) |
| Métricas de hold-out no treino | Sim (AUC-PR, Precision@K, Lift@K...) | Apenas AUC-PR/AP de treino (para overfitting) — demais delegadas ao 5_COMP |
| Feature Selection | LR-L1, RF, GBT + MI | Idem + Pearson entre features numéricas |
| Inferência | Um modelo | N modelos → tabela com `model_id` |
| Comparação | Não aplicável | 5_COMP_MODE_C compara todos os modelos |

---

## 3_TREINO_MODE_C

### Seção 1 — Configs

Estrutura idêntica ao MODE_B, organizada por bloco. Cada bloco de config precede sua etapa de execução.

#### Bloco: Gerais (topo do notebook)

```python
# MLflow
EXPERIMENT_NAME        = "..."
PR_TREINO_NAME         = "T_PR_TREINO"
MODE_CODE              = "C"
MODE_NAME              = f"T_MODE_{MODE_CODE}"

# Overrides
PR_RUN_ID_OVERRIDE       = ""
MODE_RUN_ID_OVERRIDE     = ""
PRE_PROC_RUN_ID_OVERRIDE = ""
FS_RUN_ID_OVERRIDE       = ""
TREINO_RUN_ID_OVERRIDE   = ""

# Step names
STEP_PRE_PROC_NAME          = "T_PRE_PROC_MODEL"
STEP_FEATURE_SELECTION_NAME = "T_FEATURE_SELECTION"
STEP_TREINO_NAME            = "T_TREINO"

# Versionamento
TREINO_VERSAO = "V1"
TS_EXEC       = datetime.now(...).strftime(...)
RUN_UUID      = uuid.uuid4().hex[:8]

# I/O
COTACAO_SEG_FQN = "silver.cotacao_seg_..."   # <<< AJUSTE
OUT_SCHEMA      = "gold"
DF_MODEL_FQN    = f"{OUT_SCHEMA}.cotacao_model_{TS_EXEC}_{RUN_UUID}"
DF_VALID_FQN    = f"{OUT_SCHEMA}.cotacao_validacao_{TS_EXEC}_{RUN_UUID}"

# Seeds e salts (logados obrigatoriamente em cada run que os usa)
SPLIT_SALT = "split_c1_seg_mes"   # <<< AJUSTE — string auditável
CV_SEED    = 42
FS_SEEDS   = [42, 123, 7]
```

#### Bloco: PRE_PROC_MODEL

```python
STATUS_COL           = "DS_GRUPO_STATUS"
LABEL_COL            = "label"
ID_COL               = "CD_NUMERO_COTACAO_AXA"
SEG_COL              = "SEG"
DATE_COL             = "DATA_COTACAO"
ALLOWED_FINAL_STATUS = ["Emitida", "Perdida"]
VALID_FRAC           = 0.20
SEG_TARGET           = "SEGURO_NOVO_MANUAL"   # <<< AJUSTE
DO_PROFILE           = True

# Colunas candidatas a features (definidas após inspecionar df_seg.columns)
FS_DECIMAL_COLS = [...]
FS_DIAS_COLS    = [...]
FS_CAT_COLS     = [...]
EXCLUIR_DE_FEATURES = set()   # exclusões explícitas e documentadas

# Thresholds de limpeza ML
NULL_DROP_PCT       = 0.90
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"
```

**Decisão técnica**: `FS_DECIMAL_COLS`, `FS_DIAS_COLS` e `FS_CAT_COLS` são definidas aqui (na seção PRE_PROC_MODEL) e reutilizadas pelo FS e pelo TREINO. O conjunto `EXCLUIR_DE_FEATURES` permite remover colunas de forma explícita e auditável.

#### Bloco: FEATURE_SELECTION

```python
FS_TRAIN_FRAC = 0.70
TOPK_LIST     = [5, 7, 12]
FS_METHODS_CONFIG = {
    "lr_l1": {"maxIter": 100, "regParam": 0.01, "elasticNetParam": 1.0},
    "rf":    {"numTrees": 200, "maxDepth": 8},
    "gbt":   {"maxIter": 80, "maxDepth": 5, "stepSize": 0.1},
}
```

#### Bloco: TREINO

```python
TREINO_FEATURE_SET_KEY = "top_7"   # <<< chave do feature set do FS

USE_CLASS_WEIGHT       = "auto"
CLASS_WEIGHT_THRESHOLD = 0.30

CV_FOLDS  = 3
CV_METRIC = "areaUnderPR"

# Grid de hiperparâmetros — alterar listas sem impactar a lógica
GBT_PARAM_GRID = {
    "maxDepth": [4, 6],
    "stepSize": [0.05, 0.1],
    "maxIter":  100,          # fixo, não varia no grid
}

ID_COLS            = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]
DROP_FROM_FEATURES = ID_COLS + [STATUS_COL]
```

**Decisão técnica**: `GBT_PARAM_GRID` aceita listas arbitrárias para qualquer chave que não seja `maxIter`. Os pares são gerados via `itertools.product` das listas — aumentar ou reduzir o grid não exige alteração no código de execução.

---

### Seção 2 — Imports e Helpers Globais

Idêntica ao MODE_B. Inclui: `rule_def`, `apply_rules_block`, `rules_catalog_for_logging`, `profile_basic`, `counts_by_seg`, `label_rate_by_seg`, `safe_drop_cols`, `ensure_schema`, `table_exists`, `assert_table_exists`, `mlflow_get_or_create_experiment`.

---

### Seção 3 — T_PRE_PROC_MODEL

#### Engine de regras (silver → gold)

Mesma engine do MODE_B: `rule_def`, `apply_rules_block`, toggles centralizados, catálogo logado como `rules_catalog.json`. As regras desta etapa cobrem a transição silver → gold — anteriores ao split, sem dependência de dados de treino. Transformações de ML (encoding, imputer, assembler) pertencem exclusivamente ao TREINO.

**Prefixos e categorias:**
- `PP_`: normalização/filtragem/label sobre dados existentes — modificam linhas ou colunas já presentes na silver
- `BUILD_`: criação de colunas auxiliares derivadas (`MES`, `is_valid`) — pré-requisitos para o split; não existem na silver
- `MODEL_` / `VALID_`: filtros de materialização do split — não são toggleáveis

#### Funções de regra

Mesmas do MODE_B:
- `PP_R01_normaliza_status` — normaliza EMITIDA/PERDIDA
- `PP_R02_filtra_status_finais` — mantém apenas Emitida e Perdida
- `PP_R03_cria_label` — cria coluna `label` (1.0 / 0.0)
- `BUILD_R01_add_mes` — cria coluna `MES` a partir de `DATE_COL`
- `BUILD_R02_add_split_flag` — split determinístico por hash: `xxhash64(ID, SEG, MES, SPLIT_SALT)` → `is_valid`
- `MODEL_R03_filtra_model`, `VALID_R03_filtra_validacao`, `BUILD_R04_drop_aux`

#### Toggles e catálogo de regras

Mesma engine `RULES_BY_BLOCK` do MODE_B.

#### Execução

Fluxo idêntico ao MODE_B. **Adição em MODE_C**: logar `n_linhas_por_regra_{rule_id}` como métrica MLflow após cada regra aplicada. Implementação: percorrer o `exec_log` retornado por `apply_rules_block` e, para as regras com status `APPLIED`, realizar `.count()` incremental ou usar as contagens intermediárias já calculadas (ex: `n_seg_after_rules`). Para minimizar ações adicionais, pode-se logar apenas as contagens nos pontos de transição já existentes (`n_seg_in`, `n_seg_after_rules`, `n_df_model`, `n_df_validacao`), complementadas pela execução do `exec_log` que rastreia quais regras foram aplicadas.

**Logs da run exec T_PRE_PROC_MODEL:**
- Tags: `pipeline_tipo`, `stage=TREINO`, `run_role=exec`, `mode`, `step=PRE_PROC_MODEL`
- Params: `ts_exec`, `treino_versao`, `mode_code`, `seg_target`, `input_cotacao_seg_fqn`, `df_model_fqn`, `df_validacao_fqn`, `valid_frac`, `split_salt`, `allowed_final_status`, `label_col`, `id_col`, `seg_col`, `pr_run_id`, `mode_run_id`
- Metrics: `n_seg_in`, `n_seg_after_rules`, `n_linhas_por_regra_{rule_id}` (nova), `n_df_model`, `n_df_validacao`
- Artifacts: `rules_catalog.json`, `rules_execution.json`, `profiling_df_model.json`, `profiling_df_validacao.json`, `eda_df_model_by_seg.json`

---

### Seção 4 — T_FEATURE_SELECTION

#### Imports e Helpers

Mesmos do MODE_B. Adicionar import:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns   # opcional, para heatmap
```

#### Execução

Idêntica ao MODE_B nos passos [1]–[8]. **Adição em MODE_C: passo [9] — Pearson**.

**Passo [9] — Correlação de Pearson (nova)**

Executado após o ensemble, sobre as features numéricas do ranking final (`NUM_COLS_FINAL`). Não influencia a seleção.

```
# Alto nível:
# 1. Amostrar df_audit para pandas (até 50k linhas — mesma amostra do MI ou nova)
# 2. Selecionar apenas NUM_COLS_FINAL (features numéricas)
# 3. Calcular pdf[num_cols].corr(method="pearson")
# 4. Converter para formato longo (feature_a, feature_b, correlation)
# 5. Ordenar features pelo rank do FS (para legibilidade do heatmap)
# 6. Logar como artifact:
#    - pearson/pearson_correlation.csv
#    - pearson/pearson_heatmap.png (heatmap anotado, matplotlib)
```

**Logs da run exec T_FEATURE_SELECTION:**
- Idênticos ao MODE_B +
- Artifacts: `pearson/pearson_correlation.csv` (novo), `pearson/pearson_heatmap.png` (novo)

---

### Seção 5 — T_TREINO

#### Imports e Helpers

Mesmos do MODE_B. Manter: `build_preprocess_pipeline`, `add_class_weights`, `kfold_split`, `compute_capacity_metrics`, `confusion_matrix_at_threshold`.

**Remover** o uso de `compute_capacity_metrics` na etapa de avaliação do hold-out — as métricas de hold-out são delegadas ao 5_COMP. Mantém-se apenas o cálculo de AUC-PR e AP de treino (para análise de overfitting).

#### Execução

**Diferença fundamental**: o loop itera sobre todas as combinações do grid. Para cada combo: CV + treino final + salvamento de artefatos. Não há seleção de vencedor.

```
# Alto nível:

# [1] Carga, limpeza, truncagem de cardinalidade, class weight
#     Idêntico ao MODE_B

# [2] CV 3-fold determinístico por hash
#     Para cada combo do GBT_PARAM_GRID (via itertools.product das listas):
#       model_id = f"d{maxDepth}_s{str(stepSize).replace('.','')}"
#       Para cada fold (kfold_split):
#         fit pipeline + fit GBT → evaluate (auc_pr)
#       Agregar: avg_auc_pr, std_auc_pr por combo
#       Logar métricas: cv_{model_id}_fold{i}_auc_pr, cv_{model_id}_avg_auc_pr, cv_{model_id}_std_auc_pr

# [3] Treino final — para CADA combo (não apenas o vencedor):
#       pp_fit = build_preprocess_pipeline(...).fit(df_model_ml)
#       gbt = GBTClassifier(...params do combo...)
#       gbt_model = gbt.fit(pp_fit.transform(df_model_ml))
#
#       Salvar artefatos:
#         mlflow.spark.log_model(gbt_model, f"treino_final/{model_id}/model")
#         mlflow.spark.log_model(pp_fit,    f"treino_final/{model_id}/preprocess_pipeline")
#
#       Calcular AUC-PR e AP de treino (para análise de overfitting no 5_COMP):
#         df_model_pred = gbt_model.transform(pp_fit.transform(df_model_ml))
#         auc_pr_treino = evaluator_pr.evaluate(df_model_pred)
#         ap_treino     = compute_ap(df_model_pred)
#         Logar: auc_pr_treino_{model_id}, ap_treino_{model_id}
#
#       Acumular em TRAINED_MODELS[model_id] = {params, cv_avg_auc_pr, cv_std_auc_pr,
#                                               auc_pr_treino, ap_treino}

# [4] Serializar TRAINED_MODELS como artifact:
#       cv/trained_models_registry.json
#       cv/grid_results.json, cv/fold_metrics.json (iguais ao MODE_B)
#
#     Logar param: model_ids = json.dumps(list(TRAINED_MODELS.keys()))
```

**Decisão técnica: TRAINED_MODELS**
Dicionário em memória construído ao longo do loop de treino. Serve como contrato com os notebooks 4 e 5. Chave = `model_id`, valor = `{params, cv_avg_auc_pr, cv_std_auc_pr, auc_pr_treino, ap_treino}`.

**Decisão técnica: persistência entre sessões**
Modelos e pipelines são salvos via `mlflow.spark.log_model` no artifact store do MLflow. Após encerramento do cluster, o 4_INFERENCIA carrega ambos com `mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/model")` e `.../{model_id}/preprocess_pipeline"`. O par `(TREINO_EXEC_RUN_ID, model_id)` é a referência suficiente — não é necessário reprocessar `DF_MODEL_FQN` para reconstruir o pipeline.

**Decisão técnica: sem métricas de hold-out no TREINO**
Toda a avaliação sobre `df_validacao` é feita no 5_COMP. O TREINO só calcula métricas sobre `df_model` (treino) para fins de overfitting. `CAPACIDADE_PCT` e `LIFT_TARGET` não são definidos no 3_TREINO.

**Logs da run exec T_TREINO:**
- Tags: `pipeline_tipo`, `stage=TREINO`, `run_role=exec`, `mode`, `step=TREINO`
- Params: `df_model_fqn`, `df_valid_fqn`, `seg_target`, `feature_set`, `feature_cols`, `n_features`, `n_model`, `use_class_weight`, `apply_cw`, `weight_pos`, `label_rate`, `cv_folds`, `cv_seed`, `cv_metric`, `gbt_param_grid`, `model_ids`, `mode_code`, `pr_run_id`, `mode_run_id`, `treino_container_run_id`
- Metrics: `cv_{model_id}_fold{i}_auc_pr`, `cv_{model_id}_avg_auc_pr`, `cv_{model_id}_std_auc_pr`, `auc_pr_treino_{model_id}`, `ap_treino_{model_id}` (para cada combo)
- Artifacts: `cv/grid_results.json`, `cv/fold_metrics.json`, `cv/trained_models_registry.json`, `treino_final/{model_id}/model` (um por combo), `treino_final/{model_id}/preprocess_pipeline` (um por combo)

---

## 4_INFERENCIA_MODE_C

### Seção 1 — Configs

```python
# MLflow
EXPERIMENT_NAME        = "..."
PR_INF_NAME            = "T_PR_INFERENCIA"
MODE_CODE              = "C"
INF_VERSAO             = "V1"
PR_INF_RUN_ID_OVERRIDE = ""

# Referência ao treino
TREINO_EXEC_RUN_ID = "..."   # <<< run_id do exec run T_TREINO (run_role=exec, step=TREINO)

# Modelos a inferir (subconjunto ou totalidade do grid)
MODEL_IDS = ["d4_s005", "d4_s01", "d6_s005", "d6_s01"]   # <<< AJUSTE — ou ler do MLflow

# Pipeline e modelo carregados do MLflow — não requer DF_MODEL_FQN

# Colunas (mesmas do 3_TREINO)
SEG_TARGET  = "SEGURO_NOVO_MANUAL"
SEG_COL     = "SEG"
ID_COL      = "CD_NUMERO_COTACAO_AXA"
LABEL_COL   = "label"
STATUS_COL  = "DS_GRUPO_STATUS"
DATE_COL    = "DATA_COTACAO"
ID_COLS     = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]

# Features (ler do param 'feature_cols' logado no MLflow do TREINO_EXEC_RUN_ID)
TREINO_FEATURE_COLS = [...]   # <<< preencher com o param logado

# Listas de tipo (mesmas do 3_TREINO)
FS_CAT_COLS     = [...]
FS_DECIMAL_COLS = [...]
FS_DIAS_COLS    = [...]

# Truncagem de cardinalidade (mesmos parâmetros do treino)
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"

# Input/Output
INPUT_TABLE_FQN  = "gold.cotacao_validacao_..."   # <<< df_validacao do 3_TREINO
OUT_SCHEMA       = "gold"
OUTPUT_TABLE_FQN = f"{OUT_SCHEMA}.cotacao_inferencia_mode_c_{TS_EXEC}"
WRITE_MODE       = "overwrite"
```

**Decisão técnica: MODEL_IDS**
Pode ser preenchido manualmente ou lido via `MlflowClient.get_run(TREINO_EXEC_RUN_ID).data.params["model_ids"]`. Leitura via MLflow é mais segura, pois evita inconsistências.

**Decisão técnica: sem `pred_emitida` na inferência**
Não gerar coluna `pred_emitida` — o threshold operacional será calculado no 5_COMP a partir de `CAPACIDADE_PCT`. Isso evita ambiguidade sobre qual threshold foi usado.

### Seção 2 — Imports e Helpers

Mesmos do MODE_B. **Remover**: `build_preprocess_pipeline`, `compute_top_vals` (pipeline carregado do MLflow; top_vals lido de artefato). **Manter**: `blanks_to_null`, `cast_to_double`, `apply_truncation` (aplicados sobre o df de inferência antes de passar ao pipeline), `profile_basic`, `profile_score`.

### Seção 3 — Preparos MLflow e Validações

Validações de existência das tabelas e preenchimento dos configs obrigatórios. Abre PR container e run exec.

**Adição em MODE_C**: ler `model_ids` e `feature_cols` via `MlflowClient` a partir do `TREINO_EXEC_RUN_ID`, se não preenchidos diretamente nas configs.

### Seção 4 — Carregamento de Modelos e Pipelines (via MLflow)

Em MODE_C, diferente do MODE_B, **não há reconstrução do pipeline** — modelo e pipeline de pré-processamento são carregados diretamente do MLflow. O `top_vals_by_col` (truncagem de cardinalidade) é lido do artefato `preprocess/top_vals_by_col.json` logado pelo 3_TREINO.

```
# Para cada model_id em MODEL_IDS:
#   gbt_model = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/model")
#   pp_fit    = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/preprocess_pipeline")
#
# top_vals_by_col = mlflow_client.download_artifacts(TREINO_EXEC_RUN_ID, "preprocess/top_vals_by_col.json")
# (reutilizado na truncagem do df de inferência)
```

**Decisão técnica**: o par `(TREINO_EXEC_RUN_ID, model_id)` é a referência completa e suficiente. Elimina dependência de `DF_MODEL_FQN` e garante que a inferência usa exatamente o pipeline fitado no treino, sem risco de divergência por refit.

### Seção 5 — Scoring por Modelo

Modelo e pipeline para cada `model_id` foram carregados na Seção 4. Seção 5 aplica a truncagem ao df de inferência e executa o scoring.

```
# [1] Carregar input e aplicar limpeza/truncagem (usando top_vals_by_col do MLflow)
#     df_inf_prep: blanks_to_null + cast_to_double + apply_truncation

# [2] Para cada model_id em MODEL_IDS:
#   a. Carregar gbt_model e pp_fit do MLflow (Seção 4)
#   b. df_vec = pp_fit.transform(df_inf_prep.select(*feature_cols))
#   c. df_pred = gbt_model.transform(df_vec)
#   d. p_emitida = probability[1]
#   e. rank_global = row_number() over (orderBy desc p_emitida)
#   f. df_scored = df_inf_prep.join(scores) + metadados (model_id, treino_exec_run_id, ...)
#   g. Acumular: df_scored_all = df_scored_all.union(df_scored)
#
# Colunas do df final: ID_COLS + STATUS_COL + LABEL_COL + DATE_COL + model_id + p_emitida + rank_global
#                      + metadados (treino_exec_run_id, inf_versao, mode_code, seg_inferida, inference_ts)
```

**Decisão técnica: estrutura da tabela de output (long format)**
Uma linha por cotação × modelo. `pred_emitida` não é gerada aqui — será calculada no 5_COMP com os thresholds definidos (`threshold_f1_max`, `threshold_capacidade`).

**Logs da run exec T_INFERENCIA:**
- Tags: `pipeline_tipo`, `stage=INFERENCIA`, `run_role=exec`, `mode`, `inf_versao`
- Params: `treino_exec_run_id`, `model_ids_inferred`, `input_table_fqn`, `output_table_fqn`, `seg_target`, `df_model_fqn`, `feature_cols`, `high_card_threshold`, `high_card_top_n`
- Metrics: `n_rows_input`, `n_rows_output`, `n_rows_per_model_{model_id}`, `n_null_p_emitida_{model_id}`
- Artifacts: `profiling_input.json`, `profiling_output.json`, `score_profile_per_model.json`, `preprocess/top_vals_by_col.json`

---

## 5_COMP_MODE_C

**Dados de entrada**: todas as análises operam sobre os scores gerados pelo 4_INFERENCIA_MODE_C, que aplica os modelos exclusivamente sobre `df_validacao` (hold-out). O `df_model` (treino) não é acessado diretamente — apenas métricas de treino já logadas no MLflow (`auc_pr_treino_{model_id}`) são lidas via `MlflowClient`. O isolamento do hold-out é garantido pelo `BUILD_R02_add_split_flag` no PRE_PROC_MODEL.

**Thresholds e `pred_emitida`**: o 5_COMP é a única etapa onde thresholds de classificação são definidos. Para cada `model_id`, calcula-se `threshold_f1_max` (maximiza F1 sobre scores do hold-out) e `threshold_capacidade` (score do K-ésimo elemento via `CAPACIDADE_PCT`). `pred_emitida` = f(threshold) é gerada aqui — em duas versões: `pred_emitida_f1` e `pred_emitida_k` — e usada para as confusion matrices @K correspondentes.

### Seção 1 — Configs

```python
# MLflow
EXPERIMENT_NAME         = "..."
PR_COMP_NAME            = "T_PR_COMP"
COMP_VERSAO             = "V1"
PR_COMP_RUN_ID_OVERRIDE = ""

# Referências
TREINO_EXEC_RUN_ID   = "..."   # <<< para leitura de métricas de treino
INFERENCIA_TABLE_FQN = "gold.cotacao_inferencia_mode_c_..."   # <<< output do 4_INFERENCIA

# Modelos a comparar (subconjunto ou totalidade)
MODEL_IDS = ["d4_s005", "d4_s01", "d6_s005", "d6_s01"]   # <<< AJUSTE

# Colunas estruturais
STATUS_COL = "DS_GRUPO_STATUS"
LABEL_COL  = "label"
P_COL      = "p_emitida"
MODEL_ID_COL = "model_id"
DATE_COL   = "DATA_COTACAO"
SEG_COL    = "SEG"
ID_COL     = "CD_NUMERO_COTACAO_AXA"

# Segmentação
FILTER_BY_SEG = True
SEG_TARGET    = "SEGURO_NOVO_MANUAL"   # <<< AJUSTE

# Análise de capacidade
CAPACIDADE_PCT = 0.10   # <<< capacidade principal
K_LIST         = [0.05, 0.10, 0.15, 0.20]   # capacidades para análise

# Baseline e lift
BASELINE_MODE  = "conversao_time"   # "taxa_base" | "conversao_time"
CONVERSAO_TIME = 0.30   # <<< preencher se BASELINE_MODE="conversao_time"
CONVERSAO_LIST = [0.10, 0.20, 0.30]   # cenários para curva PR
LIFT_TARGET    = 2.0

# Análise exploratória por atributo (apenas notebook)
ATTR_ANALYSIS_COLS = ["DS_PRODUTO_NOME"]   # <<< AJUSTE
```

### Seção 2 — Imports e Helpers

Mesmos do 5_COMP (MODE_B). Adições:
```python
from mlflow.tracking import MlflowClient
```
Helpers: `log_png`, `ensure_required_cols`, paleta de cores. Funções auxiliares para cálculo de métricas @K:

```
# Helpers novos ou estendidos:
# compute_metrics_at_k(df_pred, model_id, n_total, k_list, baseline, label_col)
#   → para cada k em k_list: precision, recall, lift, tp, fp, fn, tn @k
#   → retorna lista de dicts com todas as métricas
#
# compute_pr_curve(df_pred, model_id, label_col, p_col)
#   → calcula curva PR completa (precision/recall por threshold)
#   → retorna pandas DataFrame com colunas: threshold, precision, recall, f1
#
# compute_threshold_f1_max(pr_curve_pdf)
#   → encontra o threshold que maximiza F1 na curva PR
#   → retorna: threshold, f1_max, precision_at_f1, recall_at_f1
```

### Seção 3 — MLflow: Abertura de Runs

Mesma estrutura do 5_COMP (MODE_B): PR container → exec run nested. Tags e params iniciais logados na exec run.

**Adição**: logar `model_ids`, `k_list`, `conversao_list`, `lift_target`, `treino_exec_run_id` como params.

### Seção 4 — Carga e Preparação

```
# 1. Carregar tabela de inferência
# 2. Filtrar por SEG_TARGET (se FILTER_BY_SEG=True)
# 3. Cast e limpeza: p_emitida → double, label_real derivado de STATUS_COL
# 4. Verificar que todos os MODEL_IDS estão presentes na coluna model_id
# 5. Calcular baseline:
#    - "taxa_base"     → label_real.mean() calculado no hold-out (por modelo, deve ser igual)
#    - "conversao_time" → CONVERSAO_TIME
# 6. Logar n_total, n_pos, baseline
```

### Seção 5 — Análise de Overfitting

**Entrada adicional**: ler `auc_pr_treino_{model_id}` e `ap_treino_{model_id}` via `MlflowClient.get_run(TREINO_EXEC_RUN_ID).data.metrics`.

```
# Para cada model_id:
#   1. Calcular AUC-PR val (sobre df_inferencia filtrado por model_id)
#   2. Ler auc_pr_treino_{model_id} do MLflow (via MlflowClient)
#   3. gap_auc_pr = auc_pr_treino - auc_pr_val
#   4. Logar: auc_pr_treino_{model_id}, auc_pr_val_{model_id}, gap_auc_pr_{model_id}
#
# Artifacts:
#   - overfitting/overfitting_summary.json: tabela com todos os modelos, treino vs val, gap
#   - overfitting/pr_curves_treino_vs_val_{model_id}.png: curvas PR sobrepostas treino/val
#     (curva de treino requer scores de treino — não disponível diretamente na tabela de inferência)
#     → Simplificação: logar AUC-PR treino via métrica do 3_TREINO; plotar curva PR somente do val
#        e marcar o ponto de treino (auc_pr_treino como ponto de referência no gráfico)
#   - overfitting/score_distributions.png: histogramas de p_emitida por modelo e split
#     (val disponível na tabela de inferência; treino requer acesso ao df_model — omitir ou amostrar)
```

**Decisão técnica: curva PR de treino**
A curva PR de treino completa não está disponível na tabela de inferência. Alternativas: (a) apenas logar AUC-PR de treino como referência numérica; (b) requerer que o 3_TREINO também salve os scores de treino. Opção (a) é preferida para não sobrecarregar o 3_TREINO. A análise visual de overfitting principal será o gap numérico (AUC-PR treino vs val) e a distribuição de scores de validação.

### Seção 6 — Análise de Desempenho

**Parte A: Métricas globais por modelo**

```
# Para cada model_id:
#   1. Filtrar df por model_id
#   2. Calcular AUC-PR e AP (igual ao 5_COMP MODE_B)
#   3. Calcular threshold de capacidade por k em K_LIST:
#      threshold_k = score do K-ésimo elemento no ranking desc
#      TP/FP/FN/TN @K, Precision@K, Recall@K, Lift@K
#   4. Calcular threshold_f1_max e f1_max (varrer scores únicos como candidatos)
#   5. Logar métricas no MLflow para CAPACIDADE_PCT principal
#      (demais K's ficam nos artifacts)
```

**Parte B: Curvas @K comparativas (logadas no MLflow)**

```
# Gerar um gráfico por métrica (Precision@K, Recall@K, Lift@K):
#   - Eixo X: K (% do hold-out), de 0 a 100%
#   - Uma linha por model_id, cores distintas
#   - Linha tracejada: baseline aleatório (Precision@K = base_rate; Recall@K = K; Lift@K = 1)
#   - Linha horizontal: LIFT_TARGET × baseline (somente no gráfico de Lift@K)
#   - Marcar pontos em CAPACIDADE_PCT principal
# Logar como: comparativo/topk_precision.png, comparativo/topk_recall.png, comparativo/topk_lift.png
```

**Parte C: Curvas PR comparativas (logadas no MLflow)**

```
# Um gráfico com uma curva PR por model_id, cores distintas:
#   - Eixo X: recall, Eixo Y: precision
#   - Para cada K em K_LIST: marcar os pontos de operação sobre cada curva
#   - Para cada conversao em CONVERSAO_LIST: linha horizontal em precision = LIFT_TARGET × conversao
#   - Threshold F1 máximo: marcar como ponto especial (estrela) em cada curva
#   - Legenda: model_id + AUC-PR val
# Logar como: comparativo/pr_curves.png
```

**Parte D: Resumo comparativo (artifact)**

```
# comparativo/metrics_summary.json:
#   Lista de dicts, um por model_id, com:
#   - model_id, params (maxDepth, stepSize, maxIter)
#   - cv_avg_auc_pr, cv_std_auc_pr (lidos do MLflow do TREINO_EXEC_RUN_ID)
#   - auc_pr_val, ap_val
#   - auc_pr_treino, gap_auc_pr
#   - threshold_f1_max, f1_max
#   - Para cada k em K_LIST: threshold_k, precision_k, recall_k, lift_k, tp_k, fp_k, fn_k, tn_k
```

### Seção 7 — Análise Exploratória por Atributo (apenas notebook)

```
# Para cada col em ATTR_ANALYSIS_COLS:
#   Para cada model_id em MODEL_IDS:
#     Para cada valor único da coluna:
#       Calcular Precision@K, Recall@K, Lift@K (para CAPACIDADE_PCT principal)
#     Plotar curvas com uma linha por valor do atributo
#
# Não logar no MLflow — apenas exibir no notebook.
```

**Logs da run exec T_COMP:**
- Tags: `stage=COMP`, `run_role=exec`, `seg_target`, `versao_ref`
- Params: `inferencia_table_fqn`, `treino_exec_run_id`, `model_ids`, `capacidade_pct`, `k_list`, `conversao_list`, `lift_target`, `baseline_mode`, `baseline_calculado`, `filter_by_seg`, `seg_target`
- Metrics (por `model_id`): `auc_pr_val_{model_id}`, `ap_val_{model_id}`, `auc_pr_treino_{model_id}`, `gap_auc_pr_{model_id}`, `precision_at_k_{model_id}`, `recall_at_k_{model_id}`, `lift_at_k_{model_id}`, `tp_at_k_{model_id}`, `fp_at_k_{model_id}`, `fn_at_k_{model_id}`, `threshold_f1_max_{model_id}`, `f1_max_{model_id}`
- Artifacts: `comparativo/metrics_summary.json`, `comparativo/topk_precision.png`, `comparativo/topk_recall.png`, `comparativo/topk_lift.png`, `comparativo/pr_curves.png`, `overfitting/overfitting_summary.json`, `overfitting/score_distributions.png`

---

## Modificações em notebooks existentes

### 3_TREINO_MODE_C (novo) — requer nenhuma modificação retroativa

### 4_INFERENCIA_MODE_B — nenhuma modificação necessária

### 5_COMP — nenhuma modificação necessária

### Adições no 3_TREINO_MODE_C vs MODE_B (resumo)

| Ponto | MODE_B | MODE_C (adição/mudança) |
|---|---|---|
| n_linhas_por_regra | Não logado | Logado como métrica por rule_id |
| seg_target no PRE_PROC_MODEL | Não logado | Logado como param na run |
| Pearson | Não implementado | Após FS, sobre NUM_COLS_FINAL |
| Seleção de vencedor | Automática | Removida — todos os combos salvos |
| Artefatos de modelo | 1 modelo | N modelos (um por combo) |
| Pipeline de pré-proc | Não salvo no treino | Salvo por model_id; carregado no 4_INFERENCIA |
| Reconstrução do pipeline no 4 | Sim (via DF_MODEL_FQN) | Não — carregado do MLflow |
| AUC-PR / AP de treino | Não calculado | Calculado por modelo (para overfitting) |
| CAPACIDADE_PCT / LIFT_TARGET | Definidos no 3 | Removidos do 3 — apenas no 5_COMP |
| pred_emitida | Gerada no 4_INFERENCIA | Gerada no 5_COMP (f(threshold_f1_max ou threshold_k)) |

---

## Sequência de execução

```
1. 3_TREINO_MODE_C
   → Produz: df_model (gold), df_validacao (gold)
   → Produz: artefatos MLflow com N modelos + model_ids param
   → Anotar: TREINO_EXEC_RUN_ID (run exec do T_TREINO)

2. 4_INFERENCIA_MODE_C
   → Input: df_validacao + TREINO_EXEC_RUN_ID + MODEL_IDS
   → Produz: tabela gold com scores por modelo (long format)
   → Anotar: INFERENCIA_TABLE_FQN

3. 5_COMP_MODE_C
   → Input: INFERENCIA_TABLE_FQN + TREINO_EXEC_RUN_ID + MODEL_IDS
   → Produz: análises comparativas, overfitting, curvas PR, métricas @K
   → Resultado: seleção manual do modelo vencedor (se desejado)
```
