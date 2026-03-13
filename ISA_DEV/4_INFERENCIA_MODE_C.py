# Databricks notebook source
# MAGIC %md
# MAGIC ## Configs

# COMMAND ----------

from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

# =========================
# MLflow
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"  # <<< AJUSTE

PR_INF_NAME = "T_PR_INFERENCIA"
MODE_CODE   = "C"
INF_VERSAO  = "V1"
VERSAO_REF  = INF_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# Override: preencha para reutilizar container PR existente
PR_INF_RUN_ID_OVERRIDE = ""

# =========================
# REFERÊNCIA AO TREINO
# =========================
# run_id do exec run T_TREINO (run_role=exec, step=TREINO) — impresso no final do 3_TREINO_MODE_C
TREINO_EXEC_RUN_ID = ""   # <<< OBRIGATÓRIO

# =========================
# MODELOS A INFERIR
# =========================
# Subconjunto ou totalidade do grid treinado.
# Se vazio (""), MODEL_IDS será lido automaticamente do param 'model_ids' do TREINO_EXEC_RUN_ID.
MODEL_IDS = []   # <<< AJUSTE ou deixar vazio para ler do MLflow

# =========================
# COLUNAS (mesmas do 3_TREINO_MODE_C)
# =========================
STATUS_COL = "DS_GRUPO_STATUS"
LABEL_COL  = "label"
ID_COL     = "CD_NUMERO_COTACAO_AXA"
SEG_COL    = "SEG"
DATE_COL   = "DATA_COTACAO"
SEG_TARGET = "SEGURO_NOVO_MANUAL"   # <<< AJUSTE

ID_COLS = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]

# =========================
# FEATURES
# =========================
# Preencher com o param 'feature_cols' logado na run TREINO_EXEC_RUN_ID.
# Se vazio ([]), será lido automaticamente do MLflow.
TREINO_FEATURE_COLS = []   # <<< AJUSTE ou deixar vazio para ler do MLflow

# Listas de tipo — mesmas do 3_TREINO_MODE_C
FS_CAT_COLS = [
    "INTERMENDIARIO_PERFIL",
    "DS_PRODUTO_NOME",
    "DS_SISTEMA",
    "CD_FILIAL_RESPONSAVEL_COTACAO",
    "DS_ATIVIDADE_SEGURADO",
    "DS_GRUPO_CORRETOR_SEGMENTO",
]
FS_DECIMAL_COLS = [
    "VL_PREMIO_ALVO", "VL_PREMIO_LIQUIDO", "VL_PRE_TOTAL",
    "VL_ENDOSSO_PREMIO_TOTAL", "VL_GWP_CORRETOR_RESUMO",
]
FS_DIAS_COLS = [
    "DIAS_INICIO_VIGENCIA", "DIAS_VALIDADE", "DIAS_ANALISE_SUBSCRICAO",
    "DIAS_FIM_ANALISE_SUBSCRICAO", "DIAS_COTACAO", "DIAS_ULTIMA_ATUALIZACAO",
]

# Parâmetros de truncagem — mesmos do 3_TREINO_MODE_C
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"

# =========================
# INPUT / OUTPUT
# =========================
# df_validacao gerado pelo 3_TREINO_MODE_C (logado como param 'df_validacao_fqn')
INPUT_TABLE_FQN = "gold.cotacao_validacao_..."   # <<< AJUSTE

OUT_SCHEMA       = "gold"
OUTPUT_TABLE_FQN = f"{OUT_SCHEMA}.cotacao_inferencia_mode_c_{TS_EXEC}"
WRITE_MODE       = "overwrite"

print("✅ CONFIG INFERÊNCIA MODE_C carregada")
print("• input table      :", INPUT_TABLE_FQN)
print("• treino_exec_run  :", TREINO_EXEC_RUN_ID or "(a preencher)")
print("• model_ids        :", MODEL_IDS or "(ler do MLflow)")
print("• seg_target       :", SEG_TARGET)
print("• features         :", TREINO_FEATURE_COLS or "(ler do MLflow)")
print("• output table     :", OUTPUT_TABLE_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports e helpers

# COMMAND ----------

import json
import os
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.functions import vector_to_array


def ensure_schema(schema_name: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

def table_exists(fqn: str) -> bool:
    return spark.catalog.tableExists(fqn)

def assert_table_exists(fqn: str) -> None:
    if not table_exists(fqn):
        raise ValueError(f"❌ Tabela não existe: {fqn}")

def mlflow_get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id

def profile_basic(df: DataFrame, name: str) -> dict:
    return {"name": name, "n_rows": int(df.count()), "n_cols": len(df.columns)}

def profile_score(df: DataFrame, score_col: str = "p_emitida") -> dict:
    if score_col not in df.columns:
        return {"score_col": score_col, "ok": False}
    agg = df.agg(
        F.count(F.lit(1)).alias("n"),
        F.sum(F.when(F.col(score_col).isNull(), 1).otherwise(0)).alias("n_null"),
        F.min(score_col).alias("min"),
        F.avg(score_col).alias("mean"),
        F.max(score_col).alias("max"),
    ).collect()[0]
    return {
        "score_col": score_col, "ok": True,
        "n":    int(agg["n"]),
        "n_null": int(agg["n_null"]),
        "min":  float(agg["min"])  if agg["min"]  is not None else None,
        "mean": float(agg["mean"]) if agg["mean"] is not None else None,
        "max":  float(agg["max"])  if agg["max"]  is not None else None,
    }

def blanks_to_null(df: DataFrame, cols: list) -> DataFrame:
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c)))
    return df

def cast_to_double(df: DataFrame, cols: list) -> DataFrame:
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))
    return df

def apply_truncation(df: DataFrame, col_name: str, top_vals: list, outros: str = "OUTROS") -> DataFrame:
    return df.withColumn(
        col_name,
        F.when(F.col(col_name).isin(top_vals), F.col(col_name)).otherwise(F.lit(outros)),
    )

print("✅ Helpers carregados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparos MLflow e validações

# COMMAND ----------

if not TREINO_EXEC_RUN_ID:
    raise ValueError("❌ TREINO_EXEC_RUN_ID não preenchido.")
if not INPUT_TABLE_FQN or "..." in INPUT_TABLE_FQN:
    raise ValueError("❌ INPUT_TABLE_FQN não preenchido.")

ensure_schema(OUT_SCHEMA)
assert_table_exists(INPUT_TABLE_FQN)

if table_exists(OUTPUT_TABLE_FQN):
    raise ValueError(f"❌ Output já existe: {OUTPUT_TABLE_FQN}")

_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
while mlflow.active_run() is not None:
    mlflow.end_run()

client = MlflowClient()
treino_run_data = client.get_run(TREINO_EXEC_RUN_ID).data

# ── Resolver MODEL_IDS ────────────────────────────────────────────────────────
if MODEL_IDS:
    MODEL_IDS_USED = MODEL_IDS
    print("• model_ids       : preenchido manualmente →", MODEL_IDS_USED)
else:
    if "model_ids" not in treino_run_data.params:
        raise ValueError(f"❌ Param 'model_ids' não encontrado na run {TREINO_EXEC_RUN_ID}.")
    MODEL_IDS_USED = json.loads(treino_run_data.params["model_ids"])
    print("• model_ids       : lido do MLflow →", MODEL_IDS_USED)

# ── Resolver TREINO_FEATURE_COLS ──────────────────────────────────────────────
if TREINO_FEATURE_COLS:
    FEATURE_COLS_USED = TREINO_FEATURE_COLS
    print("• feature_cols    : preenchido manualmente →", FEATURE_COLS_USED)
else:
    if "feature_cols" not in treino_run_data.params:
        raise ValueError(f"❌ Param 'feature_cols' não encontrado na run {TREINO_EXEC_RUN_ID}.")
    FEATURE_COLS_USED = json.loads(treino_run_data.params["feature_cols"])
    print("• feature_cols    : lido do MLflow →", FEATURE_COLS_USED)

treino_cat_cols = [c for c in FEATURE_COLS_USED if c in FS_CAT_COLS]
treino_num_cols = [c for c in FEATURE_COLS_USED if c in FS_DECIMAL_COLS + FS_DIAS_COLS]

print("• treino_cat_cols :", treino_cat_cols)
print("• treino_num_cols :", treino_num_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abre runs MLflow

# COMMAND ----------

if PR_INF_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=PR_INF_RUN_ID_OVERRIDE)
    PR_INF_RUN_ID = PR_INF_RUN_ID_OVERRIDE
    _pr_status = "acoplada (override)"
else:
    mlflow.start_run(run_name=PR_INF_NAME)
    PR_INF_RUN_ID = mlflow.active_run().info.run_id
    _pr_status = "nova"

RUN_INF_EXEC = run_name_vts(f"T_INF_MODE_{MODE_CODE}")
mlflow.start_run(run_name=RUN_INF_EXEC, nested=True)
INF_EXEC_RUN_ID = mlflow.active_run().info.run_id

mlflow.set_tags({
    "pipeline_tipo": "T", "stage": "INFERENCIA", "run_role": "exec",
    "mode": MODE_CODE, "inf_versao": INF_VERSAO, "versao_ref": VERSAO_REF,
    "seg_target": SEG_TARGET,
})
mlflow.log_params({
    "ts_exec":             TS_EXEC,
    "inf_versao":          INF_VERSAO,
    "mode_code":           MODE_CODE,
    "seg_target":          SEG_TARGET,
    "pr_run_id":           PR_INF_RUN_ID,
    "treino_exec_run_id":  TREINO_EXEC_RUN_ID,
    "model_ids_inferred":  json.dumps(MODEL_IDS_USED),
    "input_table_fqn":     INPUT_TABLE_FQN,
    "output_table_fqn":    OUTPUT_TABLE_FQN,
    "feature_cols":        json.dumps(FEATURE_COLS_USED),
    "high_card_threshold": HIGH_CARD_THRESHOLD,
    "high_card_top_n":     HIGH_CARD_TOP_N,
    "outros_label":        OUTROS_LABEL,
})

print("✅ Runs abertas")
print("• PR  :", PR_INF_NAME,  "| run_id:", PR_INF_RUN_ID,  f"({_pr_status})")
print("• EXEC:", RUN_INF_EXEC, "| run_id:", INF_EXEC_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento via MLflow
# MAGIC
# MAGIC `top_vals_by_col` é lido do artifact store do treino.
# MAGIC Modelos e pipelines são carregados por `model_id` dentro do loop de scoring.
# MAGIC Não há reconstrução de pipeline a partir de `DF_MODEL_FQN` — o pipeline fitado
# MAGIC no treino é a fonte de verdade.

# COMMAND ----------

# ── Carregar top_vals_by_col do artifact store do treino ─────────────────────
with tempfile.TemporaryDirectory() as td:
    local_path = client.download_artifacts(TREINO_EXEC_RUN_ID, "preprocess/top_vals_by_col.json", td)
    with open(local_path) as f:
        top_vals_by_col = json.load(f)

# Republica como artifact da run de inferência (rastreabilidade)
mlflow.log_dict(top_vals_by_col, "preprocess/top_vals_by_col.json")

print("✅ top_vals_by_col carregado")
print("• colunas truncadas no treino:", list(top_vals_by_col.keys()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pré-processamento e scoring por modelo

# COMMAND ----------

# ── [1] Carga e preparação do input ──────────────────────────────────────────
df_in  = spark.table(INPUT_TABLE_FQN)
df_seg = df_in.filter(F.col(SEG_COL) == F.lit(SEG_TARGET)).cache()

n_input = int(df_seg.count())
mlflow.log_dict(profile_basic(df_seg, "input_filtered"), "profiling_input.json")
mlflow.log_metric("n_rows_input", n_input)

if n_input == 0:
    raise ValueError(f"❌ Nenhuma linha para SEG_TARGET='{SEG_TARGET}' em {INPUT_TABLE_FQN}")

# Verificar colunas obrigatórias
missing_id   = [c for c in ID_COLS if c not in df_seg.columns]
missing_feat = [c for c in FEATURE_COLS_USED if c not in df_seg.columns]
if missing_id:
    raise ValueError(f"❌ IDs faltando no input: {missing_id}")
if missing_feat:
    raise ValueError(f"❌ Features faltando no input: {missing_feat}")

# Limpeza: blank → null e cast numérico
df_inf_prep = blanks_to_null(df_seg, treino_cat_cols)
df_inf_prep = cast_to_double(df_inf_prep, treino_num_cols)

# Truncagem de cardinalidade com top_vals do treino
for c, top_vals in top_vals_by_col.items():
    if c in df_inf_prep.columns:
        df_inf_prep = apply_truncation(df_inf_prep, c, top_vals, OUTROS_LABEL)

# Selecionar colunas relevantes para o scoring
_keep_cols = list(dict.fromkeys(
    ID_COLS
    + ([STATUS_COL] if STATUS_COL in df_inf_prep.columns else [])
    + ([LABEL_COL]  if LABEL_COL  in df_inf_prep.columns else [])
    + FEATURE_COLS_USED
))
df_inf_prep = df_inf_prep.select(*_keep_cols).cache()

print("✅ Input preparado")
print("• n_rows           :", n_input)
print("• colunas truncadas:", [c for c in top_vals_by_col if c in df_inf_prep.columns])

# ── [2] Loop de scoring por model_id ─────────────────────────────────────────
# Acumula um DataFrame long (uma linha por cotação × model_id).
# pred_emitida NÃO é gerada aqui — thresholds calculados no 5_COMP_MODE_C.

df_scored_all     = None
score_profile_log = {}
fold_metrics_log  = []

for model_id in MODEL_IDS_USED:
    print(f"\n  → Scoring: {model_id}")

    # Carrega modelo e pipeline do artifact store do treino
    gbt_model = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/model")
    pp_fit    = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/preprocess_pipeline")

    # Vetorização
    df_for_model  = df_inf_prep.select(*FEATURE_COLS_USED).cache()
    df_vectorized = pp_fit.transform(df_for_model).select("features_vec").cache()

    # Scoring
    df_pred  = gbt_model.transform(df_vectorized)
    p1_col   = vector_to_array(F.col("probability")).getItem(1).cast("double")
    df_scores = df_pred.select(p1_col.alias("p_emitida"))

    # Rank global por score desc (dentro do model_id)
    w_rank = Window.orderBy(F.desc("p_emitida"))
    df_scores = df_scores.withColumn("rank_global", F.row_number().over(w_rank))

    # Reattach IDs e payload original via índice de posição (zipWithIndex → join)
    # Alternativa mais robusta: usar monotonically_increasing_id nas duas DFs antes do join
    df_inf_with_idx   = df_inf_prep.withColumn("_row_idx", F.monotonically_increasing_id())
    df_scores_with_idx = df_scores.withColumn("_row_idx", F.monotonically_increasing_id())
    df_joined = df_inf_with_idx.join(df_scores_with_idx, on="_row_idx", how="inner").drop("_row_idx")

    # Adicionar metadados do modelo
    df_model_scored = (
        df_joined
        .withColumn("model_id",          F.lit(model_id))
        .withColumn("treino_exec_run_id", F.lit(TREINO_EXEC_RUN_ID))
        .withColumn("inf_versao",         F.lit(INF_VERSAO))
        .withColumn("mode_code",          F.lit(MODE_CODE))
        .withColumn("seg_inferida",       F.lit(SEG_TARGET))
        .withColumn("inference_ts",       F.lit(TS_EXEC))
        .drop(*FEATURE_COLS_USED)   # remover features do output — payload de IDs + scores é suficiente
    )

    # Acumular
    if df_scored_all is None:
        df_scored_all = df_model_scored
    else:
        df_scored_all = df_scored_all.union(df_model_scored)

    # Perfil de score por model_id
    n_scored    = int(df_model_scored.count())
    n_null_p    = int(df_model_scored.filter(F.col("p_emitida").isNull()).count())
    score_prof  = profile_score(df_model_scored, "p_emitida")
    score_profile_log[model_id] = {**score_prof, "n_null_p_emitida": n_null_p}
    fold_metrics_log.append({"model_id": model_id, "n_rows": n_scored, "n_null_p_emitida": n_null_p})

    mlflow.log_metrics({
        f"n_rows_per_model_{model_id}":   n_scored,
        f"n_null_p_emitida_{model_id}":   n_null_p,
    })

    df_for_model.unpersist()
    df_vectorized.unpersist()
    print(f"     n_rows={n_scored}  n_null_p={n_null_p}  score_mean={score_prof.get('mean', 'N/A'):.4f}")

# ── [3] Logs e métricas agregadas ─────────────────────────────────────────────
df_scored_all = df_scored_all.cache()
n_output      = int(df_scored_all.count())

mlflow.log_dict(profile_basic(df_scored_all, "output_long_format"), "profiling_output.json")
mlflow.log_dict(score_profile_log, "score_profile_per_model.json")
mlflow.log_metrics({
    "n_rows_output":   n_output,
    "n_models_scored": len(MODEL_IDS_USED),
})

print("\n✅ Scoring concluído")
print("• n_rows output (long format):", n_output)
print("• modelos:", MODEL_IDS_USED)
df_scored_all.orderBy("model_id", F.asc("rank_global")).display(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Salva output e encerra runs

# COMMAND ----------

df_scored_all.write.format("delta").mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_FQN)
mlflow.log_metric("output_saved", 1)

print("✅ Output salvo:", OUTPUT_TABLE_FQN)

df_seg.unpersist()
df_inf_prep.unpersist()
df_scored_all.unpersist()

while mlflow.active_run() is not None:
    mlflow.end_run()

print("✅ Runs encerradas")
print("\n• output table     :", OUTPUT_TABLE_FQN)
print("• modelos inferidos:", MODEL_IDS_USED)
print("\n⚠️  Anotar OUTPUT_TABLE_FQN para uso no 5_COMP_MODE_C:")
print(f"    INFERENCIA_TABLE_FQN = \"{OUTPUT_TABLE_FQN}\"")
