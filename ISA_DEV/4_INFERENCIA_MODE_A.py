# Databricks notebook source
x = 1

# COMMAND ----------

# DBTITLE 1,Cell 1
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

# =========================
# MLflow
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"

PR_INF_NAME = "T_PR_INFERENCIA"      # container (sem logging)

INF_VERSAO = "V6"
VERSAO_REF = INF_VERSAO
TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# =========================
# Override: preencha para reutilizar container PR existente
# =========================
PR_INF_RUN_ID_OVERRIDE = None   # ex.: "abc123..." | None = cria novo container

# =========================
# INPUT: tabela para inferência (manual)
# =========================
INPUT_TABLE_FQN = "gold.cotacao_validacao_20260307_231536_1c543b4f"   # <<< AJUSTE
SEG_TARGET = "SEGURO_NOVO_MANUAL"                                           # <<< AJUSTE

# IDs usados para join payload + score
ID_COL = "CD_NUMERO_COTACAO_AXA"
SEG_COL = "SEG"
DATE_COL = "DATA_COTACAO"
ID_COLS = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]

# =========================
# MODELO: referenciar run/model anterior
# =========================
MODEL_RUN_ID = "a3dcde0164504429acdb0c69304a0279"   # <<< AJUSTE
MODEL_ARTIFACT_PATH = "model"                        # <<< AJUSTE
MODEL_URI = None                                     # opcional: "runs:/<run_id>/<artifact_path>"

# Threshold
THRESHOLD_MODE = "manual"  # "manual" | "artifact"
THRESHOLD_VALUE = 0.50
THRESHOLD_ARTIFACT_PATH = "threshold/threshold_metrics.json"  # se THRESHOLD_MODE="artifact"

# =========================
# OUTPUT
# =========================
OUT_SCHEMA = "gold"
OUTPUT_TABLE_FQN = f"{OUT_SCHEMA}.cotacao_inferencia_{TS_EXEC}"
WRITE_MODE = "overwrite"

print("✅ CONFIG INFERÊNCIA carregada")
print("• input table :", INPUT_TABLE_FQN)
print("• SEG_TARGET  :", SEG_TARGET)
print("• output table:", OUTPUT_TABLE_FQN)

# COMMAND ----------

import os, json, tempfile
import mlflow
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.functions import vector_to_array
from mlflow.tracking import MlflowClient

# -------------------------
# Metastore helpers
# -------------------------
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

# -------------------------
# Transform logging (catalog + execution)
# -------------------------
def log_dict_as_artifact(d: dict, path: str):
    mlflow.log_dict(d, path)

def profile_basic(df: DataFrame, name: str) -> dict:
    n_rows = df.count()
    n_cols = len(df.columns)
    return {"name": name, "n_rows": int(n_rows), "n_cols": int(n_cols)}

def profile_score(df: DataFrame, score_col: str) -> dict:
    if score_col not in df.columns:
        return {"score_col": score_col, "ok": False, "reason": "missing_col"}
    agg = (df.agg(
        F.count(F.lit(1)).alias("n"),
        F.sum(F.when(F.col(score_col).isNull(), 1).otherwise(0)).alias("n_score_null"),
        F.min(score_col).alias("min"),
        F.avg(score_col).alias("mean"),
        F.max(score_col).alias("max"),
    ).collect()[0])
    return {
        "score_col": score_col,
        "ok": True,
        "n": int(agg["n"]),
        "n_score_null": int(agg["n_score_null"]),
        "min": float(agg["min"]) if agg["min"] is not None else None,
        "mean": float(agg["mean"]) if agg["mean"] is not None else None,
        "max": float(agg["max"]) if agg["max"] is not None else None,
    }

# -------------------------
# Limpeza/casts (igual padrão do V3)
# -------------------------
def blanks_to_null_trim(df: DataFrame, cols: list) -> DataFrame:
    out = df
    for c in cols:
        if c in out.columns:
            out = out.withColumn(c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c)))
    return out

def cast_to_double(df: DataFrame, cols: list) -> DataFrame:
    out = df
    for c in cols:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast("double"))
    return out

def cast_str_to_int_sanitized(df: DataFrame, cols: list) -> DataFrame:
    out = df
    for c in cols:
        if c in out.columns:
            s = F.trim(F.col(c).cast("string"))
            s = F.when(s.isNull() | (s == ""), F.lit(None)).otherwise(s)
            out = out.withColumn(
                c,
                F.when(s.rlike(r"^-?\d+$"), s.cast("int"))
                 .when(s.rlike(r"^-?\d+(\.\d+)?$"), s.cast("double").cast("int"))
                 .otherwise(F.lit(None).cast("int"))
            )
    return out

# -------------------------
# Inferir colunas raw do pipeline model (como no V3)
# -------------------------
def infer_model_raw_input_cols(pipeline_model):
    cat_cols = set()
    num_cols = set()
    raw = set()

    for st in pipeline_model.stages:
        name = st.__class__.__name__

        # StringIndexerModel => categorical raw col
        if hasattr(st, "getInputCol") and name.endswith("StringIndexerModel"):
            c = st.getInputCol()
            raw.add(c)
            cat_cols.add(c)

        # ImputerModel => numeric raw cols
        if hasattr(st, "getInputCols") and name.endswith("ImputerModel"):
            cols = list(st.getInputCols())
            raw.update(cols)
            num_cols.update(cols)

    return sorted(raw), sorted(cat_cols), sorted(num_cols)

print("✅ Helpers carregados (inferência)")

# COMMAND ----------

# =========================
# Preparos
# =========================
ensure_schema(OUT_SCHEMA)
assert_table_exists(INPUT_TABLE_FQN)

if table_exists(OUTPUT_TABLE_FQN):
    raise ValueError(f"❌ Output já existe: {OUTPUT_TABLE_FQN} (timestamp repetido?)")

_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

# encerra qualquer run pendurada no runtime atual
while mlflow.active_run() is not None:
    mlflow.end_run()

# =========================
# PR container (sem logging)
# =========================
if PR_INF_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=PR_INF_RUN_ID_OVERRIDE)
    PR_INF_RUN_ID = PR_INF_RUN_ID_OVERRIDE
    _pr_status = "acoplada (override)"
else:
    mlflow.start_run(run_name=PR_INF_NAME)
    PR_INF_RUN_ID = mlflow.active_run().info.run_id
    _pr_status = "nova"

# =========================
# Exec run versionada (loga tudo aqui)
# =========================
RUN_INF_EXEC = run_name_vts("T_INF")
mlflow.start_run(run_name=RUN_INF_EXEC, nested=True)
INF_EXEC_RUN_ID = mlflow.active_run().info.run_id

print("✅ Runs abertas")
print("• PR  :", PR_INF_NAME, "| run_id:", PR_INF_RUN_ID, f"({_pr_status})")
print("• EXEC:", RUN_INF_EXEC, "| run_id:", INF_EXEC_RUN_ID)

# COMMAND ----------

client = MlflowClient()

# -------------------------
# Resolve model URI
# -------------------------
if MODEL_URI is None:
    if not MODEL_RUN_ID:
        raise ValueError("❌ Defina MODEL_RUN_ID (ou MODEL_URI).")
    MODEL_URI_USED = f"runs:/{MODEL_RUN_ID}/{MODEL_ARTIFACT_PATH}"
else:
    MODEL_URI_USED = MODEL_URI

# Threshold
THRESHOLD_USED = None
if THRESHOLD_MODE == "manual":
    THRESHOLD_USED = float(THRESHOLD_VALUE)
elif THRESHOLD_MODE == "artifact":
    if not MODEL_RUN_ID:
        raise ValueError("❌ THRESHOLD_MODE='artifact' requer MODEL_RUN_ID.")
    local = mlflow.artifacts.download_artifacts(run_id=MODEL_RUN_ID, artifact_path=THRESHOLD_ARTIFACT_PATH)
    with open(local, "r", encoding="utf-8") as f:
        js = json.load(f)
    # aceita formatos diferentes; tenta achar um threshold recomendado
    THRESHOLD_USED = js.get("recommended_threshold") or js.get("thr_precision_target") or js.get("threshold") or js.get("threshold_used")
    if THRESHOLD_USED is None:
        raise ValueError(f"❌ Não encontrei threshold no artifact: {THRESHOLD_ARTIFACT_PATH}")
    THRESHOLD_USED = float(THRESHOLD_USED)
else:
    raise ValueError(f"THRESHOLD_MODE inválido: {THRESHOLD_MODE}")

# -------------------------
# LOGGING (somente exec run)
# -------------------------
mlflow.set_tag("stage", "INFERENCIA")
mlflow.set_tag("run_role", "exec")
mlflow.set_tag("seg_target", SEG_TARGET)
mlflow.set_tag("versao_ref", VERSAO_REF)

mlflow.log_param("ts_exec", TS_EXEC)
mlflow.log_param("inf_versao", INF_VERSAO)
mlflow.log_param("versao_ref", VERSAO_REF)
mlflow.log_param("run_suffix", RUN_SUFFIX)
mlflow.log_param("seg_target", SEG_TARGET)

mlflow.log_param("pr_run_id", PR_INF_RUN_ID)
mlflow.log_param("input_table_fqn", INPUT_TABLE_FQN)
mlflow.log_param("output_table_fqn", OUTPUT_TABLE_FQN)

mlflow.log_param("model_uri", MODEL_URI_USED)
mlflow.log_param("model_run_id", MODEL_RUN_ID if MODEL_RUN_ID else None)
mlflow.log_param("threshold_mode", THRESHOLD_MODE)
mlflow.log_param("threshold_used", float(THRESHOLD_USED))

print(f"[INF] SEG_TARGET  = {SEG_TARGET}")
print(f"[INF] MODEL_URI   = {MODEL_URI_USED}")
print(f"[INF] THRESHOLD   = {THRESHOLD_USED}")

# -------------------------
# Lineage artifact
# -------------------------
tables_lineage = {
    "stage": "INFERENCIA",
    "seg_target": SEG_TARGET,
    "ts_exec": TS_EXEC,
    "model_uri": MODEL_URI_USED,
    "threshold_used": THRESHOLD_USED,
    "inputs": {"input_table": INPUT_TABLE_FQN},
    "outputs": {"output_table": OUTPUT_TABLE_FQN},
}
mlflow.log_dict(tables_lineage, "tables_lineage.json")

# -------------------------
# Load input + filter SEG
# -------------------------
df_in = spark.table(INPUT_TABLE_FQN)

if SEG_COL not in df_in.columns:
    raise ValueError(f"❌ Coluna {SEG_COL} não existe em {INPUT_TABLE_FQN}")

df_seg = df_in.filter(F.col(SEG_COL) == F.lit(SEG_TARGET)).cache()

mlflow.log_dict(profile_basic(df_seg, "input_filtered"), "profiling_input.json")
mlflow.log_metric("n_rows_input_filtered", int(df_seg.count()))

# -------------------------
# Load model (pipeline)
# -------------------------
winner_model = mlflow.spark.load_model(MODEL_URI_USED)

# infer raw columns
WINNER_FEATURE_COLS, CAT_FEATURES, NUM_DOUBLE_FEATURES = infer_model_raw_input_cols(winner_model)
if len(WINNER_FEATURE_COLS) == 0:
    raise RuntimeError("❌ Não consegui inferir colunas raw do modelo pelo pipeline (StringIndexer/Imputer).")

mlflow.log_dict(
    {
        "winner_feature_cols": WINNER_FEATURE_COLS,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_DOUBLE_FEATURES,
    },
    "model/model_input_cols.json"
)

print("[MODEL INPUT] raw cols:", WINNER_FEATURE_COLS)

# -------------------------
# Monta payload completo (mantém colunas originais)
# e subset para modelo (IDs + raw inputs)
# -------------------------
missing_id = [c for c in ID_COLS if c not in df_seg.columns]
if missing_id:
    raise ValueError(f"❌ IDs faltando no input: {missing_id}")

df_prep_all = df_seg.cache()

# cria colunas faltantes para o modelo (se alguma raw col não existe)
missing_for_model = [c for c in WINNER_FEATURE_COLS if c not in df_prep_all.columns]
df_for_model = df_prep_all
for c in missing_for_model:
    # heurística: se é cat => string; se é num => double
    if c in CAT_FEATURES:
        df_for_model = df_for_model.withColumn(c, F.lit(None).cast("string"))
    else:
        df_for_model = df_for_model.withColumn(c, F.lit(None).cast("double"))

mlflow.log_dict({"missing_for_model": missing_for_model}, "model/missing_for_model.json")

df_for_model = df_for_model.select(*ID_COLS, *WINNER_FEATURE_COLS)

# limpeza/casts coerentes
df_for_model = blanks_to_null_trim(df_for_model, [c for c in WINNER_FEATURE_COLS if c in CAT_FEATURES])
df_for_model = cast_to_double(df_for_model, [c for c in WINNER_FEATURE_COLS if c in NUM_DOUBLE_FEATURES])

# (opcional) se houver cols QTD_* vindo como string, você pode ativar:
# df_for_model = cast_str_to_int_sanitized(df_for_model, [c for c in WINNER_FEATURE_COLS if c.startswith("QTD_")])

# -------------------------
# Inferência
# -------------------------
pred = winner_model.transform(df_for_model)

df_scored_core = (
    pred.select(
        *ID_COLS,
        vector_to_array(F.col("probability")).getItem(1).cast("double").alias("p_emitida")
    )
)

w_rank = Window.orderBy(F.desc("p_emitida"))
df_scored_core = df_scored_core.withColumn("rank_global", F.row_number().over(w_rank))

# join payload + score
df_scored_final = (
    df_prep_all.join(df_scored_core, on=ID_COLS, how="inner")
    .withColumn("pred_emitida", (F.col("p_emitida") >= F.lit(float(THRESHOLD_USED))).cast("int"))
    .withColumn("model_run_id", F.lit(MODEL_RUN_ID if MODEL_RUN_ID else ""))
    .withColumn("model_uri", F.lit(MODEL_URI_USED))
    .withColumn("inference_ts", F.lit(TS_EXEC))
    .withColumn("threshold_used", F.lit(float(THRESHOLD_USED)))
    .withColumn("seg_inferida", F.lit(SEG_TARGET))
).cache()

mlflow.log_dict(profile_basic(df_scored_final, "output_scored"), "profiling_output.json")
mlflow.log_dict(profile_score(df_scored_final, "p_emitida"), "profiling_score.json")

mlflow.log_metric("n_rows_output", int(df_scored_final.count()))
mlflow.log_metric("n_pred_emitida_1", int(df_scored_final.filter(F.col("pred_emitida")==1).count()))

print("[OK] scored rows:", df_scored_final.count())
df_scored_final.orderBy(F.asc("rank_global")).display(10)

# COMMAND ----------

# salva output
df_scored_final.write.mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_FQN)

mlflow.log_metric("output_saved", 1)

print("✅ Output salvo:", OUTPUT_TABLE_FQN)

# encerra EXEC e PR
while mlflow.active_run() is not None:
    mlflow.end_run()

print("✅ Runs encerradas (T_INF + T_PR_INFERENCIA)")