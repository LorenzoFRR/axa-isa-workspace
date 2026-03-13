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
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"

PR_INF_NAME = "T_PR_INFERENCIA"
MODE_CODE   = "B"
INF_VERSAO  = "V8"
VERSAO_REF  = INF_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# Override: preencha para reutilizar container PR existente
PR_INF_RUN_ID_OVERRIDE = 'e7af2dc5cb8b45c194656889f4b28fd2'

# =========================
# MODELO — run do T_TREINO onde o GBT foi salvo
# =========================
TREINO_EXEC_RUN_ID  = "2ef6e6424b614fea974acc71039713be"   # <<< RUN_ID do exec run T_TREINO (run_role=exec, step=TREINO)
MODEL_ARTIFACT_PATH = "treino_final/model"

# Salvar pp_final_fit reconstruído de volta na run de treino
SAVE_PP_TO_TREINO_RUN      = False
PP_ARTIFACT_PATH           = "treino_final/preprocess_pipeline"

# =========================
# RECONSTRUÇÃO DO PRÉ-PROCESSAMENTO
# pp_final_fit é reconstruído a partir dos dados de treino (DF_MODEL_FQN),
# usando os mesmos parâmetros, produzindo pipeline idêntico ao do treino.
# =========================
DF_MODEL_FQN = "gold.cotacao_model_20260308_210014_0b5d9356"   # <<< tabela de treino (ex.: gold.cotacao_model_20260308_...)

SEG_TARGET = "SEGURO_NOVO_MANUAL"   # <<< AJUSTE
SEG_COL    = "SEG"
ID_COL     = "CD_NUMERO_COTACAO_AXA"
LABEL_COL  = "label"
DATE_COL   = "DATA_COTACAO"
ID_COLS    = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]

# Features usadas no treino — preencher com o param 'feature_cols' logado no MLflow
TREINO_FEATURE_COLS = ["DS_PRODUTO_NOME", "DIAS_ULTIMA_ATUALIZACAO", "DIAS_INICIO_VIGENCIA", "VL_PREMIO_LIQUIDO", "DIAS_FIM_ANALISE_SUBSCRICAO", "VL_PRE_TOTAL", "DIAS_ANALISE_SUBSCRICAO"]

# Listas de tipo de coluna — mesmas do 3_ML_TREINO_MODE_B_1
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

# Parâmetros de truncagem de cardinalidade — mesmos do treino
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"

# =========================
# THRESHOLD
# =========================
THRESHOLD_MODE  = "mlflow"   # "manual" | "mlflow"
# "manual"  → usa THRESHOLD_VALUE diretamente
# "mlflow"  → lê threshold_operacional do param TREINO_EXEC_RUN_ID
THRESHOLD_VALUE = 0.50

# =========================
# INPUT — dados novos para scoring
# =========================
INPUT_TABLE_FQN = "gold.cotacao_validacao_20260308_210014_0b5d9356"

# =========================
# OUTPUT
# =========================
OUT_SCHEMA       = "gold"
OUTPUT_TABLE_FQN = f"{OUT_SCHEMA}.cotacao_inferencia_mode_b_{TS_EXEC}"
WRITE_MODE       = "overwrite"

print("✅ CONFIG INFERÊNCIA MODE_B carregada")
print("• input table     :", INPUT_TABLE_FQN)
print("• df_model_fqn    :", DF_MODEL_FQN)
print("• treino_exec_run :", TREINO_EXEC_RUN_ID)
print("• seg_target      :", SEG_TARGET)
print("• features        :", TREINO_FEATURE_COLS)
print("• output table    :", OUTPUT_TABLE_FQN)

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
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
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

def profile_score(df: DataFrame, score_col: str) -> dict:
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
        "n": int(agg["n"]), "n_null": int(agg["n_null"]),
        "min":  float(agg["min"])  if agg["min"]  is not None else None,
        "mean": float(agg["mean"]) if agg["mean"] is not None else None,
        "max":  float(agg["max"])  if agg["max"]  is not None else None,
    }


# ── Helpers de pré-processamento (mesmos do treino) ───────────────────────────

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

def compute_top_vals(df: DataFrame, col_name: str, top_n: int) -> list:
    """Computa as top_n categorias mais frequentes de uma coluna (sem nulos/brancos)."""
    return (
        df.filter(F.col(col_name).isNotNull() & (F.trim(F.col(col_name)) != ""))
          .groupBy(col_name).count()
          .orderBy(F.col("count").desc())
          .limit(top_n)
          .select(col_name)
          .rdd.flatMap(lambda x: x)
          .collect()
    )

def apply_truncation(df: DataFrame, col_name: str, top_vals: list, outros: str = "OUTROS") -> DataFrame:
    """Aplica truncagem de cardinalidade usando top_vals pré-computados."""
    return df.withColumn(
        col_name,
        F.when(F.col(col_name).isin(top_vals), F.col(col_name)).otherwise(F.lit(outros)),
    )

def build_preprocess_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    idx_cols = [f"{c}__idx" for c in cat_cols]
    ohe_cols = [f"{c}__ohe" for c in cat_cols]
    imp_cols = [f"{c}__imp" for c in num_cols]
    stages = []
    for c, out_c in zip(cat_cols, idx_cols):
        stages.append(StringIndexer(inputCol=c, outputCol=out_c, handleInvalid="keep"))
    if idx_cols:
        stages.append(OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, dropLast=False))
    if num_cols:
        stages.append(Imputer(inputCols=num_cols, outputCols=imp_cols, strategy="mean"))
    stages.append(VectorAssembler(inputCols=ohe_cols + imp_cols, outputCol="features_vec"))
    return Pipeline(stages=stages)


print("✅ Helpers carregados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparos MLflow e validações

# COMMAND ----------

if not TREINO_EXEC_RUN_ID:
    raise ValueError("❌ TREINO_EXEC_RUN_ID não preenchido.")
if not DF_MODEL_FQN:
    raise ValueError("❌ DF_MODEL_FQN não preenchido.")
if not INPUT_TABLE_FQN:
    raise ValueError("❌ INPUT_TABLE_FQN não preenchido.")
if not TREINO_FEATURE_COLS:
    raise ValueError("❌ TREINO_FEATURE_COLS não preenchido.")

ensure_schema(OUT_SCHEMA)
assert_table_exists(DF_MODEL_FQN)
assert_table_exists(INPUT_TABLE_FQN)

if table_exists(OUTPUT_TABLE_FQN):
    raise ValueError(f"❌ Output já existe: {OUTPUT_TABLE_FQN}")

_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
while mlflow.active_run() is not None:
    mlflow.end_run()

client = MlflowClient()

# Resolve threshold
if THRESHOLD_MODE == "manual":
    THRESHOLD_USED = float(THRESHOLD_VALUE)
elif THRESHOLD_MODE == "mlflow":
    run_params = client.get_run(TREINO_EXEC_RUN_ID).data.params
    if "threshold_operacional" not in run_params:
        raise ValueError(f"❌ Param 'threshold_operacional' não encontrado na run {TREINO_EXEC_RUN_ID}.")
    THRESHOLD_USED = float(run_params["threshold_operacional"])
else:
    raise ValueError(f"THRESHOLD_MODE inválido: {THRESHOLD_MODE}")

MODEL_URI_USED = f"runs:/{TREINO_EXEC_RUN_ID}/{MODEL_ARTIFACT_PATH}"

print("✅ Preparos ok")
print("• model_uri      :", MODEL_URI_USED)
print("• threshold_mode :", THRESHOLD_MODE)
print("• threshold_used :", THRESHOLD_USED)

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
    "versao_ref":          VERSAO_REF,
    "seg_target":          SEG_TARGET,
    "pr_run_id":           PR_INF_RUN_ID,
    "input_table_fqn":     INPUT_TABLE_FQN,
    "df_model_fqn":        DF_MODEL_FQN,
    "output_table_fqn":    OUTPUT_TABLE_FQN,
    "treino_exec_run_id":  TREINO_EXEC_RUN_ID,
    "model_uri":           MODEL_URI_USED,
    "model_artifact_path": MODEL_ARTIFACT_PATH,
    "threshold_mode":      THRESHOLD_MODE,
    "threshold_used":      THRESHOLD_USED,
    "feature_cols":        json.dumps(TREINO_FEATURE_COLS),
    "high_card_threshold": HIGH_CARD_THRESHOLD,
    "high_card_top_n":     HIGH_CARD_TOP_N,
    "outros_label":        OUTROS_LABEL,
})

print("✅ Runs abertas")
print("• PR  :", PR_INF_NAME,   "| run_id:", PR_INF_RUN_ID,   f"({_pr_status})")
print("• EXEC:", RUN_INF_EXEC,  "| run_id:", INF_EXEC_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reconstrução do pipeline de pré-processamento
# MAGIC
# MAGIC Reconstrói `pp_final_fit` a partir dos dados de treino (`DF_MODEL_FQN`),
# MAGIC aplicando as mesmas transformações do `3_ML_TREINO_MODE_B_1`.
# MAGIC O pipeline resultante é idêntico ao original pois dados e parâmetros são os mesmos.

# COMMAND ----------

treino_cat_cols = [c for c in TREINO_FEATURE_COLS if c in FS_CAT_COLS]
treino_num_cols = [c for c in TREINO_FEATURE_COLS if c in FS_DECIMAL_COLS + FS_DIAS_COLS]

# ── [1] Carga e limpeza dos dados de treino ───────────────────────────────────
df_model_raw = spark.table(DF_MODEL_FQN)
df_model_seg = df_model_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET))

df_model_seg = blanks_to_null(df_model_seg, treino_cat_cols)
df_model_seg = cast_to_double(df_model_seg, treino_num_cols)

# ── [2] Truncagem de cardinalidade — computa top_vals a partir dos dados de treino
#        top_vals_by_col é reutilizado na inferência para garantir consistência
top_vals_by_col: dict = {}

for c in treino_cat_cols:
    card = int(df_model_seg.select(F.countDistinct(c)).collect()[0][0])
    if card > HIGH_CARD_THRESHOLD:
        top_vals = compute_top_vals(df_model_seg, c, HIGH_CARD_TOP_N)
        top_vals_by_col[c] = top_vals
        df_model_seg = apply_truncation(df_model_seg, c, top_vals, OUTROS_LABEL)

mlflow.log_dict(
    {c: vals for c, vals in top_vals_by_col.items()},
    "preprocess/top_vals_by_col.json",
)
print("• high_card_cols truncados:", list(top_vals_by_col.keys()))

# ── [3] Fit do pipeline de pré-processamento ──────────────────────────────────
df_model_ml = df_model_seg.select(*TREINO_FEATURE_COLS).cache()
pp_final_fit = build_preprocess_pipeline(treino_cat_cols, treino_num_cols).fit(df_model_ml)
df_model_ml.unpersist()

print("✅ pp_final_fit reconstruído")
print("• treino_cat_cols:", treino_cat_cols)
print("• treino_num_cols:", treino_num_cols)

# ── [4] Salva pipeline reconstruído na run de treino (opcional) ────────────────
if SAVE_PP_TO_TREINO_RUN:
    with mlflow.start_run(run_id=TREINO_EXEC_RUN_ID, nested=True):
        mlflow.spark.log_model(pp_final_fit, artifact_path=PP_ARTIFACT_PATH)
        mlflow.log_param("pp_reconstructed_from_inf_run", INF_EXEC_RUN_ID)
    print(f"✅ pp_final_fit salvo na run de treino ({TREINO_EXEC_RUN_ID}) → {PP_ARTIFACT_PATH}")
else:
    print("⚠️ SAVE_PP_TO_TREINO_RUN=False — pipeline não salvo no MLflow")

mlflow.log_param("pp_artifact_path", PP_ARTIFACT_PATH if SAVE_PP_TO_TREINO_RUN else "not_saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carrega modelo GBT

# COMMAND ----------

gbt_model = mlflow.spark.load_model(MODEL_URI_USED)
print("✅ GBT carregado:", MODEL_URI_USED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pré-processamento e scoring dos dados de inferência

# COMMAND ----------

# ── [1] Carga e filtragem ─────────────────────────────────────────────────────
df_in  = spark.table(INPUT_TABLE_FQN)

if SEG_COL not in df_in.columns:
    raise ValueError(f"❌ Coluna {SEG_COL} não existe em {INPUT_TABLE_FQN}")

df_seg = df_in.filter(F.col(SEG_COL) == F.lit(SEG_TARGET)).cache()
n_input = int(df_seg.count())
mlflow.log_dict(profile_basic(df_seg, "input_filtered"), "profiling_input.json")
mlflow.log_metric("n_rows_input_filtered", n_input)
print("• n_rows input :", n_input)

# ── [2] Mesma limpeza e truncagem — usando top_vals do treino ─────────────────
df_inf_prep = blanks_to_null(df_seg, treino_cat_cols)
df_inf_prep = cast_to_double(df_inf_prep, treino_num_cols)

for c, top_vals in top_vals_by_col.items():
    if c in df_inf_prep.columns:
        df_inf_prep = apply_truncation(df_inf_prep, c, top_vals, OUTROS_LABEL)

# ── [3] Verifica colunas necessárias ─────────────────────────────────────────
missing_id  = [c for c in ID_COLS if c not in df_seg.columns]
missing_feat = [c for c in TREINO_FEATURE_COLS if c not in df_inf_prep.columns]
if missing_id:
    raise ValueError(f"❌ IDs faltando no input: {missing_id}")
if missing_feat:
    raise ValueError(f"❌ Features faltando no input: {missing_feat}")

# ── [4] Vetorização e scoring ────────────────────────────────────────────────
df_for_model  = df_inf_prep.select(*ID_COLS, *TREINO_FEATURE_COLS).cache()
df_vectorized = pp_final_fit.transform(df_for_model).select(*ID_COLS, "features_vec").cache()
df_pred       = gbt_model.transform(df_vectorized)

p1_col = vector_to_array(F.col("probability")).getItem(1).cast("double")
df_scored_core = (
    df_pred.select(*ID_COLS, p1_col.alias("p_emitida"))
)

w_rank = Window.orderBy(F.desc("p_emitida"))
df_scored_core = df_scored_core.withColumn("rank_global", F.row_number().over(w_rank))

# ── [5] Join com payload original ─────────────────────────────────────────────
df_scored_final = (
    df_seg.join(df_scored_core, on=ID_COLS, how="inner")
    .withColumn("pred_emitida",    (F.col("p_emitida") >= F.lit(float(THRESHOLD_USED))).cast("int"))
    .withColumn("model_run_id",    F.lit(TREINO_EXEC_RUN_ID))
    .withColumn("model_uri",       F.lit(MODEL_URI_USED))
    .withColumn("inference_ts",    F.lit(TS_EXEC))
    .withColumn("threshold_used",  F.lit(float(THRESHOLD_USED)))
    .withColumn("inf_versao",      F.lit(INF_VERSAO))
    .withColumn("mode_code",       F.lit(MODE_CODE))
    .withColumn("seg_inferida",    F.lit(SEG_TARGET))
).cache()

mlflow.log_dict(profile_basic(df_scored_final, "output_scored"), "profiling_output.json")
mlflow.log_dict(profile_score(df_scored_final, "p_emitida"), "profiling_score.json")
mlflow.log_dict({"tables_lineage": {
    "stage": "INFERENCIA", "seg_target": SEG_TARGET, "ts_exec": TS_EXEC,
    "treino_exec_run_id": TREINO_EXEC_RUN_ID, "model_uri": MODEL_URI_USED,
    "inputs": {"input_table": INPUT_TABLE_FQN, "df_model_fqn": DF_MODEL_FQN},
    "outputs": {"output_table": OUTPUT_TABLE_FQN},
}}, "tables_lineage.json")

n_output     = int(df_scored_final.count())
n_pred_1     = int(df_scored_final.filter(F.col("pred_emitida") == 1).count())
mlflow.log_metrics({"n_rows_output": n_output, "n_pred_emitida_1": n_pred_1})

print("✅ Scoring ok")
print("• n_rows output  :", n_output)
print("• n_pred_emitida :", n_pred_1)
df_scored_final.orderBy(F.asc("rank_global")).display(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Salva output e encerra runs

# COMMAND ----------

df_scored_final.write.mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_FQN)
mlflow.log_metric("output_saved", 1)

print("✅ Output salvo:", OUTPUT_TABLE_FQN)

df_for_model.unpersist()
df_vectorized.unpersist()
df_scored_final.unpersist()
df_seg.unpersist()

while mlflow.active_run() is not None:
    mlflow.end_run()

print("✅ Runs encerradas")
print("• output :", OUTPUT_TABLE_FQN)
print("• threshold_used :", THRESHOLD_USED)

# COMMAND ----------

