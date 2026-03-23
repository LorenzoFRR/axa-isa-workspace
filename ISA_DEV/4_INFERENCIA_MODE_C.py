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
INF_VERSAO  = "V10.0.0"
VERSAO_REF  = INF_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# Override: preencha para reutilizar container PR existente
PR_INF_RUN_ID_OVERRIDE = "e7af2dc5cb8b45c194656889f4b28fd2"

# =========================
# REFERÊNCIA AO TREINO
# =========================
# run_id do exec run T_TREINO (run_role=exec, step=TREINO) — impresso no final do 3_TREINO_MODE_C
TREINO_EXEC_RUN_ID = "1c516039c9af476397d28a8cddd99674"   # <<< OBRIGATÓRIO

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
SEG_SLUG   = SEG_TARGET.lower()     # ex: "seguro_novo_manual"

ID_COLS = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]

# =========================
# FEATURES
# =========================
# Preencher com o param 'feature_cols' logado na run TREINO_EXEC_RUN_ID.
# Se vazio ([]), será lido automaticamente do MLflow.
TREINO_FEATURE_COLS = []   # <<< AJUSTE ou deixar vazio para ler do MLflow

OUTROS_LABEL = "OUTROS"   # constante usada na truncagem; deve coincidir com o treino

# =========================
# INPUT / OUTPUT
# =========================
# df_validacao gerado pelo 3_TREINO_MODE_C (logado como param 'df_validacao_fqn')
INPUT_TABLE_FQN = "gold.cotacao_validacao_20260319_195931_cdc3278a"   # <<< AJUSTE

OUT_SCHEMA       = "gold"
OUTPUT_TABLE_FQN = f"{OUT_SCHEMA}.cotacao_inferencia_mode_{MODE_CODE.lower()}_{SEG_SLUG}_{TS_EXEC}"
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

for _p in ("treino_cat_cols", "treino_num_cols"):
    if _p not in treino_run_data.params:
        raise ValueError(f"❌ Param '{_p}' não encontrado na run {TREINO_EXEC_RUN_ID}.")

treino_cat_cols = json.loads(treino_run_data.params["treino_cat_cols"])
treino_num_cols = json.loads(treino_run_data.params["treino_num_cols"])

print("• treino_cat_cols : lido do MLflow →", treino_cat_cols)
print("• treino_num_cols : lido do MLflow →", treino_num_cols)

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

RUN_INF_EXEC = run_name_vts("T_INF")
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
# Skipa colunas numéricas: QTD_* e similares chegam como string no schema da tabela
# e são classificadas como cat em T_PRE_PROC_MODEL, mas em T_TREINO são castadas para
# double e entram em treino_num_cols. Aplicar apply_truncation nelas reconverteria de
# double para string (F.lit("OUTROS")), quebrando pp_fit.transform().
for c, top_vals in top_vals_by_col.items():
    if c in df_inf_prep.columns and c not in treino_num_cols:
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

# ── [2] Loop de scoring por model_id — formato wide (uma linha por cotação) ───
# Colunas por model_id: p_emitida_{model_id}, rank_global_{model_id}
# Metadados compartilhados (mesmos para todos os modelos da execução):
#   treino_exec_run_id, inf_versao, mode_code, seg_inferida, inference_ts

# df_wide parte do input completo — os scores são adicionados via join
df_wide           = df_seg
score_profile_log = {}

for model_id in MODEL_IDS_USED:
    print(f"\n  → Scoring: {model_id}")

    # Carregar modelo e pipeline do artifact store do treino
    gbt_model = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/model")
    pp_fit    = mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/preprocess_pipeline")

    # Vetorização — ID_COL incluído para join após scoring
    df_for_model  = df_inf_prep.select(ID_COL, *FEATURE_COLS_USED).cache()
    df_vectorized = pp_fit.transform(df_for_model).select(ID_COL, "features_vec").cache()

    # Scoring + extração de p_emitida e rank global
    p_col = f"p_emitida_{model_id}"
    r_col = f"rank_global_{model_id}"

    df_pred = gbt_model.transform(df_vectorized)
    df_model_scores = (
        df_pred
        .withColumn(p_col, vector_to_array(F.col("probability")).getItem(1).cast("double"))
        .withColumn(r_col, F.row_number().over(Window.orderBy(F.desc(p_col))))
        .select(ID_COL, p_col, r_col)
    )

    # Join lateral — mantém uma linha por cotação
    df_wide = df_wide.join(df_model_scores, on=ID_COL, how="left")

    # Perfil de score por model_id
    n_scored   = int(df_model_scores.count())
    n_null_p   = int(df_model_scores.filter(F.col(p_col).isNull()).count())
    score_prof = profile_score(df_model_scores, p_col)
    score_profile_log[model_id] = {**score_prof, "n_null_p_emitida": n_null_p}

    mlflow.log_metrics({
        f"n_rows_per_model_{model_id}": n_scored,
        f"n_null_p_emitida_{model_id}": n_null_p,
    })

    df_for_model.unpersist()
    df_vectorized.unpersist()
    print(f"     n_rows={n_scored}  n_null_p={n_null_p}  score_mean={score_prof.get('mean', 'N/A'):.4f}")

# ── Metadados compartilhados da execução ──────────────────────────────────────
df_wide = (
    df_wide
    .withColumn("treino_exec_run_id", F.lit(TREINO_EXEC_RUN_ID))
    .withColumn("inf_versao",         F.lit(INF_VERSAO))
    .withColumn("mode_code",          F.lit(MODE_CODE))
    .withColumn("seg_inferida",       F.lit(SEG_TARGET))
    .withColumn("inference_ts",       F.lit(TS_EXEC))
)

# ── [3] Logs e métricas agregadas ─────────────────────────────────────────────
df_wide  = df_wide.cache()
n_output = int(df_wide.count())

mlflow.log_dict(profile_basic(df_wide, "output_wide_format"), "profiling_output.json")
mlflow.log_dict(score_profile_log, "score_profile_per_model.json")
mlflow.log_metrics({
    "n_rows_output":   n_output,
    "n_models_scored": len(MODEL_IDS_USED),
})

print("\n✅ Scoring concluído")
print("• n_rows output (wide format):", n_output)
print("• modelos:", MODEL_IDS_USED)
_show_cols = (
    [ID_COL]
    + [f"p_emitida_{m}"   for m in MODEL_IDS_USED]
    + [f"rank_global_{m}" for m in MODEL_IDS_USED]
)
df_wide.select(*[c for c in _show_cols if c in df_wide.columns]).orderBy(
    F.asc(f"rank_global_{MODEL_IDS_USED[0]}")
).display(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Salva output e encerra runs

# COMMAND ----------

df_wide.write.format("delta").mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_FQN)
mlflow.log_metric("output_saved", 1)

tables_lineage = {
    "stage":              "INFERENCIA",
    "ts_exec":            TS_EXEC,
    "mode":               MODE_CODE,
    "inf_versao":         INF_VERSAO,
    "seg_target":         SEG_TARGET,
    "treino_exec_run_id": TREINO_EXEC_RUN_ID,
    "model_ids":          MODEL_IDS_USED,
    "inputs":             {"input_table":  INPUT_TABLE_FQN},
    "outputs":            {"output_table": OUTPUT_TABLE_FQN},
}
mlflow.log_dict(tables_lineage, "tables_lineage.json")

print("✅ Output salvo:", OUTPUT_TABLE_FQN)

df_seg.unpersist()
df_inf_prep.unpersist()
df_wide.unpersist()

while mlflow.active_run() is not None:
    mlflow.end_run()

print("✅ Runs encerradas")
print("\n• output table     :", OUTPUT_TABLE_FQN)
print("• modelos inferidos:", MODEL_IDS_USED)
print("• colunas de score :", [f"p_emitida_{m}" for m in MODEL_IDS_USED])
print("\n⚠️  Anotar OUTPUT_TABLE_FQN para uso no 5_COMP_MODE_C:")
print(f"    INFERENCIA_TABLE_FQN = \"{OUTPUT_TABLE_FQN}\"")